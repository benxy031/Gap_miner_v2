# Usporedba: GapMiner-Horizon Golden RC1 vs gapminer_v2

Datum: 2026-09-05. Izvor: `/home/dejan/Git/GapMiner-Horizon-Golden-Intermediate-RC1-20260904`
(content-addressed RC, **nije frozen** dok operator ne završi live test).

---

## 1. Što je Horizon

Horizon je **multi-coin mining suite** s Qt 6 UI-jem ("Frontier") koji kontrolira tri
coin-prostora:

| Metal | Coin | PoW | Status u RC1 |
|---|---|---|---|
| Golden | **Gapcoin** | prime-gap merit (naš prostor) | stabilan "exact S512/M22 production tuple" |
| Silver R202 | **Riecoin** | prim-constelacije (GPU, sm_86) | 16/16 accepted shares (arhivska kvalifikacija) |
| Bronze R167 | **Riecoin** | prim-constelacije (CPU) | 1.3736× vs interni referent; **claim "brži od službenog rieMinera" je RETRACTED** |

Layout: `source/gapcoin-golden` (Gapcoin miner, fork originalnog gapminera Jonnyja
Freya), `source/riecoin-silver-r202` / `riecoin-bronze-r167` / `riecoin-network-backend-*`,
`source/horizon-ui` (Qt6), `third_party/{cgbn,gmp}`, `windows-x64/` + `linux-ubuntu-24.04-x86_64/`
runtime-i, `evidence/` (nepromjenjivi kvalifikacijski računi), `SHA256SUMS.txt`.
`tools/` u `gapcoin-golden` je R&D groblje (desetine e03c/e03d eksperimenata — vidi §8).

**Release inženjering** (nama nedostaje): content-addressing, SHA256SUMS za svaki
publikirani fajl, `evidence/` računi s retrakcijama, fail-closed hardverska granica
(Windows: točno `sm_86`; Ubuntu: `sm_86` + PTX compute-7.5/8.6; sve ostalo fail-closed),
FNV allowliste za pinane artifakte (OCC-zero payload), "LOCKED" stanja UI-ja na
identity/config/hardware mismatch.

---

## 2. Gapcoin Golden — arhitektura host/plugin

Host (`Miner.cpp`, `main.cpp`) je originalni gapminer proširen s:

- **GPU plugin ABI v2** (`gpu/plugin/v2/gapminer_gpu_executor_abi_v2.h`): fiksna x64
  C/POD granica host ↔ `gapminer-cuda.{so,dll}` (dlopen). Limit: hash 8×LE32,
  max shift 1024 bita, adder 32 riječi, 40 riječi kandidat, 16 zapisa/rezultat,
  CRT scan span ≤ 2^20, ≤65535 CRT redaka/batch. **Strogo fail-closed**: zero-filled
  payload ne smije proizvesti kandidata (EMPTY=1, NONEMPTY=0 bitovi), nepoznati
  flagovi se odbijaju. Plugin NIKAD ne vraća share/primality dokaz — samo sirova
  opažanja ili negotiated "pinned-GMP630 PRP verdikt".
- **Geometry export**: host šalje primorijal (modulus), CRT offset, interval size,
  offsets kandidata (32-bit riječi) — GPU interno radi **row-walk po translate-ima**
  (`hazard` = redak m: `start = (hash<<shift) + offset + m·P`).
- Host-side obavezni `PoW::valid()` gate (GMP `mpz_probab_prime_p(n,25)` +
  `mpz_nextprime`) prije dispeča — GPU je filter, ne autoritet.

**Bitna arhitektonska razlika:** Horizon GPU miner hoda **translate retke unutar
jednog headera u batchu** (hazard batch). Naš `worker_thread_run_crt` radi **jedan
aligned prozor po header nonce-u** (SHA256 po prozoru) — row-walk postoji kod nas
samo u `gap_hunt` modu, ne u mining CRT putu.

---

## 3. CUDA srž — `CudaCrtPipeline.cu` (12.829 linija)

### 3.1 Konfiguracija

- `kBlockThreads = 128`; CGBN **768-bit, TPI=4** (S512 kandidati = 256+512 bita).
- Legacy prozor 4872; max prozor 2^20. Legacy CRT prime count 14.
- Autotune limiti: sieve primes do 20M (hard max 50M), 6 autotuner dimenzija:
  sieve core, fermat variant, CGBN TPI, sieve depth, hazard batch, CRT tail round.
- **OCC-zero S512 tail-16M** (produkcijski): 16M sieve primova, tail od 1M, **74
  CRT prima, 761 redaka, scan-end 11319, zadnji prim 295.075.147** — pinan FNV
  allowlistom.
- Endpoint predikat: **GMP-6.3.0-stil BPSW** (jaki MR + jaki Lucas, Selfridge D,
  extra MR baza iz svježeg default rand-state-a) — reimplementiran u CUDA tako da
  je **bit-kompatibilan s GMP 6.3.0 `mpz_millerrabin(n,25)`** (priprema na hostu:
  square gate, Selfridge D, GMP redoslijed).

### 3.2 Sito (markiranje)

- **Prime-major preko hazarda**: kernel `mark_composites` prima `base_residues[p]`
  i `step_residues[p]` pa računa `residue = (base + m·step) mod p` za redak m —
  isti `CombinedCrt` matematički model kao CPU strane: `N(m,x) = base + m·K + x`.
- `mark_composites_combined_euclid`: za `p > window` koristi **Euclidean
  modulo-event skipping** (`modulo_search_euclid_u32`, stack-based CRT-Euclid
  pretraga) — preskače hazarde u kojima prim ne pogađa, umjesto da po redu
  računa `% p`. Ovo je Seth Troisi-jev trik (Apache-2.0, atribucija očuvana).
- `prepare_batch_residues` / `prepare_static_residues_{division,reciprocal}`:
  residue-i se računaju **na GPU-u**, batch-per-hazard; reciprocal varijanta bez
  divizije.
- Kompakcija preko **CUB** (`device_radix_sort`, `device_reduce`, `device_scan`) —
  `count_candidates`, `compact_candidates_indexed`, `count_marked_candidates`.
- "Bitmap frontier" i "sparse CRT macro-sieve cache": planirani epochovi,
  kontrolni D2H prijenosi se broje zasebno (`bitmap_frontier_*` rate metrike).

### 3.3 Fermat/Euler filter

- **Baza-2 Fermat s eksponentom (n−1)/2 + Euler–Jacobi gate**
  (`passes_base_two_euler_jacobi`): ista semantika kao naš CPU Euler filter,
  ali u CUDA.
- CGBN varijante:
  - `fermat_candidates` — binary ladder, **fast-base-2 trik**: za full-width
    modul (R/2 ≤ n < R) `R − n` je Montgomery-jedan bez konverzije; svaki
    postavljeni bit košta **jedno kanonsko modularno dupliranje** (`double_modulo`),
    bez množenja tablicom.
  - `evaluate_base_two_windowed` — fixed-window (WindowBits, shared-window
    varijanta kroz smem), "nounroll" po windowu.
  - `evaluate_base_two_binary` — opt-in diferencijalni kandidat.
  - `native_cios_*` (iza `GAPMINER_CUDA_NATIVE_CIOS_CANDIDATE=0`): port
    pscamillo/mr_blackwell **Montgomery CIOS PTX primitiva** (24 limbova/768-bit,
    TPI=8) — usporedni kandidat, default OFF.
- Posebni kernel `classify_gmp630_prepared_prp_kernel` (MR+Lucas, GMP-6.3.0
  kompatibilan), `filter_crt_blockers_base3_mr_kernel` (baza-3 MR za blokere).

### 3.4 M19 adaptive prime net / MinimumMerit frontier

- **Post-sieve egzaktni scheduler za M19 (merit-19) cilj**: za verificirane proste
  L < R i `T(L) = even_ceiling(19·ln L)`: ako `R−L < T(L)`, unutrašnjost
  **provjerljivo ne treba eksponencijacije** (gap merit < 19). Inače se proba
  **najdalji survivor koji može zatvoriti ispod-M19 interval** ("right-cover
  order" — izbjegava ~2× prekomjerno certifikaciju midpoint stabala).
- Pali baza-2 Fermat je **zvučan** kompozitni verdikt; survivor nikad nije
  tretiran kao prim bez testa. Multi-net scheduler round-robina nezavisne
  mreže u jedan device batch.
- Ovo je formaliziranija verzija Kehrig skoka od našeg jump2: certifikat
  "nema testa" umjesto chain-state-a. Cilja **M19 (mining difficulty)**, ne
  record režim.

### 3.5 Progressive CRT tail + PACER

- `count/compact/resolve/mark/finalize_progressive_tail_round`: rep sita iza
  pokrivača rješava se u rundama (progressive) umjesto jednim prolazom.
- **PACER sparse tail-factor calendar** (S512, 512 redaka, minimalni tail
  faktor 1M): kalendar događaja po prostim faktorima repa — `sparse_tail_calendar`
  (Euclidean floor_sum + rank-select), `tail_sorted_tape` (radix-sort
  reduce-by-key), `tail_persistent_wheel` (2^19 slot event wheel).

### 3.6 Autotune + verifikacija u samom pipeline-u

`run()`: `initialize()` → 6× autotune → **`verify()` protiv GMP oracle**
(mismatches ⇒ EXIT_FAILURE) → warmup → mjerenje s bogatim Rate-om
(real_fermat_tests/s, probable_pps, sieve positions/s, modeled hazard/s,
qualified gap candidates/s, gpu_stage_wall_fraction...).

### 3.7 `CUDAFermat.cu` (legacy, 4.210 linija)

Vlastita BigInt (320/768-bit?) implementacija s `montgomery_mul_cios`,
`monMul320_words`, trace kernelima i `quick_composite` (mali primovi u
`__constant__`) — legacy non-CRT GPU put; ekvivalent našeg skalarnog
`fermat_kernel_t` puta, ali u 32-bitnim limbovima.

---

## 4. CRT coveri

- Produkcija: **`crt-22m-512s-761-verified.txt`** — 74 prima, size 11319,
  761 kandidat, offset 198-znamenki (S512/M22 tuple). Varijante:
  `crt-25m-512s-wizz`, `crt-30m-512s`, `crt-40m-512s-experimental`.
- Auto-discovery prihvaća **samo `-verified` fajlove**; CRT prioritetni model =
  Poisson `valid_probability / expected_tests` na merit 22.
- Naši coveri iste klase: `shift507_p74_*` (74 prima) — isti red veličine;
  mi imamo lex/strong/objective varijante i m23..m40 raspon; oni pinaju
  jedan verified tuple po izdanju.
- Generatori: `ctr-evolution` (originalni evolucijski, mutacija do **pune
  enumeracije parova primova**) + `CrtOffsetOptimizer` (greedy residue +
  local search, coverage counteri). Naš `gen_crt` je greedy + local sweep —
  njihov pair-enumeration mutation level je jači alat (mi to nemamo).

## 5. CPU strane (Golden)

- `ChineseSieve.cpp`: CRT redovi s **priority queue best-first schedulingom** —
  prioritet = `hazard po očekivanom Fermat testu` (Poisson model valid
  probability); queue limit 10000, batch rows 64. **Best-first smanjuje
  očekivano vrijeme do prvog valjanog gapa** (ne rate, ali latency do share-a).
- Fermat backend-i: gmp / openssl / openssl-word / **avx512-ifma** (8 i 16
  laneova, in-tree Clang kernel; AMS30 multi-buffer default OFF).
- `CpuCrtAutotuner`: profil **v16**, FNV-1a fingerprint (CPU + IFMA + shift +
  threads + CRT hash + merit bucket 0.25), tunira 5 dimenzija; sidra za
  sieve primes po shiftu (129/512/1024).
- `CombinedCrtSieve`: bit-za-bit ekvivalentan originalu + `sparse_combined_mark`
  (Euclid skip) + POPCNT ekstrakcija.
- Non-CRT put: klasični Eratostenov bitmap + pseudo-wheel 3/5/7 + baza-2 Fermat
  (OpenSSL BN). Euler-Jacobi filter default **off**.
- Autotune lowering: `crt` auto kad shift ≥ 129.

## 6. Riecoin strane (kontekst)

- Silver R202 (GPU): jedan dugački `.cu` lanac (`riecoin-rlow-e2e-r202-...1148...`),
  CGBN, sm_86 + compute-7.5 PTX; mjereno **770.6M positions/s, 675.6k PRP/s**
  na 1150-bit targetu, 16/16 accepted shares (pool: Stelo.xyz:2005).
- Bronze R167 (CPU): reciklira službeni rieMiner-ov ISPC asm; offline 1.3736×
  vs interni referent, ali službeni je **1.698× CPU-efikasniji** — poštena
  retrakcija u evidence.
- Stratum: Gapcoin = legacy JSON-RPC getwork protokol (jansson, `mining.request/
  submit`, `blockchain.block.new` push); Riecoin = puni stratum v1 (extraNonce,
  difficulty-offset, 30 s ACK, SecureZeroMemory za credentiale).

## 7. Usporedna tablica

| Dimenzija | Horizon Golden | gapminer_v2 |
|---|---|---|
| Scope | 3 coina + Qt6 UI + release inženjering | Gapcoin miner + gap_hunt record lovac, CLI |
| CUDA primality | CGBN 768-bit TPI=4, baza-2 Fermat+Euler-Jacobi, fixed-window/binary, fast-base-2 trik (R−n = Mont-1), CIOS PTX kandidat | CGBN AL=2..20 TPI=4/8, MR baza-2 (CGBN) + scalar fallback, sliding window WIN_BITS=3 |
| Endpoint verifikacija | GMP-6.3.0-pinned BPSW (MR+Lucas, Selfridge D) — bit-kompatibilna s `mpz_millerrabin(n,25)` | Baillie–PSW (`primality_bpsw`) |
| Sito (CRT) | prime-major preko hazard-batcha + **Euclid skip za p>window** + CUB radix/scan | per-prozor residue kernel + mark kernel; ekstrakcija single-block ordered scan; nema Euclid skip-a |
| Row-walk (mining) | GPU hoda translate retke (hazarde) po headeru | mining CRT = 1 prozor po header nonce-u (row-walk samo u gap_hunt) |
| Scheduling | priority queue best-first + Poisson hazard model (CPU) | FIFO batch, 2 flighta |
| Autotune | 6-dim CUDA + CPU autotuner v16 s fingerprint profilima | env knobs + ad-hoc A/B |
| Skok/no-test | M19 adaptive prime net (certifikat „nema testa"), right-cover order | **MINING_JUMP2**: jump2 chunk-parallel Kehrig lanac u mining fused putu (+51% na shift507, 1 thread) + jump2 u gap_huntu |
| Cover generator | evolucijski s par-enumeracijom + greedy optimizer | greedy + local sweep |
| Verifikacija u runtime | `verify()` vs GMP oracle prije svakog runa | parity harnessi + test_gap_hunt vanjski |
| CPU primality | gmp/openssl/openssl-word/**AVX512-IFMA** (8/16 lane) | GMP default + port limb-Montgomery put (sporiji od GMP 6) |
| UI/ops | Qt6 dashboard, canary gateovi, receipts, SHA256SUMS, fail-closed sm_86 | CLI, git, testovi |
| Record hunting | **nema** | gap_hunt + jump2 (3×), watcher, σ analiza (tail_compare) |

## 8. R&D eksperimenti (tools/ — što su istraživali)

- **e03d-rns-***: **fixed-base RNS** (residue number system) eksponencijacija:
  division-free 16-bit RNS redukcija, warp-parallel MMA tenzori (C0..C4),
  recip-window16 stream, lazy multiwarp (lazy carry + CRT korekcijski bit).
  Ruta istražena umjesto Montgomery-ja — produkcija ostala na CGBN.
- **e03d-mr-blackwell-cios-euler-bridge**: A/B CGBN vs pscamillo/mr_blackwell
  CIOS PTX primitivi.
- **e03d-pacer-***: kalendar rijetkih tail-faktora (Euclid floor_sum, radix
  sort tape, 2^19 event wheel).
- **e03c3-***: minimum-merit frontier falsifieri (CGBN vs 2160-modulus C4 fixture).
- **OCC-zero**: S512 tail-16M prime-elision artifact (16M prima, 761 redaka).
- **gpu-crt-***: replay/contract testovi ABI-ja (probe vectors, empty-row
  regresije, void-field bench sa 74 prima).
- **avx512-***: IFMA kernel A/B vs GMP (CPU).
- **benchmark-cpu-crt-five-way.ps1**: official / crtwizz / R2 / R3 / current A/B
  na shiftovima 129/512/1024 (CPU).

## 9. Što vrijedi uzeti od njih (prioritizirano)

1. **Euclidean modulo-event skip u GPU situ** (`sparse_combined_mark` /
   `mark_composites_combined_euclid`): za prime p > window, umjesto
   per-prozor `% p` za svih 10–20M primova — batch residue + skip. Direktna
   ušteda u našem fused CRT putu (mark je danas velik dio 1.7 ms/prozor).
   **Status: IMPLEMENTIRANO (2026-09-05)** — u `gpu_sieve_rows_mark_kernel`
   (fused CRT row-batch put, default ON): sparse primovi (p > window) koriste
   `gpu_sieve_modulo_event_euclid_u32` (port `modulo_search_euclid_u32`) i
   skaču preko no-event redaka; parni korak = jedna lattica, neparni = dvije
   (fo flip); p|P = fiksne grane. Parity: 100k model slučajeva + prošireni
   `test_gpu_sieve.c` (100k prime tablica, obje parnosti, step 30030/255255).
   **A/B (shift507, 4T, FUSED_GPU=1, rows=64 vs 0)**: 50k: +1.8%, 500k:
   +3.0%, 2M: +5.6%, 5M: **+10.9%** (2412 vs 2174 win/s) — win raste s
   dubinom sita, 0 merit kandidata (čisto). Per-window single-mark put nije
   mijenjan.
2. **Row-batch (hazard) walk u mining CRT putu**: GPU hoda translate retke
   jednog headera — eliminira SHA256-per-prozor limit našeg
   `worker_thread_run_crt`. Potencijalno velik win za CRT mining.
   **Status: IMPLEMENTIRANO (2026-09-05)** — `CRT_ROWS_BATCH` (default 64,
   ON bez env; 0 = off): row-walk `base + m·P` po nonceu (worker_gpu.c) +
   `gpu_sieve_mark_rows_from_base` (jedan residue prolaz za base+P, svi retci
   jednim kernelom; ekstrakcija preko eksplicitnog bitmap pointera;
   tail re-mark u arenu red 0). Parity testovi u test_gpu_sieve.c (parni i
   neparni korak). **Mjereno: NEUTRALNO** (shift507 FUSED_GPU=1: 2319 vs
   2348 win/s; hybrid: 1543 vs 1542) — per-window residue nisu bili
   bottleneck. Radi s HALF_CLASS i QUARTER_CLASS (per-row base_mod60).
   Bug ulovljen tijekom rada: P (primorial) je PARAN za neke covere → flip
   `fo^(m&1)` po retku je bio kriv za neparne retke → false gapovi do 45
   merit; fix: parnost koraka eksplicitno iz `step_limbs[0]`.
3. **Priority-queue best-first + Poisson hazard model**: ne diže rate, ali
   reže očekivano vrijeme do prvog valjanog share-a (bitno za pool mining).
4. **AVX512-IFMA Fermat backend** za CPU-only hostove (naš limb put je sporiji
   od GMP 6; IFMA je njihov odgovor).
5. **M19-stil „no-test" certifikati** kao sljedeća evolucija jump2 (right-cover
   order) — kod nas bi cilj bio min_merit umjesto 19.
   **Status: IMPLEMENTIRANO (2026-09-06)** — `MINING_JUMP2` (env, off default)
   u fused mining putu (`worker_gpu.c`): port jump2 chaina (chunked MR runde
   preko K prozora po letu, backward search, threshold po prozoru
   `ceil(merit · ln(window_base))`), K default 64 (1..128), chunk 64.
   Lanac staje na merit-frontieri (preskočena unutrašnjost je provjerljivo
   ispod praga), sparse `is_prime` view → postojeći `crt_scan_gaps` netaknut;
   gap-dist health se isključuje (frontier cut distorzira male gapove, isto
   kao HALF_CLASS). Fail-closed: greška lanca → full scan istog leta → CPU
   put. `MINING_JUMP2_VERIFY=1` (dev-only) full-skana svaki let i uspoređuje
   emitirane merit-kandidate: **0 razlika** na shift507 (2370 letova) i
   shift998. Ograničenje: zahtijeva `--threads 1` (gather staging je
   per-kontekst). **Mjereno (RTX 3070, shift507 live merit, 1 thread):**
   K=64 **2953 win/s vs 1961 full-scan (+51%)**; K=32 2514, K=128 3115;
   single-thread lanac > 4-thread full-scan (2319). Class modovi rade:
   lanac hoda po class-filtriranoj listi (preskočeni parovi su provjerljivo
   ispod praga, kvalificirani par emitira FIND_END faza), VERIFY 0 razlika
   za HALF_CLASS i QUARTER_CLASS; mjereno 1 thread shift507: HALF 1973 →
   2630 (+33%), QUARTER 928 → 1018 (+9.7%).
6. **Autotuner s fingerprint profilima** (v16) umjesto env knobs — udobnost,
   ne brzina.
7. **Release disciplina**: SHA256SUMS, evidence računi, fail-closed ABI/PTX
   granice — nama nedostaje ako ikad shipamo binarne artefakte.
8. **CUB radix-sort kompakcija** ako naša single-block ordered-scan ikad
   postane limit (sada nije — namjerno smo je odabrali zbog reda).

## 10. Što je bolje kod nas (ne dirati)

- **gap_hunt + jump2**: oni nemaju ništa slično record-hunting modu (3.0×
  mjereno, egzaktno validirano).
- Jednostavnost: ~10k linija C99 + CUDA vs 1588 fajlova, ABI slojevi, plugin
  ugovori, 3 coina.
- Naš fused async double-buffered pipeline + GPU_MR_BATCH je produkcijski
  dizajn (njihov CUDA pipeline je benchmark-orijentiran, plugin ga hrani).
- HALF/QUARTER_CLASS containment mašinerija s on-demand rezolucijom — oni
  imaju M19 no-test intervale (srodno, ali drugi sloj i drugi cilj).
- Validacijski alatni lanac: test_gap_hunt, verify_gap_candidate.py,
  tail_compare (σ analiza), parity harnessi.

## 11. Zaključak

Horizon je **release-inženjerski zreliji i širi** (3 coina, UI, računi, PTX
discipline), a njegov CUDA CRT pipeline ima **dva algoritamska aduta koja smo
obojicu preuzeli** (2026-09-05): Euclid-skip markiranje i hazard row-walk u
mining modu (oba neutralna do +11% ovisno o dubini sita). U **čistoj
Gapcoin GPU aritmetici smo ravnopravni** (CGBN 1280-bit TPI=8 vs njihov
768-bit TPI=4; oba koriste CGBN Montgomery; oni imaju fast-base-2 trik koji
kod S512 štedi množenja tablicom). U **record huntu smo ispred** (gap_hunt +
jump2 + σ alat). Najveći konkretni dobici za nas: točke 1 i 2 iz §9.
