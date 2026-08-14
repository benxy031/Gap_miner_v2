# Plan za više gapova s meritom 30+

Ovaj dokument sažima što sam provjerio iz lokalne baze, iz postojećih
cpugapminer dokumenata i iz službenih Gapcoin izvora na GitHubu, te pretvara
to u konkretan plan za kod, hardver i algoritam.

## Što sam provjerio

### Lokalna baza

- `gapchain.sqlite3` trenutno ima **768** zapisa s `merit >= 30`.
- Isti broj vrijedi i za `merit > 30`, pa nije riječ o inclusive/exclusive
  razlici nego o stvarnom trenutnom stanju baze.
- Najveći merit u bazi je oko **41.94**.
- Gruba raspodjela repa izgleda ovako:
  - `merit >= 21`: 2,237,927
  - `merit >= 25`: 127,353
  - `merit >= 30`: 768
  - `merit >= 35`: 3

### Lokalni repo materijali

- [docs/CRT_GENERATION.md](docs/CRT_GENERATION.md) pokazuje da CRT mode potpuno
  zaobilazi normalni sieve i da je za više merit ciljeve ključna kvalitetna CRT
  selekcija i dobar `gap_target`/`ctr-bits` par.
- [docs/CRT_PHASE_WORKFLOW.md](docs/CRT_PHASE_WORKFLOW.md) potvrđuje da je
  najsigurniji workflow: evaluacija postojećih CRT fileova, generiranje novih,
  selector, pa tek onda full launch.
- [docs/GEN_CRT_EXHAUST.md](docs/GEN_CRT_EXHAUST.md) potvrđuje da za male i
  srednje crt skupove `gen_crt_exhaust` može dati točan minimum, dok je za veće
  shiftove praktičniji `gen_crt`.
- [docs/sample_stride_guide.md](docs/sample_stride_guide.md) pokazuje da je u
  GPU non-CRT putanji optimalni stride obično oko 5–7 za target oko 21, a veći
  stride brzo povećava Phase-2 overhead.
- [README.md](README.md) i [pgt.md](pgt.md) potvrđuju da su ključne metrika
  throughputa `gaps`, `tested/s`, `accepted` i da se mora čuvati razlika između
  candidate throughputa i stvarno prihvaćenih rezultata.

### Službeni Gapcoin izvor

- Službeni [Gapcoin repo README](https://github.com/Gapcoin/gapcoin) potvrđuje
  da je projekt osjetljiv na kvalitetu testiranja i da su build/test discipline
  dio očekivanog workflowa.
- Službena [Gapcoin wiki](https://github.com/Gapcoin/gapcoin/wiki) trenutno je
  prazna, što znači da nemamo dodatnu službenu tuning dokumentaciju iz tog izvora.

## Zaključak iz nalaza

Ako je cilj više kandidata s meritom 30+, usko grlo nije samo sirovi pps. Za
ovaj rep baze najveći dobitak vjerojatno dolazi iz kombinacije:

1. bolje CRT selekcije za targete koji realno daju duže gape,
2. dubljeg, ali kontroliranog sieva tamo gdje GPU Fermat više nije bottleneck,
3. autotune profila koji nisu optimizirani samo za `tested/s`, nego i za
   pronalazak rijetkih visokomeritnih kandidata.

## Specifičan plan

### 1. Kod

#### 1.1. Uvesti merit-aware profiliranje

- U `src/main.c` dodati poseban profilni signal za visoke merit targete:
  - `merit>=30 hit rate`
  - `best merit per minute`
  - `accepted / tested`
  - `qual / pairs`
- Trenutni STATS već daje dobar temelj (`keep(odd)%`, `false_gaps`, `pps`,
  `tested/s`, `best`, `last_gap`, `last_qual_gap`), ali treba jedan agregat koji
  odgovara na pitanje: "koji profil proizvodi više kandidata 30+?"

#### 1.2. Autotune prebaciti s čistog throughputa na weighted objective

- Trenutni autotune već koristi `speed|balanced|quality`.
- Za high-merit lov treba dodati još jedan sloj ili mod koji daje bonus za:
  - veći `best merit`
  - veći broj `gap` događaja iznad 30
  - niži `false_gap` i stabilan `keep(odd)%`
- Ne treba rušiti postojeći speed path; treba dodati `high-merit` ili `quality+` cilj.

#### 1.3. Logirati per-profile tail učinak

- Za svaki autotune profil logirati:
  - `merit30_hits`
  - `merit35_hits`
  - `best_merit`
  - `tested/s`
  - `pps`
  - `keep(odd)%`
- Time se može izabrati profil koji je sporije, ali proizvodi više visokih
  meritova.

#### 1.4. Razdvojiti tuning za CRT i non-CRT

- CRT i non-CRT ne smiju dijeliti isti tuning prioritet.
- Non-CRT treba optimizirati za throughput + kvalitetu repa.
- CRT treba optimizirati za `ctr-primes`, `ctr-bits`, `gap_target` i izbor
  merit bandova.

#### 1.5. Dodati poseban "high merit campaign" preset

- U CLI dodati preset ili documented bundle tipa:
  - `--auto-tune-objective quality`
  - viši `sieve-primes`
  - stabilniji `sample-stride`
  - `--stats-verbose`
- Cilj je reproducibilan high-merit run, a ne samo maksimum kroz 5 sekundi.

### 2. Hardver

#### 2.1. Prioritet je više VRAM-a, ne samo više TFLOPS-a

- Za ovaj workload veći `sieve-size` i dublji GPU batching traže stabilnu VRAM
  rezervu.
- Ampere/Ada kartice s više VRAM-a daju više manevarskog prostora za:
  - veći batch,
  - dublji GPU sieve,
  - manje fallbackova.

#### 2.2. Za merit 30+ vrijedi testirati veće GPU_BATCH i veći GPU_BITS gdje ima smisla

- Ako GPU Fermat prestane biti bottleneck, vrijedi ispitati:
  - `GPU_BITS=1024` ili više za više shiftove,
  - veći batch ako VRAM to dopušta,
  - više GPU uređaja ako je host više-GPU.

#### 2.3. Ne hardcodati jednu karticu

- Ako se radi multi-GPU, raspodjela po uređajima mora ostati dinamična.
- Treba zadržati per-device pooling i round-robin/slot raspodjelu, jer high-merit
  runovi često traže dugi kontinuirani rad bez VRAM blowupova.

#### 2.4. Hardver za benchmark ciklus

- Preporuka za planiranje:
  - barem jedan GPU s 12 GB+ VRAM-a za ozbiljan non-CRT throughput test
  - po mogućnosti drugi GPU za paralelni CRT/selector test
  - dovoljno RAM-a i SSD prostora za duge logove i benchmark artefakte

### 3. Algoritam

#### 3.1. Fokusirati CRT selekciju na merit 30+ bandove

- S obzirom da trenutna baza ima samo 768 gapova iznad 30, CRT treba tretirati
  kao rijedak-event generator.
- To znači da treba pretraživati CRT fileove po merit bandovima i zadržati samo
  one koji povećavaju šansu za tail evente, a ne samo one s najmanje kandidata.

#### 3.2. Kombinirati dvije metrike

- Za svaki profil treba evaluirati:
  - `expected throughput`
  - `tail quality`
- `tail quality` ovdje znači: koliko često run daje `best merit >= 30`, `>=35`
  ili novi rekord.

#### 3.3. Za non-CRT koristiti empirijski optimum, ne teorijski maksimum

- [docs/sample_stride_guide.md](docs/sample_stride_guide.md) već pokazuje da je
  optimalni stride često oko 5, ne 1 i ne maksimalno dozvoljeni safe stride.
- Za merit 30+ treba čuvati taj princip: najbolji profil nije nužno onaj s
  najvećim `tested/s`.

#### 3.4. Dublji sieve samo do točke gdje ne guši Fermat

- Ako GPU Fermat već nije bottleneck, dublji sieve može pomoći.
- Ako dublji sieve smanjuje `tested/s` previše, tail output može pasti.
- Zato treba tražiti ekvilibrij, ne ekstrem.

#### 3.5. Poseban scoring za high-merit search

- Predlažem scoring funkciju za autotune:

```text
score = a * log(1 + merit30_hits) + b * log(1 + best_merit)
        + c * log(1 + gaps_above_30)
        - d * false_gap_rate
        + e * normalized_pps
```

- Težine `a..e` treba kalibrirati na realnim runovima.

## Preporučeni eksperimentalni redoslijed

1. Zaključati baseline na trenutnom najboljem profilu i zabilježiti
   `merit>=30` i `merit>=35` hitove po satu.
2. Pokrenuti non-CRT A/B testove s manjim skupom profila:
   - stride 4, 5, 6, 7
   - različiti `sieve-primes` oko trenutnog optimuma
3. Za svaki profil mjeriti:
   - `best merit`
   - `merit30_hits`
   - `false_gaps`
   - `pps`
   - `tested/s`
4. Tek onda proširiti na CRT presete za bandove koji daju bolji tail.
5. Nakon toga dodati high-merit autotune preset u kod.

## Predloženi radni zadaci u kodu

- Dodati high-merit stats counter i log line.
- Dodati merit-aware autotune objective.
- Dodati benchmark skriptu koja broji `merit>=30` i `merit>=35` po runu.
- Za CRT napraviti merit-band selector s naglaskom na tail performance.
- Nakon toga napraviti kratki regresijski test da se ne pokvari postojeći
  `pps`/`tested/s`/`false_gaps` izlaz.

## CRT-only preporuke za high-merit

Ovaj dio je namjerno ograničen samo na CRT putanju. Non-CRT putanje se ne
trebaju mijenjati za ovaj zadatak.

CRT tuning ovdje cilja **samo shift 450 i više**. Sve ispod 450 ostaje izvan
opsega ovog plana.

### 1. Promijeniti cilj selekcije CRT fileova za shift 450+

- Trenutni CRT workflow već ima dobar temelj u [docs/CRT_GENERATION.md](docs/CRT_GENERATION.md)
  i [docs/CRT_PHASE_WORKFLOW.md](docs/CRT_PHASE_WORKFLOW.md), ali za high-merit
  lov na shiftu 450+ treba rangirati fileove po tail ponašanju, ne samo po
  ukupnom broju kandidata.
- Prednost treba dati CRT fileovima koji daju bolje rezultate na merit bandovima:
  - 30–32
  - 33–35
  - 36+
- To znači da je `geo_mean_prob` koristan, ali nije dovoljan sam za sebe.
  Treba dodati dodatni weighted score koji favorizira rijetke, visoke meritove.

### 2. U `tools/eval_crt_merit.py` dodati high-merit scoring sloj za shift 450+

- Zadržati postojeći model za procjenu, ali nadograditi izlaz tako da se za
  svaki CRT file na shiftu 450+ izračuna:
  - broj pobjeda na merit 30+
  - broj pobjeda na merit 35+
  - prosječni/medijan `gap_target` tail score
  - broj bandova gdje file pobjeđuje na vrhu distribucije
- U praksi to znači da selector ne bira više samo "najmanje kandidata", nego
  "najveća šansa za 30+ i 35+".

### 3. Uvesti CRT band-weighted portfolio za shift 450+

- `tools/eval_crt_merit.py` već generira `phase3` selector i portfolio JSON.
- Za high-merit cilj na shiftu 450+ treba napraviti portfolio s težinama, npr.:
  - `merit 30–32` = osnovna težina
  - `merit 33–35` = viša težina
  - `merit 36+` = najviša težina
- Selector treba birati CRT file prema ciljanom merit bandu, ali i prema tome
  koji file najbolje pokriva gornji rep u prethodnim evaluacijama.

### 4. Preferirati `--fitness-mode probability` za CRT generiranje na shiftu 450+

- Kod generiranja novih CRT fileova (`tools/gen_crt.c`) treba favorizirati
  `--fitness-mode probability` kada je cilj high-merit rep na shiftu 450+.
- Razlog: candidate-count minimum ne mora biti najbolji za tail događaje.
  Probability mode bolje hvata distribuciju survivors oko širih gap zona.
- Candidate mode ostaje koristan za baseline i regresiju, ali ne treba biti
  primarni kriterij za high-merit CRT fileove.

### 5. Koristiti `gen_crt_exhaust` samo gdje ima smisla za shift 450+

- Za male i srednje CRT skupove gdje je prostor pretrage razuman, koristiti
  `gen_crt_exhaust` kao ground truth i benchmark.
- Za veće shiftove držati se `gen_crt`, ali uz hard filter:
  - kandidat mora pobijediti postojeći CRT file barem na jednom high-merit bandu
  - ili mora imati jasno bolji tail score, čak i ako je ukupni candidate count
    malo lošiji

### 6. CRT runtime treba ostati jednostavan za shift 450+

- U runtimeu ne treba dodavati dodatnu kompleksnost u non-CRT sieve.
- Za CRT na shiftu 450+ je važnije:
  - pravilan `--crt-file`
  - pravi `gap_target`
  - dovoljno dobar `ctr-primes` i `ctr-bits`
  - dobar selector koji je već pripremljen offline
- Drugim riječima: CRT kvalitetu treba izgurati u generiranju i selekciji,
  ne u runtime improvizaciji.

### 7. Preporučeni CRT high-merit workflow za shift 450+

1. Generirati ili skupiti CRT fileove za isti shift.
2. Pokrenuti evaluaciju po merit bandovima s fokusom na 30+.
3. Izbaciti fileove koji imaju dobar candidate count, ali slab tail.
4. Zadržati samo one fileove koji daju bolje 30+/35+ rezultate.
5. Generirati selector i run helper samo iz tog skupa.
6. Koristiti selector za ciljane high-merit kampanje.

### 8. Praktični CRT startovi za shift 450 / 512 / 768

Ovo su polazne točke za testiranje. Ne mijenjaju non-CRT putanje i služe samo
kao CRT tuning baseline za high-merit lov.

| Shift | Preporučeni start | Fokus | Napomena |
|------:|-------------------|-------|----------|
| 450 | postojeći `m30_phase3` file kao baseline | `30–35` | prvi CRT rep gdje high-merit band postaje ozbiljno isplativ; krenuti s `probability` fitnessom i phase3 selectorom |
| 512 | postojeći `m30_phase3` file kao baseline | `30–35` | pogodniji za dulje i stabilnije CRT portfolio testove; usporediti s `m23/m27/m30` bandovima |
| 768 | postojeći `m28` / `m30` bandovi kao baseline | `30+`, posebno `35+` | ovdje tražiti najjači tail; zadržati samo fileove koji pobjeđuju na gornjem repu, ne nužno na ukupnom candidate countu |

#### Preporučeni CRT redoslijed testiranja

1. Usporedi postojeće CRT fileove unutar istog shifta.
2. Generiraj nove varijante s `--fitness-mode probability`.
3. Zadrži samo one koji pobjeđuju na `30–32`, `33–35` ili `36+` bandovima.
4. Ponovno izgradi selector i run helper iz pobjedničkog skupa.
5. Testiraj na stvarnim high-merit runovima prije nego odbaciš stariji baseline.

## Napomena o broju gapova >30

- Trenutno stanje lokalne baze je **768**, ne 769.
- Ako negdje vidiš 769, vjerojatno je riječ o drugoj bazi, drugom snapshotu ili
  drugačijem brojilu.

## Kratka preporuka

Ako želiš samo najkraći odgovor: za više gapova s meritom 30+ treba ići na
`quality-first autotune`, jači i stabilniji GPU batch, te merit-band CRT
selekciju. Čisti throughput nije dovoljan jer je rep iznimno rijedak.
