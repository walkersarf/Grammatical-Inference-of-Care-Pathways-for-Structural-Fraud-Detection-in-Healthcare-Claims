# Grammatical Inference of Care Pathways for Structural Fraud Detection in Healthcare Claims

## 1. Install Synthea

Requires **Java 17** and Git.

```bash
git clone https://github.com/synthetichealth/synthea.git
cd synthea
./gradlew build -x test
```

## 2. Configure CSV export

Synthea writes FHIR by default. Edit `src/main/resources/synthea.properties`:

```properties
exporter.csv.export = true
exporter.csv.folder_per_run = false
exporter.years_of_history = 0
```

`folder_per_run = false` keeps output at a stable path.
`years_of_history = 0` exports complete patient histories rather than a recent window.

## 3. Generate the population

```bash
./run_synthea -p 5000 -a 40-90 -s 1 Massachusetts
```

| Flag | Purpose |
|---|---|
| `-p 5000` | population size |
| `-a 40-90` | age range — diabetes prevalence is far higher in older cohorts |
| `-s 1` | fixed seed for reproducibility |


Verify the output:

```bash
ls output/csv                             # conditions, medications, procedures, observations, encounters
grep -ci diabetes output/csv/conditions.csv    # should be in the thousands
```

If `conditions.csv`, `medications.csv` or `procedures.csv` are missing, the
population was too healthy to generate them — raise the age floor and regenerate.

## 4. Set up this project

```bash
git clone <this-repo>
cd <this-repo>
pip install aalpy pandas scikit-learn matplotlib
```

Point `data/` at Synthea's CSV output:

```bash
ln -s /path/to/synthea/output/csv ./data     # Windows: copy the folder, or use mklink /D
```

## 5. Run the pipeline

```bash
python src/inspect_csv.py data                                  # see what the data contains
python src/build_corpus.py --csv data --out results/corpus.pkl  # cohort filter + encoding
python src/inspect_corpus.py results/corpus.pkl                 # sanity-check lengths
python src/learn.py --corpus results/corpus.pkl --eps 0.1       # induce the automaton
python src/inject_eval.py                                       # inject fraud, evaluate
```

Outputs land in `results/` (`corpus.pkl`, `model.pkl`, `eval.pkl`) and
`figures/` (`induced_automaton.dot`, and `.pdf` if Graphviz is installed).

---

## Source files

| File | Role |
|---|---|
| `inspect_csv.py` | Dumps the most frequent `DESCRIPTION` values per table. Run first — it tells you what symbols the data actually contains. |
| `symbol_map.py` | **The alphabet.** Maps raw descriptions to event-role symbols; defines time-gap quantisation. The one file you tune by hand. |
| `build_corpus.py` | Filters to the diabetic cohort, merges all event tables into per-patient timelines, encodes them as symbol strings. |
| `inspect_corpus.py` | Corpus diagnostics: length distribution, alphabet frequencies. |
| `learn.py` | Splits by patient, runs ALERGIA, saves model and DOT export. |
| `mc_io.py` | Serialises AALpy's `MarkovChain` to a plain dict (it isn't directly picklable). |
| `score.py` | Anomaly score: `−log P(s)` under the learned chain; `inf` on impossible transition. |
| `inject_eval.py` | Injects fraud as string edits, scores clean vs. corrupted, reports ROC/PR. |

---

## How the encoding works

Each patient becomes an ordered string of event symbols with **time-gap terminals**
inserted between consecutive events:

```
START ENCOUNTER_AMB GAP_LE30 HBA1C_LAB GAP_LE180 MED_METFORMIN GAP_LE30 RETINAL_EXAM
```

Gaps are quantised as `GAP_LE30` (≤30 days), `GAP_LE180` (≤180 days), `GAP_GT180`.
Encoding time as *terminals* rather than as features means temporal violations
become ungrammatical, detectable by the same parsing mechanism as any other
structural violation.

Three encoding decisions materially condition results and are documented here
because they are the classical primitive-selection problem in syntactic pattern
recognition:

- **Consecutive duplicates are collapsed.** A dialysis patient treated 3×/week for
  years would otherwise generate thousands of identical repeats.
- **Histories are truncated to the first 60 clinical events**, modelling the
  diagnostic arc rather than lifetime billing volume.
- **High-frequency, low-information symbols are excluded** (routine glucose,
  recurring dialysis) — they flood strings without marking pathway steps.

### Fraud as string edits

Synthea generates only compliant episodes, so fraud is constructed. Each billing
fraud category maps to an elementary edit operation:

| Edit | Fraud type | Meaning |
|---|---|---|
| Substitution | Upcoding | Service billed at higher intensity than delivered |
| Insertion | Phantom billing | Service billed but never rendered |
| Expansion | Unbundling | One bundled code split into components, billed separately |

Because corruption is applied under our control, ground truth is exact and
attributable to a specific mechanism — which exclusion-list labels cannot provide.

---

## License

MIT. Synthea is licensed separately under Apache 2.0.
