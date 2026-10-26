# PEL Framework Reproduction Toolkit

**Research-code companion for a Generative AI-Driven Personalized English Learning Framework**  
Release status: **methods and code toolkit, NOT an independent reproduction of the reported classroom findings**.

This repository supplies transparent *implementation paths* for the three proposed modules: **learner profiling**, **instructional exercise generation**, and **adaptive learning-path optimization**, plus validation of generated exercises, cluster-aware classroom analysis, six-condition crossover analysis, and descriptive reconstruction of Figures 4–8.

The original student-level/teacher-rating data were **not provided**. Do **not** interpret an example or run of this repository as having verified the manuscript's effect sizes, confidence intervals, randomization results, or expert assessment counts.

## What is included

| Component | Code | Input needed | What it can show |
|---|---|---|---|
| Model integrity | `scripts/verify_model.py` | online model hub access for `--online` | exact immutable SHA + official config validation |
| Content generation | `scripts/generate_exercises.py` | GPU + Qwen checkpoint | JSON exercises with structural filtering and prompt hashes; independent teacher review still needed |
| BERT proficiency | `scripts/train_proficiency.py` | licensed EFCAMDAT text + original 16-level metadata | learner-disjoint training and held-out classification metrics |
| Task performance | `scripts/train_task_model.py` | actual learner task records, NOT just public TOEFL questions | binary task-correctness prediction |
| Behavioral profiling | `scripts/train_behavior.py` | consented MOOCCube-derived canonical behavior extract | clustering + pseudo-label agreement, not gold learning-style accuracy |
| Affect fusion | `scripts/evaluate_affect.py` | paired text/audio scores + independent human labels | fusion and human-labeled macro F1; source emotion models must be separately documented |
| Grammar error prediction | `scripts/train_grammar.py` | independently error-labeled examples | supervised error-risk assessment |
| Learning path | `scripts/optimize_path.py` | prerequisite graph and normalized task estimates | exact shortest feasible plan for up to 16 tasks |
| Classroom statistics | `scripts/analyze_classroom.py` | real student-by-time records and 8 class assignments | pair-level confidence intervals and exact 4-pair randomization; optional mixed model |
| ITT sensitivity | `scripts/analyze_itt.py` | all randomized students, including missing post outcomes | 20-draw exploratory MAR multiple imputation and pooled paired-class estimates |
| Six-condition comparison | `scripts/analyze_ablation.py` | six observations per student, multiple parallel task versions | repeated ANOVA, partial eta² and corrected comparisons |
| Teacher quality assessment | `scripts/evaluate_exercise_quality.py` | 200 tasks × 3 strategies × 5 raters | four quality dimensions and inter-rater agreement |
| Figures | `scripts/plot_figures.py` | supplied summary means | charts without invented SD bars or significance stars |

## Reproduce the code environment

Python **3.10** is recommended. Install a virtual environment and either core analytics or GPU/model dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements-core.txt
python -m pytest -q
```

For model inference / training:

```bash
python -m pip install -r requirements-model.txt
python scripts/verify_model.py --online
python scripts/generate_exercises.py --cefr B1 --count 2 --words adapt review
```

The model download is ~15 GB and CUDA/BF16 support is expected for the specified Qwen inference setting. Do not commit any checkpoint to GitHub. Model loading is pinned to the **exact 40-character revision** in `configs/model.json`; the BERT backbone is pinned separately in `configs/components.json`. The Hugging Face `config.json` from the pinned Qwen revision is publicly inspectable at:

https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/4709f6c0771f0185a675b046268cdc1d1f2c74ce/config.json

> The `--online` verifier downloads only `config.json` and checks architecture fields; it does not download 15 GB of weights. Without `--online`, it validates the revision-string shape only and labels that check accurately.

## Run offline examples without proprietary research data

```bash
python scripts/verify_model.py
python scripts/optimize_path.py
python scripts/plot_figures.py
python -m pytest -q
```

The figure script only redraws means from the provided workbook in `data/public_summary/figures_4_8.csv`. It **does not estimate** standard deviations, mixed-model statistics or error bars from a spreadsheet that lacks individual observations. Figure 7 uses the workbook's IDs A/B/C/D; descriptive names in the manuscript draft have not been confirmed from the questionnaire. Figure 8's E/F/G identification follows the manuscript's descriptions, not the original workbook's column labels.

## Run with real, authorized data

See `data/schemas/SCHEMAS.md`. Place your licensed or consented CSVs **outside the Git repository or under ignored** `data/raw/`. Example commands:

```bash
# Split by learner_id (does not assume exact row count ratios)
python scripts/prepare_splits.py --input data/raw/efcamdat_authorized.csv \
  --output-dir data/processed/efcamdat --ratio 0.75 0.15 0.10

# Train appropriate task models on your own authorized extracts
python scripts/train_proficiency.py --split-dir data/processed/efcamdat
python scripts/train_behavior.py --split-dir data/processed/mooc_behavior
python scripts/train_task_model.py --split-dir data/processed/toefl_responses
python scripts/train_grammar.py --split-dir data/processed/annotated_grammar
python scripts/evaluate_affect.py --input data/raw/paired_affect_scores.csv

# Analyze independent teacher-class pairs, exactly 8 classes (4 instructor pairs)
python scripts/analyze_classroom.py --input data/raw/classroom_long.csv --mixed-model
python scripts/analyze_itt.py --input data/raw/classroom_long.csv --draws 20

# Repeated-measure evaluation: each student experiences all 6 conditions
python scripts/analyze_ablation.py --input data/raw/crossover.csv

# Blind expert ratings: same 200 tasks across 3 strategies, 5 raters
python scripts/evaluate_exercise_quality.py --input data/raw/teacher_ratings.csv
```

Run `python scripts/run_manifest.py --inputs data/raw/classroom_long.csv` to record the actual software environment and input hash. All computed outputs stay in ignored `results/`. After each run archive the input SHA-256, software configuration, code commit, executed commands and approvals in a separate restricted research record. Keep real teacher/student IDs pseudonymized even outside Git.

## Important statistical cautions

* **Eight classroom clusters:** methods cannot treat 235 students as 235 independent randomized units. A paired four-teacher design supports only `2^4=16` allocations; the minimum two-sided nonzero exact randomization *p* is 0.125. Exploratory multiple imputation uses a missing-at-random assumption and fixed class indicators; it does not implement fully multilevel joint imputation. The optional mixed model provides only **approximate** Wald intervals, not Satterthwaite or Kenward–Roger inference. The paired-class and exact statistics are separately reported.
* **Ablation:** the six-period short-task crossover is **not** a randomized 16-week comparison of six classroom treatments. Repeated ANOVA assumes sphericity; test it before making inferential claims or fit an appropriate repeated-observation mixed model with period/task-version covariates.
* **Labels:** TOEFL public questions do not supply a cohort of 15,000 participants; MOOCCube behavior clustering does not produce validated learning-style truth; a model-generated emotion pseudo-label is not an independent human gold standard.
* **Expert evaluation:** the code expects genuine rater decisions. It does not manufacture CELTA/DELTA credentials, ratings, confidence intervals or quality scores.
* **Security:** the repository intentionally omits learner data, faces/voices, model weights and institutional ethics documentation. The user must provide a real data-availability / IRB statement with approval status before release of research participant data.

## Repository map

```text
configs/                     Pinned model identifiers, algorithms and dataset registry
prompts/                     Full generation prompt text (versioned in Git)
src/pel_repro/              Reusable model-independent utilities
scripts/                    Runnable training, inference, statistical, and plotting CLIs
data/schemas/               Exact input CSV fields and label constraints
data/public_summary/        Manuscript source-workbook means (no invented variance)
tests/                      Offline unit and integration checks
examples/                   Nonhuman toy graph for path-planner smoke test
PROVENANCE.md               What is / is not supported by supplied materials
```

## Cite and use responsibly

See `PROVENANCE.md` and `REFERENCES.md`. The MIT license applies **only to this toolkit code**, not to Qwen weights, publisher datasets, public exam items, or any classroom recordings. This project is intended to enable a traceable reproduction workflow; obtaining independent experimental evidence remains the responsibility of the actual study investigators.
