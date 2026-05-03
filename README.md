# MASC Replication — Phase 2: Improvisation

This repository contains the "Improvisation" phase of the MASC Mobile App Screen Classification project. In this phase, we moved beyond the baseline replication (Phase 1) and implemented methodological improvements grounded directly in the Data Science course curriculum.

## Difference Log: Phase 1 vs Phase 2

The core engineering effort in this phase involved taking the original replication code and applying formal Data Science optimization techniques.

| Component | Phase 1 (Replication) | Phase 2 (Improvisation) |
|---|---|---|
| **Codebase** | Monolithic `masc_classification.py` | Streamlined into a modular pipeline in `masc_optimized.py` |
| **Algorithms Used** | 10 generic ML classifiers | Strictly focused on the **4 Course Algorithms** (Logistic Regression, MLP, Decision Tree, Naive Bayes) |
| **Feature Scaling** | None (numeric element counts fed directly into models) | Applied `StandardScaler` to the 10 numeric structural features |
| **Text Engineering** | TF-IDF using only unigrams (single words) | TF-IDF using unigrams and bigrams (`ngram_range=(1,2)`) |
| **Class Imbalance** | Ignored | Applied `class_weight='balanced'` in Decision Tree and Logistic Regression |
| **Hyperparameter Tuning**| Default parameters from `scikit-learn` | Systematic `GridSearchCV` implemented to find optimal model configurations |
| **Evaluation Strategy** | Single run | 4-step Ablation Study evaluating the impact of every individual change |

## Repository Structure

```
masc_improvisation/
├── code/
│   └── masc_optimized.py       # The improved classification pipeline with GridSearchCV
├── data/processed/
│   ├── Labels.csv              # Class labels (from Phase 1)
│   └── MASC_Features.csv       # Extracted features (from Phase 1)
├── figures/
│   ├── ablation_study.png      # Chart showing incremental improvements
│   └── ablation_results.csv    # Raw metrics for all 4 stages
├── Improvisation_Report.md     # Detailed formal report of the methodology and results
└── README.md                   # This difference log
```

## How to Run

```bash
# Move to the code directory
cd code

# Run the optimization pipeline
python masc_optimized.py
```
*This will execute the full 4-stage ablation study, run GridSearchCV over the algorithms, and generate the charts and CSVs in the `figures/` directory.*
