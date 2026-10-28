# 🫀 Heart Failure Survival Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning classification project evaluating clinical records to predict mortality risk in patients with heart failure. The pipeline tackles severe class imbalance, prevents data leakage via robust Scikit-Learn pipelines, and optimizes prediction recall and F1-score for clinical viability.

---

## 📌 Project Overview
Heart failure is a prevalent cardiovascular condition characterized by high mortality rates. Timely and accurate risk stratification allows medical practitioners to prioritize critical care.

This project investigates clinical indicators predicting the target `DEATH_EVENT`. Given the severe real-world consequences of false negatives (predicting survival when a patient is at risk), optimization prioritizes **Recall** and **F1-Score** on the positive class rather than raw accuracy alone.

---

## 📊 Dataset & Features
The analysis is based on the **Heart Failure Clinical Records Dataset** (299 patients, 12 clinical features):
- **Source:** [Kaggle - Heart Failure Clinical Records](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)
- **Target Distribution:**
  - `0` (Survived): 203 patients (~68%)
  - `1` (Deceased): 96 patients (~32%)

### Feature Summary
- **Demographics & Habits:** `age`, `sex`, `smoking`
- **Clinical Comorbidities:** `anaemia`, `diabetes`, `high_blood_pressure`
- **Biochemical & Lab Markers:** `creatinine_phosphokinase`, `ejection_fraction`, `platelets`, `serum_creatinine`, `serum_sodium`
- **Follow-up:** `time` (monitoring period in days)

---

## ⚙️ Engineering & Modeling Workflow
1. **Exploratory Data Analysis (EDA):** Evaluated feature correlations, non-normal distributions, and class skewness.
2. **Leak-Free Pipelines:** Encapsulated transformations (`StandardScaler`) within `sklearn.pipeline.Pipeline` workflows across stratified train/test splits (80/20).
3. **Imbalance Mitigation:** Configured cost-sensitive learning via `class_weight="balanced"` to penalize minority class misclassifications.
4. **Hyperparameter Optimization:** Conducted 5-Fold Stratified Cross-Validation (`GridSearchCV`) optimizing specifically for `f1`.
5. **Model Evaluation:** Benchmarked traditional linear classifiers, distance-based learners, tree ensembles, and hard-voting combinations.

---

## 📈 Benchmark & Results

### Test Set Performance Summary

| Model | Accuracy | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: |
| K-Nearest Neighbors (KNN) | 70% | 0.21 | 0.31 |
| Decision Tree Classifier | 75% | 0.42 | 0.52 |
| Voting Classifier (Hard) | 78% | 0.47 | 0.58 |
| Logistic Regression (Balanced) | 80% | 0.58 | 0.65 |
| **Random Forest (Tuned)** | **85%** | **0.68** | **0.74** |

### Key Observations:
- **Best Model:** Tuned **Random Forest Classifier** (`max_depth=3`, `n_estimators=100`) yielded an **F1-Score of 0.74** and **Recall of 0.68** on unseen test data.
- **Cross-Validation Reliability:** Mean CV F1-score reached **0.784 (± 0.082)** on the training set, confirming model generalization without acute overfitting.

---

## 🛠️ Tech Stack & Requirements
- **Language:** Python
- **Core Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`

All dependencies are pinned in `requirements.txt`.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Leen-Bsharat/Heart-failure-classification.git
cd Heart-failure-classification
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Notebook or Standalone Script
**To run the interactive Jupyter Notebook:**
```bash
jupyter notebook heart_failure_analysis.ipynb
```
**To execute the standalone Python script:**
```bash 
python3 heart_failure_analysis.py
```

## 👤 Author

**Leen Bsharat**

* GitHub: [@Leen-Bsharat](https://github.com/Leen-Bsharat)
* Repository: [Heart-failure-classification](https://github.com/Leen-Bsharat/Heart-failure-classification)
