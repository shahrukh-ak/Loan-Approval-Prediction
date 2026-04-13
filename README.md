# Loan Approval Prediction

Trains and benchmarks four binary classifiers to predict whether a loan application should be approved. The evaluation framework goes beyond accuracy, tracking training time, prediction speed, and serialised model size to support deployment decision-making.

## Business Context

For a lending institution, automated loan screening reduces manual review overhead. Choosing the right model requires balancing predictive accuracy with operational constraints like inference latency and memory footprint, which this project quantifies explicitly.

## Dataset

`loan_lead_data.csv` contains applicant-level features (financial, demographic, and behavioural) and a binary `Loan Approved` target variable.

## Methodology

**Preprocessing:** Categorical features are one-hot encoded with `pd.get_dummies`. A 60/40 train/test split is used with `random_state=42`.

**Class Balancing:** SMOTE is applied to the training set to correct class imbalance.

**Models:**
- Logistic Regression (max_iter=1000)
- Decision Tree (random_state=42)
- XGBoost (n_estimators=50, learning_rate=0.1, eval_metric="logloss")
- SVM (default RBF kernel)

**Metrics per Model:**
- Accuracy and full classification report
- Training time (seconds)
- Single-pass prediction time (seconds)
- Serialised model size (KB via pickle)

**Visualisation:** Side-by-side horizontal bar charts comparing all metrics across models.

## Project Structure

```
06_loan_approval_prediction/
├── loan_approval.py   # Full benchmarking pipeline
├── requirements.txt
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
imbalanced-learn
xgboost
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Place `loan_lead_data.csv` in the same directory and run:

```bash
python loan_approval.py
```

Outputs: a printed summary table and `model_benchmark.png`.
