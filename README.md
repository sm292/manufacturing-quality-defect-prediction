
# Manufacturing Quality & Defect Prediction

An end-to-end industrial Data Science project that applies Machine Learning to semiconductor manufacturing quality data to identify potential production failures and support quality inspection decisions.

Built with Python, Scikit-learn, Pandas, Plotly, and Streamlit.

## Project Overview

Manufacturing defects can increase production costs, waste materials, and reduce product reliability. Machine Learning can help quality teams identify patterns in historical production sensor data and prioritize observations that may require further inspection.

This project uses the UCI SECOM dataset to develop a classification pipeline that predicts the likelihood of a manufacturing failure from sensor measurements.

The project includes data preparation, missing-value handling, feature selection, model training, validation-based threshold selection, test-set evaluation, and an interactive dashboard.

**Important:** This is a research and portfolio project. Predictions are not a substitute for physical inspection or validated industrial quality-control procedures.

## Project Objectives

- Develop a Machine Learning model to classify manufacturing observations as pass or fail.
- Handle missing values and remove constant sensor features.
- Select informative sensor variables from a high-dimensional dataset.
- Address class imbalance in manufacturing failure detection.
- Evaluate performance using metrics beyond accuracy.
- Visualize failure probabilities, model performance, and sensor importance.
- Provide downloadable prediction results through a Streamlit dashboard.

## Key Features

### Data Processing
- Load historical semiconductor manufacturing data.
- Handle missing sensor measurements using median imputation.
- Remove constant features.
- Select informative variables using statistical feature selection.

### Machine Learning
- Random Forest classification.
- Class-weight balancing to help address the minority failure class.
- Validation-based decision-threshold selection.
- Chronological holdout evaluation when the dataset timestamps are available and valid.

### Interactive Dashboard
- Manufacturing quality overview.
- Failure detection and missed-failure counts.
- Interactive confusion matrix.
- Precision, recall, F1-score, balanced accuracy, and ROC-AUC.
- Failure-probability distribution.
- Sensor feature-importance charts.
- Filterable inspection records.
- CSV exports.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning and evaluation |
| Joblib | Model serialization |
| Plotly | Interactive visualizations |
| Streamlit | Dashboard application |
| Matplotlib | Optional additional visualizations |
| Seaborn | Optional statistical visualizations |

## Dataset

**Dataset:** SECOM — Semiconductor Manufacturing Data

**Source:** UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/179/secom

**DOI:** https://doi.org/10.24432/C54305

The dataset contains:

- 1,567 manufacturing observations.
- Approximately 590 sensor features in the supplied feature file.
- 104 failed observations and 1,463 passing observations.
- Missing sensor values.
- Binary manufacturing quality labels.

The failure class represents approximately 6.64% of all observations, making this an imbalanced classification problem.

### Dataset Files

The download script retrieves the following files:

- `secom.data` — sensor measurements.
- `secom_labels.data` — manufacturing outcome labels and timestamp information.

These files are stored locally in `data/raw/` and should not be committed to GitHub unless the dataset licence and repository policies permit redistribution.

## Project Structure

```text
manufacturing-quality-defect-prediction/
│
├── app.py
├── download_data.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── raw/
│       ├── secom.data
│       └── secom_labels.data
│
├── models/
│   └── quality_model.joblib
│
└── outputs/
    ├── metrics.json
    ├── test_predictions.csv
    ├── confusion_matrix.csv
    ├── feature_importance.csv
    └── threshold_tuning.csv
```

Generated files appear after the dataset has been downloaded and the training script has completed successfully.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd manufacturing-quality-defect-prediction
```

Alternatively, open the repository directly in GitHub Codespaces.

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Download the dataset

```bash
python download_data.py
```

This downloads the required UCI files into `data/raw/`.

If downloading fails, visit the official UCI dataset page and download the files manually.

### 4. Train the model

```bash
python train_model.py
```

The script trains the model, selects a decision threshold using validation data, evaluates the final holdout set, and saves the resulting model and reports.

### 5. Launch the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal. In GitHub Codespaces, open the forwarded port, usually `8501`, from the **Ports** panel.

## Machine Learning Workflow

1. **Data loading:** Load sensor readings and pass/fail labels.
2. **Data validation:** Verify feature and label row counts.
3. **Chronological ordering:** Sort by timestamps when all timestamps are valid.
4. **Data splitting:** Reserve the final 20% of observations for holdout testing.
5. **Validation split:** Use a stratified split of the earlier training period for threshold selection.
6. **Preprocessing:** Impute missing values and remove constant features.
7. **Feature selection:** Use ANOVA-based `SelectKBest` to select up to 40 sensor features.
8. **Model training:** Train a class-weighted Random Forest classifier.
9. **Threshold selection:** Choose the threshold that maximizes validation F1-score over the configured candidate range.
10. **Evaluation:** Calculate final test-set classification metrics.
11. **Export:** Save the model, predictions, confusion matrix, feature importance, and metrics.
12. **Dashboard:** Explore results and inspect the impact of different decision thresholds.

## Evaluation Metrics

Accuracy alone can be misleading when failures are uncommon.

| Metric | What it measures |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Balanced accuracy | Average recall across the two classes |
| Precision | Proportion of flagged observations that are actual failures |
| Recall | Proportion of actual failures detected |
| F1-score | Balance between precision and recall |
| ROC-AUC | Ability to rank failures above passing observations |
| Confusion matrix | Counts of correct and incorrect classifications |

For manufacturing quality applications, recall is particularly important because a false negative represents a failed observation that the model classified as passing.

However, excessive false positives may also increase inspection costs. The appropriate decision threshold should reflect operational requirements.

## Generated Outputs

| Output | Description |
|---|---|
| `models/quality_model.joblib` | Trained pipeline and selected threshold |
| `outputs/metrics.json` | Test-set evaluation metrics |
| `outputs/test_predictions.csv` | Actual labels, predictions, and failure probabilities |
| `outputs/confusion_matrix.csv` | Confusion matrix counts |
| `outputs/feature_importance.csv` | Ranked selected sensor features |
| `outputs/threshold_tuning.csv` | Validation precision, recall, and F1 across candidate thresholds |

## Dashboard Usage

The dashboard allows users to:

- Review the test-set failure rate.
- Change the classification threshold interactively.
- Examine correctly identified failures, missed failures, and false alarms.
- Explore failure-probability distributions.
- View ranked sensor feature importance.
- Inspect individual test observations.
- Filter results by actual outcome, predicted outcome, and probability band.
- Download filtered inspection records and feature importance.

Changing the threshold in the dashboard changes the displayed classifications; it does not retrain the model.

## Limitations and Future Improvements

This project has several important limitations:

- The dataset is relatively small and has a highly imbalanced failure class.
- Missing values and high-dimensional sensor data make reliable generalization challenging.
- Feature importance indicates predictive association, not causation.
- A threshold optimized for F1-score may not minimize industrial costs.
- A stratified random validation split does not fully represent future production conditions.
- Model performance can change as manufacturing equipment and processes change.
- The dataset is historical and does not provide a live production integration.

Potential future improvements include:

- Time-aware validation and rolling-window evaluation.
- Threshold selection based on the cost of false negatives and false positives.
- Probability calibration.
- Precision-recall curves and average precision.
- Explainability using SHAP or permutation importance.
- Drift monitoring and retraining.
- Integration with a simulated production monitoring system.
- Unit tests, experiment tracking, and automated deployment.

## Reproducibility

The training pipeline uses a fixed random seed for the Random Forest and validation split.

The project saves the fitted preprocessing and model pipeline together, along with the selected decision threshold.

For reproducible results, use a consistent Python environment and record package versions. Exact results can vary across software versions and environments.

## Disclaimer

This project is intended for educational, analytical, and portfolio purposes. It has not been validated for real-world production decisions, safety-critical processes, or autonomous rejection of manufactured products.

Any real deployment would require domain-expert review, independent validation, monitoring, and a defined quality-control procedure.

## Author

**Sumit Mondal**

Data Science | Machine Learning | Industrial Analytics | Python | SQL | Power BI | Tableau

GitHub: https://github.com/sm292

## License

Review the UCI dataset terms and any applicable third-party package licences before redistributing data or publishing derivative artifacts.

If you publish this project, consider adding an MIT licence for your original code while respecting the dataset's separate terms.
