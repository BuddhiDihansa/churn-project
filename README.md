# Customer Churn Prediction System

An end-to-end machine learning project that predicts customer churn and explains the main drivers behind each prediction through an interactive Streamlit dashboard.

## Overview

Customer churn is a critical business problem for telecom providers. This project builds a production-style pipeline that trains a churn classifier, saves the model artifacts, and exposes the results through a clean user interface with probability visualization and SHAP-based explainability.

## Key Features

- Churn probability prediction using a trained scikit-learn pipeline
- Interactive Streamlit dashboard for customer-level analysis
- Percentage donut chart to visualize churn likelihood
- SHAP explanations for local feature contributions
- Reproducible training script that regenerates the model and metadata

## Dataset

- IBM Telco Customer Churn dataset
- Approximately 7,000 customer records
- Features include demographics, account information, service subscriptions, and billing details

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Streamlit
- SHAP
- openpyxl

## Project Structure

```text
README.md
requirements.txt
app/
  churn_utils.py
  train_model.py
  streamlit_app.py
  artifacts/
    churn_model.pkl
    churn_model_meta.json
data/
  Telco_customer_churn.xlsx
notebooks/
  01_EDA.ipynb
  02_baseline_DT.ipynb
  03_RF_pipeline.ipynb
  04_SHAP.ipynb
```

## Setup

1. Create and activate your virtual environment.

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If you are using the project virtual environment on Windows, you can also run:

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train the Model

Run the training script to regenerate the model and metadata used by the app:

```bash
python app/train_model.py
```

This step reads the Excel dataset, prepares the features, trains the pipeline, evaluates it, and saves:

- `app/artifacts/churn_model.pkl`
- `app/artifacts/churn_model_meta.json`

## Run the Streamlit App

Start the dashboard with:

```bash
streamlit run app/streamlit_app.py
```

If you want to use the project virtual environment directly on Windows:

```bash
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

## What the App Shows

- Customer input form grouped by profile, services, billing, and value indicators
- Churn probability output
- Risk band and percentage chart
- SHAP explanation for the selected prediction
- Downloadable JSON report for the current prediction

## Model Notes

The final model is a Random Forest classifier wrapped in a preprocessing pipeline with:

- SimpleImputer for missing values
- StandardScaler for numeric features
- OneHotEncoder for categorical features

Latest reported performance from the training script:

- Cross-validation ROC-AUC: 0.8452 +/- 0.0116
- Test ROC-AUC: 0.8326

## Business Value

This project helps teams:

- Identify high-risk customers earlier
- Prioritize retention campaigns
- Explain why a customer is at risk of leaving
- Support data-driven decision-making with a visual dashboard

## Notebooks

The notebooks in the `notebooks/` folder document the exploratory and modeling workflow:

- `01_EDA.ipynb` for exploration
- `02_baseline_DT.ipynb` for the decision tree baseline
- `03_RF_pipeline.ipynb` for the random forest pipeline
- `04_SHAP.ipynb` for explainability analysis

## Requirements

The project depends on the packages listed in `requirements.txt`, including:

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- openpyxl
- shap

## License

No license has been added yet. If you plan to publish the repository publicly, add a license file before sharing it broadly.
