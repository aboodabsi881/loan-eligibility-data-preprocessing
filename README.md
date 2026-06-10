# Loan Eligibility Data Preprocessing & Cleaning Pipeline

A comprehensive data engineering and preprocessing project focused on preparing financial logs (`loan_train.csv`) for predictive machine learning models. This repository implements structural data auditing, descriptive statistics, robust null-value imputation pipelines, and categorical feature transformations using `scikit-learn`.

---

## 📌 Project Overview
Raw financial and credit datasets are frequently plagued by missing records, mixed data types, and non-numeric fields that break standard machine learning algorithms. This project builds a reliable, reproducible preprocessing pipeline to clean customer application files, handling structural anomalies, imputing null entries cleanly without inducing data leakage, and encoding categorical text attributes into model-ready numerical vectors.

## 📊 Dataset Features
The project manipulates loan applications across 13 core dimensions, tracking key demographic and financial risk factors:
* **Loan_ID:** Unique transaction sequence tracking index (dropped during preprocessing).
* **Gender / Married / Dependents:** Applicant demographic criteria profile.
* **Education / Self_Employed:** Professional and educational background classification.
* **ApplicantIncome / CoapplicantIncome:** Combined monthly economic intake.
* **LoanAmount / Loan_Amount_Term:** Requested financial terms (amount and repayment tenure).
* **Credit_History:** Past credit compliance metrics (crucial risk parameter).
* **Property_Area:** Geographic operational zone classification (`Urban`, `Semiurban`, `Rural`).
* **Loan_Status:** Target label indicating application approval or denial.

---

## 🛠️ Data Pipeline Architecture & Implementation

### 1. Structural Auditing & Integrity Checks
* **Dimensions & Formats:** Evaluates data frame shapes (`.shape`), programmatic attribute classifications (`.dtypes`), and data distribution footprints (`.info()`, `.describe()`).
* **Missing Value Auditing:** Uses conditional matrix aggregations (`df.apply(lambda x: sum(x.isnull()), axis=0)`) to isolate exact null-counts per feature.
* **Duplicate Detection:** Verifies records integrity and clears structural overlap indices via `.duplicated().value_counts()`.

### 2. Strategy-Based Null Imputation
Different statistical filling criteria are implemented depending on feature types to prevent data distortion:
* **Continuous Numerical Fields:** Missing records in `LoanAmount` and `Loan_Amount_Term` are seamlessly resolved using column-specific mean values (`.fillna(df[col].mean())`).
* **Categorical / Boolean Fields:** Employs explicit mode/majority imputation for non-numeric fields, applying fallback baselines such as `'Male'` for gender, `'Yes'` for marriage logs, `'No'` for self-employment attributes, and `'0'` for dependency entries.

### 3. Feature Transformation & Automated Encoding
* **Uninformative Feature Pruning:** Drops high-cardinality metadata identifiers like `Loan_ID` via `.drop(axis=1)` to focus exclusively on predictive variables.
* **Automated Batch Encoding:** Implements an automated type-filtering layout to extract all remaining text parameters (`object` types) and maps them into integers sequentially using `scikit-learn`'s `LabelEncoder()`.

---

## 💻 Tech Stack & Dependencies
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Preprocessing Framework:** `scikit-learn`
* **Environment:** Jupyter Notebook

## 🚀 Getting Started
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/loan-eligibility-data-preprocessing.git](https://github.com/your-username/loan-eligibility-data-preprocessing.git)
   cd loan-eligibility-data-preprocessing
