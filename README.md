# Credit Card Fraud Detection

**Project Goal:** Develop a robust and financially optimized machine learning model to accurately detect fraudulent transactions within a highly imbalanced dataset, minimizing chargeback losses and false positives.

---

## Key Results & Best Model Performance

The **Random Forest Classifier** was selected as the optimal model because it offered the best balance between identifying fraud (Recall) and minimizing false alarms (Precision), resulting in the largest net financial gain.

| Model | AUC | Precision (Class 1) | Recall (Class 1) | Financial Gain (Test Set Example) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Optimal)** | **0.981** | **0.75** | **0.83** | **+ $56,700** |
| Neural Network | 0.965 | 0.72 | 0.79 | *(Calculation Pending)* |
| Logistic Regression | 0.973 | 0.06 | 0.89 | - $180,700 |
| K-Neighbors | 0.935 | 0.33 | 0.87 | *(Calculation Pending)* |

---

## Technical Workflow

This project executes an end-to-end data science pipeline, focusing on solutions for handling imbalanced data:

### 1. Data Analysis and Pre-processing
* **EDA:** Initial analysis confirmed **extreme class imbalance** (less than 0.2% fraud cases) and identified high skew in the `Amount` feature.
* **Feature Engineering:** Applied a **log-transform** (`np.log1p`) to the transaction `Amount` to normalize its distribution, which also reduced its **Variance Inflation Factor (VIF)** from 13.17 to 5.03, mitigating multicollinearity.
* **Scaling:** Used **StandardScaler** on `Time` and `log_amount` to standardize inputs for distance-based and linear models.
* **Class Balancing:** Employed the **SMOTE (Synthetic Minority Over-sampling Technique)** algorithm on the training data to generate synthetic samples and address the class imbalance problem.

### 2. Modeling & Evaluation
* **Comparative Modeling:** Benchmarked performance across multiple algorithms: **Logistic Regression**, **Decision Tree**, **Random Forest**, **Neural Network**, and **K-Neighbors Classifier**.
* **Metrics:** Focused on **AUC**, **Precision**, and **Recall** to evaluate performance, understanding that Recall (catching fraud) and Precision (avoiding false alarms) are critical for financial applications.
* **Interpretability:** Used the **Random Forest feature importance** scores to identify the top three drivers of fraud detection: **V14, V10, and V11**.

### 3. Business Impact
* **Cost-Benefit Analysis:** Developed a simplified **cost matrix** to translate model performance (TP, FP, FN) into a tangible **Net Financial Gain**, justifying the Random Forest model as the financially optimal choice despite marginally lower raw AUC compared to some others.

---

## Key Visualizations

* 

<img width="695" height="547" alt="image" src="https://github.com/user-attachments/assets/8d9c43c2-6d5c-4fdf-af16-b721db7f3aca" />


* 

<img width="695" height="547" alt="image" src="https://github.com/user-attachments/assets/40f36115-649c-4c3f-9387-34ca07ee7efa" />



---

## Repository Contents

* `CCFraudDetection.ipynb`: The primary project notebook containing all code, analysis, and model results.
* `README.md`: This summary document.
* `.gitignore`: Specifies files (like the raw data, `.csv`) to be ignored by Git.
* `LICENSE`: MIT License governing the project's use.
