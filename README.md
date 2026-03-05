# Bank Marketing ML Pipeline

End-to-end machine learning system for predicting whether a customer will subscribe to a term deposit during a marketing campaign.

## 📌 Overview

The project demonstrates a production-style ML workflow including experiment tracking, model optimization, and API deployment.
This project analyzes the **Portuguese Bank Marketing dataset** to predict whether a customer will subscribe to a term deposit after a
marketing phone call. The notebook applies **data exploration, feature engineering, and machine learning** techniques to uncover insights and
build predictive models.


## Pipeline Diagram
```
Customer dataset  
        ↓  
Feature preprocessing  
        ↓  
Model training (Logistic Regression)  
        ↓  
MLflow experiment tracking  
        ↓  
Model registry  
        ↓  
FastAPI prediction service  
```
---
## ⚙️ Methods

-   **Data Preprocessing**: Handling missing values, encoding categorical features, and scaling.
-   **Class Imbalance Handling**: SMOTE, ADASYN, and SMOTETomek applied
    to balance classes.
-   **Feature Selection**: ANOVA F-test, Recursive Feature Elimination
    (RFE), and model-based importance.
-   **Models Used**: Logistic Regression, Decision Tree, KNN, Random
    Forest, Gradient Boosting, and XGBoost.
-   **Evaluation Metrics**: ROC-AUC, Precision-Recall Curve, F1-score,
    and SHAP interpretability.

## 📊 Results

-   Ensemble methods (Random Forest, Gradient Boosting, XGBoost) outperformed baseline models.
-   Feature importance and SHAP values highlighted key drivers of deposit subscription (e.g., contact duration, previous campaign
    outcomes, and client demographics).
-   Balanced sampling methods improved recall for the minority class (subscribers).

## 🚀 Key Takeaways

-   Direct marketing effectiveness can be **quantitatively assessed and
    predicted** with ML.
-   Model interpretability tools like SHAP provide transparency into
    customer decision drivers.
-   The project demonstrates the value of combining **EDA, resampling,
    and ensemble modeling** for real-world imbalanced classification
    problems.

------------------------------------------------------------------------

## 🛠️ Dependencies and Setup Instructions

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📂 Data

Download `bank-additional-full.csv` from:\
- Kaggle: [Bank Marketing
Dataset](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv)

Place the file in the same directory as the notebook or inside a `data/`
folder.

------------------------------------------------------------------------

## ▶️ Usage

1.  Install required libraries\
2.  Download the dataset\
3.  Run all notebook cells in sequence\
4.  Analyze results and implement recommendations

------------------------------------------------------------------------

## 📁 Repository Structure

    bank-marketing-ml/  
    │
    ├── README.md                           # Project description and results  
    ├── bank_marketing_analysis.ipynb       # Main analysis notebook  
    ├── requirements.txt                    # Python dependencies  
    ├── data/  
    │   └── bank-additional-full.csv        # Dataset    
    ├── models/                             # Saved model files    
    ├── results/                            # Output files and plots  
