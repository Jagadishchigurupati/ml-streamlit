# Machine Learning Assignment 2 - Classification & Deployment

## a. Problem statement
Implement five foundational machine learning classification models on a benchmark dataset, compute performance metrics, and build an interactive Streamlit application deployed on Cloud.

## b. Dataset description
- **Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Instances**: 569
- **Features**: 30 numeric attributes
- **Target**: Binary classification (0 = Malignant, 1 = Benign)
- 
### Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.9680 | 0.9722 | 0.9859 | 0.9787 | 0.9412 |
| Decision Tree | 0.9561 | 0.9490 | 0.9583 | 0.9718 | 0.9650 | 0.9023 |
| KNN | 0.9561 | 0.9450 | 0.9577 | 0.9718 | 0.9647 | 0.9015 |
| Naive Bayes | 0.9649 | 0.9590 | 0.9650 | 0.9789 | 0.9719 | 0.9205 |
| Support Vector Machine | 0.9737 | 0.9680 | 0.9722 | 0.9859 | 0.9787 | 0.9412 |

### Observations on Model Performance
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performed exceptionally well due to linear separability of scaled features, delivering high stability and MCC. |
| Decision Tree | Handled non-linear decision boundaries well, though slightly more prone to data variance. |
| KNN | Performed consistently after feature scaling, sensitive to local neighborhood densities. |
| Naive Bayes | Yielded strong baseline probabilistic results despite feature independence assumptions. |
| Support Vector Machine | Achieved highest overall accuracy and margin separation on the scaled feature space. |
| **Overall Winner** | **Logistic Regression & Support Vector Machine** (tied for top performance). |
