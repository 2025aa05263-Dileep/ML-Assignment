# ML-Assignment

# Wine Quality Prediction - ML Classification Model Comparison

## a. Problem Statement
The goal of this project is to predict the quality of wine based on its physicochemical properties. Using the Wine Quality Dataset from the UCI Machine Learning Repository, we aim to perform binary classification to determine if a wine is of "Good Quality" (score ≥ 6) or "Poor Quality" (score < 6). This prediction helps winemakers optimize production processes and ensure consistent quality based on objective physicochemical tests.

## b. Dataset Description
- **Dataset:** Wine Quality Dataset (Red and White wine)
- **Source:** UCI Machine Learning Repository
- **Instances:** 6,497 (1,599 red + 4,898 white wines)
- **Features (12):**
  - **Fixed Acidity, Volatile Acidity, Citric Acid, pH:** Measurements of acidity levels.
  - **Residual Sugar, Density:** Related to sweetness and body.
  - **Chlorides, Free Sulfur Dioxide, Total Sulfur Dioxide, Sulphates:** Related to salts and preservatives.
  - **Alcohol:** Percentage of alcohol by volume.
  - **Type:** Binary indicator for Red or White wine.
- **Target Variable:** Binary (0: Poor Quality, 1: Good Quality)

## c. Models Used
Six different machine learning models were implemented and compared using standard evaluation metrics.

### Model Comparison Table
| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ---                       | ---      | ---    | ---       | ---    | ---    | ---    |
| Logistic Regression       | 0.7392   | 0.8057 | 0.7665    | 0.8457 | 0.8042 | 0.4214 |
| Decision Tree             | 0.7654   | 0.7486 | 0.8166    | 0.8117 | 0.8141 | 0.4961 |
| kNN                       | 0.7408   | 0.8004 | 0.7780    | 0.8262 | 0.8014 | 0.4308 |
| Naive Bayes               | 0.6831   | 0.7445 | 0.7179    | 0.8226 | 0.7667 | 0.2861 |
| Random Forest (Ensemble)  | 0.8362   | 0.9047 | 0.8538    | 0.8943 | 0.8736 | 0.6425 |
| XGBoost (Ensemble)        | 0.8285   | 0.8825 | 0.8513    | 0.8834 | 0.8670 | 0.6265 |

### Performance Observations
| ML Model Name             | Observation about model performance |
| ---                       | ---                                 |
| Logistic Regression       | Provides a solid baseline. It works well for linearly separable features and provides interpretable probabilistic predictions. |
| Decision Tree             | Captures non-linear relationships effectively but is prone to overfitting if not carefully tuned. Offers high interpretability via decision rules. |
| kNN                       | Performs well by identifying similar wine profiles. Highly sensitive to feature scaling, which was addressed during preprocessing. |
| Naive Bayes               | Most efficient in terms of training time. While its independence assumption is strong for chemical data, it still serves as a useful baseline. |
| Random Forest (Ensemble)  | Significantly improves accuracy by reducing variance through bagging. One of the top performers, handling feature interactions very well. |
| XGBoost (Ensemble)        | Achieves excellent results by iteratively correcting errors from previous trees (boosting). Shows the best balance across all metrics. |
