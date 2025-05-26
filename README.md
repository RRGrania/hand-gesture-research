## 📊 Model Comparison

During the training phase, we evaluated three different classification models using the same hand gesture dataset. The table below summarizes their performance based on **Accuracy** and **F1 Score**:

| Model           | Accuracy | F1 Score |
|-----------------|----------|----------|
|  **XGBoost**        | **93.61%** | **93.63%** |
| Random Forest | 77.68%   | 78.23%   |
|  SVM             | 69.50%   | 69.14%   |

---

### Observations

-  **XGBoost** achieved the highest accuracy and F1 score, making it the most effective model for this hand gesture classification task.
-  **Random Forest** performed moderately well but lagged behind XGBoost.
-  **SVM** struggled with the multiclass nature of the task and yielded the lowest scores.

###  MLflow Integration

All models were **logged and tracked using MLflow**, enabling:
-  Version control
-  Performance tracking
-  Easy deployment
