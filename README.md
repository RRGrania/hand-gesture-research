# Hand Gesture Recognition - Model Comparison and MLflow Tracking

## 📊 Model Comparison

During the training phase, we evaluated four different classification models using the same hand gesture dataset. The table below summarizes their performance based on **Accuracy** and **F1 Score**:

| Model            | Accuracy | F1 Score |
|------------------|----------|----------|
| **XGBoost**      | **99.83%** | **99.83%** |
| Random Forest    | 99.83%   | 99.83%   |
| SVM              | 99.83%   | 99.83%   |
| Keras FFNN       | 99.83%   | 99.83%   |

---

### Observations

- All models performed **exceptionally well**, achieving near-perfect accuracy and F1 scores on the test set.
- This high performance suggests the dataset is well-structured and that the features are highly separable.
- The **Keras Feedforward Neural Network (FFNN)** performed on par with ensemble and kernel-based models.

###  MLflow Integration

All models were **tracked and logged using MLflow**, enabling:

- ✅ **Version Control** – Track experiments with different hyperparameters or architectures.
- ✅ **Performance Tracking** – Automatically recorded evaluation metrics.
- ✅ **Model Reproducibility** – Models were logged with **signatures** and **input examples** for easier reproducibility.
- ✅ **Deployment-Ready Artifacts** – Models are stored and ready to be loaded or served via MLflow-compatible tools.

---

## Usage

To reproduce training and logging:

```bash
python train_and_log_models.py
