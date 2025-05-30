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
python train_and_log_models.py
![WhatsApp Image 2025-05-29 at 22 25 26_23e551db](https://github.com/user-attachments/assets/98908eed-d72c-4d9f-a245-a3698cd32e51)
![WhatsApp Image 2025-05-29 at 22 23 02_6cd5fb02](https://github.com/user-attachments/assets/3e064296-92b6-4f20-b875-587aad7b8c9d)
![WhatsApp Image 2025-05-29 at 22 28 23_460a6547](https://github.com/user-attachments/assets/48a73293-b837-48b0-b8e1-6a7d66213c23)
![WhatsApp Image 2025-05-29 at 22 28 49_ef891220](https://github.com/user-attachments/assets/a4d3c532-b9c5-4a2d-89be-cf4b61e5bb06)




## Usage

To reproduce training and logging:

```bash
python train_and_log_models.py



