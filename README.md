# 🚀 ML Pipeline with GitHub Actions

This repository contains an end-to-end machine learning pipeline for a classification task using the Titanic dataset. The entire workflow is automated using **GitHub Actions**.

## 📌 Project Overview

- **Dataset**: Titanic survival dataset (public dataset from Kaggle / seaborn)
- **Model**: Classification using `scikit-learn`
- **Pipeline Includes**:
  - Data loading and preprocessing (handling missing values, normalization)
  - Model training
  - Model evaluation (accuracy)
  - Model serialization using `joblib`
  - Unit testing (data and model)
  - CI/CD with GitHub Actions

---

---

## 🧪 Tests

The pipeline includes the following tests:
- ✅ Preprocessing functions (e.g., missing value handling, normalization)
- ✅ Model performance (e.g., accuracy ≥ 80%)

Run locally with:
```bash
python -m pytest tests/

