# Random Forest Classifier - Breast Cancer

A machine learning project using **Random Forest** to classify tumors as **malignant** or **benign** with the Breast Cancer dataset from `scikit-learn`.

## Steps
- Load dataset
- Train-test split (80/20)
- Train Random Forest model
- Evaluate with Accuracy, Confusion Matrix, Precision, Recall, F1-score, ROC-AUC
- Plot ROC Curve

## Results
- Accuracy: ~95–97%
- ROC-AUC: ~0.99
- Strong precision, recall, and F1-score

## Run
```bash
pip install scikit-learn matplotlib numpy
python random_forest_classifier.py
