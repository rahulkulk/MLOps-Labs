Feature Selection Lab 

This notebook explores multiple feature selection techniques using the Breast Cancer dataset and evaluates how reducing features affects model performance and efficiency. The notebook has been enhanced to produce more reliable and production-aligned results.

Key Improvements Added
1. Pipeline for Scaling + Model
All model training now uses a Pipeline(StandardScaler → RandomForestClassifier) to prevent data leakage and ensure consistent preprocessing across training, testing, and cross-validation.

2. Cross-Validation for Reliable Evaluation
Each feature selection method now includes 5-fold cross-validation, making accuracy comparisons more stable and less dependent on a single train/test split.

3. Training & Inference Time Metrics
The evaluation table now reports:
- Training time
- Inference time
- Feature count: This highlights the computational benefits of selecting fewer features.

What You Can Learn
- How filter, wrapper, and embedded feature selection techniques work
- How feature subsets impact accuracy, generalization, and speed
- Why proper scaling and cross-validation are critical for trustworthy feature selection
- How to choose an optimal feature subset balancing performance vs. efficiency

Output: The final results table compares all feature selection methods using:
- Accuracy
- ROC
- Precision
- Recall
- F1 score
- Feature count
- Training time
- Inference time
- Cross-validation accuracy
