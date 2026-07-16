# Fraud-Detection-ML
A Python-based machine learning project that detects fraudulent credit card transactions using supervised classification models. The project compares Logistic Regression, Random Forest and Histogram Gradient Boosting classifiers using scikit-learn pipelines, hyperparameter tuning and cross-validation. Model performance is evaluated using ROC-AUC, precision, recall and Precision-Recall curves on a highly imbalanced dataset.

---

# Features

Features of the fraud detection project:

Data Processing
- Downloads the public Kaggle credit card fraud dataset.
- Removes missing values and duplicate transactions.
- Splits the data into stratified training and test sets.
- Standardises features using a scikit-learn Pipeline.

Machine Learning Models
- Logistic Regression
- Random Forest Classifier
- Histogram Gradient Boosting Classifier

Model Optimisation
- Randomised hyperparameter search.
- Five-fold cross-validation.

Evaluation metrics
- Cross-validation and test ROC-AUC
- Precision
- Recall
- Training and prediction time
- ROC curves for all models
- Precision-recall curves for all models

---

# Methodology

The project follows a standard supervised machine learning workflow.

### Data Cleaning and Preparation

Missing values and duplicate transactions are removed before training the models. The dataset is highly imbalanced, containing only **0.17% fradulent trasactions**. This made accuracy an unsuitable evaluation metric; instead, we prioritised recall and precision and used stratification to preserve the proportion of fraudulent transactions in both the training and test datasets.

### Model training

Each model is trained within a scikit-learn Pipeline that applies StandardScaler before fitting the classifier. Standardisation transforms each feature as;

\[
z = \frac{x-\mu}{\sigma}
\],

where x is the original feature, \mu is the feature mean, and \sigma is the feature standard deviation. Hyperparameter tuning uses RandomisedSearchCV with five-fold cross-validation to find the best parameters with the highest ROC_AUC score to use in the final model.

### Evaluation Metrics

The ROC curve plots the true positive rate against the false positive rate for different classification thresholds. The ROC-AUC measures the area under the ROC curve and quantifies the classifier's ability to predict fraudulent transactions. A score of 1 is perfect classification, and 0.5 indicates random guessing.

Precision measures the proportion of predicted fraudulent cases that are actually fraudulent;

\[
Precision = \frac{TP}{TP + FP}
\].

Recall measures the proportion of fraudulent transactions successfully detected;

\[
Recall = \frac{TP}{TP + FN}
\].

In this case, identifying fraudulent transactions is the priority, so recall is considered more important than precision, although both are useful.

---

# Example Results

After hyperparameter tuning and cross-validation, the following data was obtained;

| Model                  | CV ROC_AUC | Test ROC_AUC | Precision | Recall | Training Time (s) | 
|:------------------------|-----------:|-------------:|----------:|-------:|------------------:|
| Logistic Regression     | 0.9839     | 0.9661       | 0.0537    | 0.8873 | 15.75             | 
| Random Forest           | 0.9819     | 0.9689       | 0.8699    | 0.7535 | 1082.52           | 
| HistGradientBoosting    | 0.9832     | 0.9744       | 0.1005    | 0.8592 | 51.92             | 

The Histogram Gradient Boosting classifier achieved the highest overall ROC-AUC on the unseen test data, suggesting the strongest overall classification performance.

The Logistic Regression model achieved the highest recall, successfully identifying almost 89% of fraudulent transactions, although this came at the cost of a very low precision and therefore a high number of false positives.

The Random Forest classifier produced substantially higher precision, meaning that transactions flagged as fraudulent were much more likely to be genuine fraud cases. However, this improvement came with a reduction in recall, missing a larger proportion of fraudulent transactions.

These results demonstrate the typical trade-off between precision and recall in fraud detection, where reducing false positives often increases the number of missed fraudulent transactions.

Training times also differed considerably between the models. Logistic Regression trained in approximately 16 seconds, Histogram Gradient Boosting required around 52 seconds, while Random Forest required over 18 minutes.

Example outputs can be found in

```
results/model_results.csv
results/roc_curves.png
results/pr_curves.png
```

---

# Future Improvements

Possible future extensions include

- Feature importance analysis
- XGBoost and LightGBM classifiers
- Cost-sensitive learning
- Threhold optimisation

---

# How to Run

```bash
pip install -r requirements.txt
python main.py

