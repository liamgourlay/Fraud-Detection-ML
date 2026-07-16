import time

import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_data():
    """
    Load in dataset from Kaggle and convert to DataFrame.

    Returns
    ---------
    df: DataFrame
        Dataset is returned as a DataFrame
    """
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print("Path to dataset files:", path)
    df = pd.read_csv(path + "/creditcard.csv")
    return df


def clean_data(df):
    """
    Cleans DataFrame by dropping missing data and removing duplicated rows.

    Parameters
    ---------
    df: DataFrame
        DataFrame of fraud transaction data.

    Returns
    ----------
    df: DataFrame
        Returns a cleaned DataFrame.
    """
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    # Calculate and print the percentage of data that is classified as fraud.
    print(f"Percentage of fraud transcations in dataset: {(df['Class'].value_counts(normalize=True) * 100)[1]}")
    return df


def split_data(df, test_size=0.3, random_state=42):
    """
    Splits data into a training set and testing set.

    Parameters
    ----------
    df: DataFrame
        A cleaned DataFrame of fraud transaction data.
    test_size: float
        Designating the fraction of data to test a given model on.
    random_state: int
        A random state to allow for reproducibility.

    Returns
    ----------
    X_train: DataFrame
        A DataFrame containing the features to train on.
    X_test: DataFrame
        A DataFrame containing the features to make predictions with.
    y_train: DataFrame
        A DataFrame containing the class indicators to train on.
    y_test: DataFrame
        A DataFrame containing the indicators to compare predictions with.
    """
    # Separate features and class.
    X = df.drop(columns="Class")
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_models():
    """
    Lists the models that will be used as tuples with their functions and hyperparameters to test on.

    Returns
    ----------
    models: list
        Lists the models with their functions and hyperparameters to test with as tuples.
    """
    models = [
        ("Logistic Regression", LogisticRegression(
                                    max_iter=1000,
                                    class_weight="balanced"),
         {"model__C": [0.01, 0.1, 1, 10]}),
        ("Random Forest", RandomForestClassifier(
                                random_state=42,
                                n_jobs=-1,
                                class_weight="balanced"),
         {"model__n_estimators": [100, 200],
          "model__max_depth": [None, 10, 20],
          "model__min_samples_leaf": [1, 5]}),
        ("HistGradientBoosting", HistGradientBoostingClassifier(
                                    class_weight="balanced",
                                    random_state=42),
         {"model__learning_rate": [0.01, 0.1],
          "model__max_depth": [3, 5],
          "model__max_iter": [100, 200]})
    ]
    return models


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """
    Trains the different models on the training data.
    The modes are evaluated using cross validation and hyperparameter tuning is used to find the best parameters to run the models with.
    The results are evaluated looking at ROC_AUC, precision and recall as well as run time.

    Parameters
    ----------
    X_train: DataFrame
        A DataFrame containing the features to train on.
    X_test: DataFrame
        A DataFrame containing the features to make predictions with.
    y_train: DataFrame
        A DataFrame containing the class indicators to train on.
    y_test: DataFrame
        A DataFrame containing the indicators to compare predictions with.

    Returns
    ----------
    results: DataFrame
        A DataFrame containing roc_auc, precision, recall and training time for each model.
    roc_curves: list
        A list of tuples containing the data from the roc_curves of each model.
    pr_curves: list
        A list of tuples containing the data from the precision-recall curves of each model.
    """
    # Setup metrics dictionary and curve data lists to store results for each model.
    metrics = {
        "models": [],
        "CV ROC_AUC": [],
        "Test ROC_AUC": [],
        "precision": [],
        "recall": [],
        "Training Time (s)": [],
        "Prediction Time (s)": []
    }
    roc_curves = []
    pr_curves = []

    for model_name, model, param_grid in models:

        # Create a scaling and fitting pipeline for each model.
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

        # Use cross validation and hyperparameter tuning to find the best parameters for each model.
        grid = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            n_iter=10,
            random_state=42
        )

        # Time each model for training and prediction.
        train_start = time.time()
        grid.fit(X_train, y_train)
        train_time = time.time() - train_start

        best_model = grid.best_estimator_

        # Use the best parameters to predict class based on test data.
        predict_start = time.time()
        y_pred_probs = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)
        predict_time = time.time() - predict_start

        # Save cross validation ROC_AUC scores
        metrics["models"].append(model_name)
        metrics["CV ROC_AUC"].append(grid.best_score_)

        # Save test ROC_AUC score based on the test data and save ROC curve data.
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
        test_auc = roc_auc_score(y_test, y_pred_probs)
        metrics["Test ROC_AUC"].append(test_auc)
        roc_curves.append((model_name, fpr, tpr, test_auc))

        # Save precision recall data.
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_pred_probs)
        avg_precision = average_precision_score(y_test, y_pred_probs)
        pr_curves.append((model_name, precision_vals, recall_vals, avg_precision))

        # Save precision and recall scores from test data.
        class_report = classification_report(y_test, y_pred, output_dict=True)
        metrics["precision"].append(class_report["1"]["precision"])
        metrics["recall"].append(class_report["1"]["recall"])

        # Save training and prediction times.
        metrics["Training Time (s)"].append(train_time)
        metrics["Prediction Time (s)"].append(predict_time)

    # Convert metrics dictionary to DataFrame and save it.
    results = pd.DataFrame(metrics)
    results.to_csv("results/model_results.csv", index=False)
    return results, roc_curves, pr_curves


def plot_roc_curves(roc_curves):
    """
    Plots and saves the roc curves for each model.

    Parameters
    ---------
    roc_curves: list
        A list of tuples containing the data for the roc_curves for each model.
    """
    plt.figure()

    # Plot roc curves for each model.
    for model_name, fpr, tpr, test_auc in roc_curves:
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {test_auc:.3f})")

    # Configure plot, include y=x line for comparison.
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True)

    plt.savefig("results/roc_curves.png")
    plt.show()


def plot_pr_curves(pr_curves):
    """
    Plots the precision-recall curves for each model.

    Parameters
    ----------
    pr_curves: list
        A list of tuples containing data for the precision and recall of each model.
    """
    plt.figure()

    # Plot precision recall curves for each model.
    for model_name, precision_vals, recall_vals, avg_precision in pr_curves:
        plt.plot(recall_vals, precision_vals, label=f"{model_name} (AP = {avg_precision:.3f})")

    # Configure plots.
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(True)

    plt.savefig("results/pr_curves.png")
    plt.show()


def display_results(results):
    """
    Displays the results DataFrame clearly.

    Parameters
    results: DataFrame
        A DataFrame containing metrics for each model.
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(results.to_string(index=False))

def main():
    df = load_data()
    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    models = get_models()

    results, roc_curves, pr_curves = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    display_results(results)

    plot_roc_curves(roc_curves)
    plot_pr_curves(pr_curves)


if __name__ == "__main__":
    main()
