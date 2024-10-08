
from sklearn.metrics import accuracy_score, classification_report
import os

def automl_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate an AutoML (TPOT) model."""
    model = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]  # Probability estimates

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"AutoML (TPOT) Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Return model, predictions, and scores
    return model.fitted_pipeline_, y_pred, y_scores
