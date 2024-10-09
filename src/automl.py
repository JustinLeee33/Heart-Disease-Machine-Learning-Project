from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
import os

def automl_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate an AutoML (TPOT) model with multiclass support."""
    model = TPOTClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    
    # Predict labels
    y_pred = model.predict(X_test)
    
    # Get predicted probabilities for each class
    y_scores = model.predict_proba(X_test)  # y_scores will have shape (n_samples, n_classes)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"AutoML (TPOT) Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Ensure the plot directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Return the fitted pipeline, predictions, and predicted probabilities
    return model.fitted_pipeline_, y_pred, y_scores
