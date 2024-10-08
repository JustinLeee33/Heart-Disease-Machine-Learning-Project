from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def ensemble_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate an ensemble VotingClassifier model."""
    
    # Define base models (make sure lr, rf, svm are defined elsewhere)
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(random_state=42)
    svm = SVC(probability=True, random_state=42)
    
    # VotingClassifier (Ensemble)
    model = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('svm', svm)], voting='soft')
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Get predicted probabilities for each class
    y_scores = model.predict_proba(X_test)  # y_scores is now a 2D array (n_samples, n_classes)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Ensemble Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(plot_dir, 'ensemble_report.txt'), 'w') as f:
        f.write(report)
    
    # Return model, predictions, and scores (probabilities for each class)
    return model, y_pred, y_scores
