from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

def ensemble_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate an Ensemble (Voting) model."""
    
    # Define base models
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100)
    svm = SVC(probability=True)
    
    # VotingClassifier (Ensemble)
    model = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('svm', svm)], voting='soft')
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Ensemble Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Return model, predictions, and scores
    return model, y_pred, y_scores
