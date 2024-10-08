
from sklearn.metrics import accuracy_score, classification_report
import os

def ensemble_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
 
    
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
