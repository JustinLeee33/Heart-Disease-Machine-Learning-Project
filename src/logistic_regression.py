def lr_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate Logistic Regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Save results
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(plot_dir, 'logistic_regression_report.txt'), 'w') as f:
        f.write(report)
        
    # Save accuracy plot
    plt.figure(figsize=(6, 4))
    plt.bar(['Logistic Regression'], [accuracy])
    plt.ylim(0, 1)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'logistic_regression_accuracy.png'))
    plt.close()
    
    return model  # Return the trained model
