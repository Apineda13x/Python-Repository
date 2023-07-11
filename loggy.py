
def logregression(X_train, X_test, y_train, y_test):
    import sklearn
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    logreg = GridSearchCV(model,param_grid={"penalty": ['l1','l2']})
    logreg.fit(X_train, y_train)
    y_pred_log = logreg.predict(X_test)

    print("Logistic regression accuracy for test set:", logreg.score(X_test, y_test)),
    print("\nClassification report:"),
    print(classification_report(y_test, y_pred_log)),
    print(logreg.best_estimator_.coef_),
    print(logreg.best_estimator_.intercept_),
    print(X_train.columns)       
    
    return
    