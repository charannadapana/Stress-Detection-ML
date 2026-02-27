from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

def train_models(X_train_vec, y_train):

    log_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {'C': [0.01, 0.1, 1, 10]},
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    svm_grid = GridSearchCV(
        LinearSVC(class_weight="balanced"),
        {'C': [0.01, 0.1, 1, 10]},
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    nb_grid = GridSearchCV(
        MultinomialNB(),
        {'alpha': [0.1, 0.5, 1.0]},
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    log_grid.fit(X_train_vec, y_train)
    svm_grid.fit(X_train_vec, y_train)
    nb_grid.fit(X_train_vec, y_train)

    return {
        "Logistic Regression": log_grid.best_estimator_,
        "Linear SVM": svm_grid.best_estimator_,
        "Naive Bayes": nb_grid.best_estimator_
    }