from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(X_train, X_test):
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1,2)
    )

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    return tfidf, X_train_vec, X_test_vec