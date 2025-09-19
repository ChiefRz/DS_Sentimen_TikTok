import joblib

# Misalkan 'svm_model' adalah model Anda dan 'tfidf_vectorizer' adalah vectorizer Anda
model_dan_vectorizer = {
    'model': joblib.load('model_svm.joblib'),
    'vectorizer': joblib.load('vectorizer.joblib')
}

joblib.dump(model_dan_vectorizer, 'sentiment_model.joblib')