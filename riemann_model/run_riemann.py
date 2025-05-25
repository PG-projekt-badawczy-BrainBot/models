import os
import numpy as np
import pandas as pd
from loader.loader import load_patient_dataset, load_patient_dataset_binary
from riemann_model.riemann import calculate_covariance, approximate_riemannian_mean, embed_into_tangent, vectorize_tangent_matrices
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score


def main():
    X_train, X_test, y_train, y_test = preprocess()
    Ctest = [4, 4.5, 5, 5.5, 6]
    for c in Ctest:
        print('x'*50)
        print(f'C = {c}')
        model = SVC(C=c, gamma='scale', kernel='rbf')
        print('Training model with 5-fold cv')
        model.fit(X_train, y_train)
        print('Train')
        yhat_train = model.predict(X_train)
        print(classification_report(y_train, yhat_train))
        print('Test')
        yhat_test = model.predict(X_test)
        print(classification_report(y_test, yhat_test))
        print('x'*50)


def preprocess():
    file_path = os.path.join("../../", 'processed_eeg_data', 'S001')
    print('Loading dataset')
    X_train, X_test, y_train, y_test = load_patient_dataset(file_path, window_offset=450, window_size=512)
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))
    print('Calculating covariance matrices')
    X_train = calculate_covariance(X_train)
    X_test = calculate_covariance(X_test)
    print('Approximating riemannian mean')
    mean_embed_train = approximate_riemannian_mean(X_train)
    print('Embedding into tangent space')
    X_train = embed_into_tangent(X_train, mean_embed_train)
    X_test = embed_into_tangent(X_test, mean_embed_train)
    print('Vectorizing')
    X_train = vectorize_tangent_matrices(X_train)
    X_test = vectorize_tangent_matrices(X_test)
    print('Done')
    print(f'Train shape: {X_train.shape}, test shape: {X_test.shape}')
    print('First train sample:')
    print(X_train[0])
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    main()