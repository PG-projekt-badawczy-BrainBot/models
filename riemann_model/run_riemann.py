import os
import numpy as np
import pandas as pd
from loader.loader import load_patient_dataset, load_patient_dataset_binary
from riemann_model.riemann import calculate_covariance, approximate_riemannian_mean, embed_into_tangent, vectorize_tangent_matrices
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from EEG1DCNN.physionet_data_loader import PhysionetDataLoader
from sklearn.metrics import accuracy_score
import random

"""
# probably best phisical aligment of electrodes
electrodes= [
    'FC3', 'FC4', 'FC1', 'FC2',
    'C3', 'C4', 'C1', 'C2',
    'CP5', 'CP6', 'CP3', 'CP4',
    'P5', "P6", "FC6"
]
"""
# best electrodes found with genetic algorithm
elecdrodes = ['FC5', 'CP2', 'P8', 'FC2', 'CPz', 'Cz', 'F6', 'AF3', 'PO8', 'Fpz', 'FT7', 'CP3', 'FC6', 'PO3', 'F1', 'TP8', 'T10', 'T8', 'FC4', 'POz', 'CP6', 'FT8', 'Fp1', 'C4', 'P6', 'AF4', 'F2', 'P2', 'TP7', 'CP4', 'Oz', 'O1']

def genetic_search_electrodes(subject_id='S003', n_generations=10, pop_size=20, n_channels=32, mutation_rate=0.1):
    data_dir = r"C:\Users\kasia\projekt_badawczy\eeg-motor-movementimagery-dataset-1.0.0\files"
    raw = PhysionetDataLoader(data_dir).load_subject_data(subject_id)
    all_channels = raw.ch_names
    n_total_channels = len(all_channels)

    def create_individual():
        return random.sample(range(n_total_channels), n_channels)

    def evaluate(individual):
        selected_electrodes = [all_channels[i] for i in individual]
        X_train, X_test, y_train, y_test = preprocess(selected_electrodes)
        model = SVC(C=5.5, gamma='scale', kernel='rbf')
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        return acc

    def crossover(parent1, parent2):
        cut = random.randint(1, n_channels - 1)
        child = parent1[:cut] + [gene for gene in parent2 if gene not in parent1[:cut]]
        return child[:n_channels]

    def mutate(individual):
        if random.random() < mutation_rate:
            idx = random.randint(0, n_channels - 1)
            new_gene = random.choice([i for i in range(n_total_channels) if i not in individual])
            individual[idx] = new_gene
        return individual

    population = [create_individual() for _ in range(pop_size)]
    scores = [evaluate(ind) for ind in population]

    for generation in range(n_generations):
        print(f"\nGeneration {generation+1}")
        sorted_pop = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        population, scores = zip(*sorted_pop)
        population, scores = list(population), list(scores)

        print(f"Best accuracy: {scores[0]:.4f} | Electrodes: {[all_channels[i] for i in population[0]]}")

        # Elitism: keep top 2
        new_population = population[:2]

        # Generate new children
        while len(new_population) < pop_size:
            parents = random.sample(population[:10], 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            new_population.append(child)

        population = new_population
        scores = [evaluate(ind) for ind in population]

    best = max(zip(population, scores), key=lambda x: x[1])
    best_electrodes = [all_channels[i] for i in best[0]]
    print(f"\nBest electrode set: {best_electrodes}")
    print(f"Final accuracy: {best[1]:.4f}")


def random_search_electrodes(subject_id='S003', n_iter=3, n_channels=32):
    data_dir = r"C:\Users\kasia\projekt_badawczy\eeg-motor-movementimagery-dataset-1.0.0\files"
    raw = PhysionetDataLoader(data_dir).load_subject_data(subject_id)
    all_channels = raw.ch_names

    results = []

    for i in range(n_iter):
        selected_electrodes = random.sample(all_channels, n_channels)
        print(f"\n{i+1}/{n_iter}: Testing with electrodes: {selected_electrodes}")

        try:
            X_train, X_test, y_train, y_test = preprocess(selected_electrodes)

            model = SVC(C=5.5, gamma='scale', kernel='rbf')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            results.append((selected_electrodes, acc))
            print(f"Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"Error with electrodes {selected_electrodes}: {e}")

    results.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 5 electrode sets:")
    for electrodes, acc in results[:5]:
        print(f"{acc:.4f}: {electrodes}")


def main():
    X_train, X_test, y_train, y_test = preprocess(elecdrodes)
    Ctest = [5.5]
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


def preprocess(electrodes=None):
    subject_id = 'S003'
    file_path = os.path.join("../../", 'processed_eeg_data', subject_id)
    print('Loading dataset')
    X_train, X_test, y_train, y_test = load_patient_dataset(file_path, window_offset=450, window_size=512)

    if electrodes != None:
        data_dir = r"C:\Users\kasia\projekt_badawczy\eeg-motor-movementimagery-dataset-1.0.0\files"
        data_loader = PhysionetDataLoader(data_dir)
        raw = data_loader.load_subject_data(subject_id)
        channel_name_to_index = {ch: idx for idx, ch in enumerate(raw.ch_names)}
        selected_indices = [channel_name_to_index[e] for e in electrodes]
        X_train = X_train[:, :, selected_indices]
        X_test  = X_test[:, :, selected_indices]
        print(f'Selected electrodes: {electrodes}')
        print(f'Train shape: {X_train.shape}, test shape: {X_test.shape}')


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
    #genetic_search_electrodes()
    main()