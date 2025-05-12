import numpy as np
import re
from os import PathLike, path
import os
from collections import defaultdict

from loader.dataset_creator import EDFDatasetCreator

def load_test_data(path: PathLike):
    dc = EDFDatasetCreator(path, 128, 0)
    X, y = dc.create_dataset()
    return X, np.array(y)

# Load patient's data for all experiments
def load_patient_dataset(path: PathLike, window_offset = 100, random_state = None, even_classes = False, window_size=128):
    return __load_patient_dataset(path, _replace_labels, window_offset, random_state, even_classes, window_size)

# Load patient's data for all experiments and keep only binary labels. Get last 2 experiments as validation set
def load_patient_dataset_binary(path: PathLike, window_offset = 100, random_state = None, even_classes = False, window_size=128):
    return __load_patient_dataset(path, _replace_labels_binary, window_offset, random_state, even_classes, window_size)

# Read and load all patient data, replacing annotations according to index_swap function. Keeps 2 experiments as validation set
def __load_patient_dataset(path: PathLike, index_swap_function, window_offset = 100, random_state = None, even_classes = False, window_size = 128):
    paths = _get_filtered_file_paths(path)
    X = []
    y = []
    X_test = []
    y_test = []
    # this is cursed, but works
    TEST_EXPERIMENTS = [9, 10]
    for idx, p in enumerate(paths):
        Xl, yl = _read_single_experiment(p, window_size, window_offset, index_swap_function)
        if idx in TEST_EXPERIMENTS:
            X_test.append(Xl)
            y_test.append(yl)
        else:
            X.append(Xl)
            y.append(yl)
    # unmessup the arrays
    X = np.array(X).reshape(-1, window_size, 64)
    y = np.array(y).flatten()
    X_test = np.array(X_test).reshape(-1, window_size, 64)
    y_test = np.array(y_test).flatten()
    # shuffle
    rng = np.random.default_rng(seed=random_state)
    train_shuffle = np.arange(X.shape[0])
    rng.shuffle(train_shuffle)
    test_shuffle = np.arange(X_test.shape[0])
    rng.shuffle(test_shuffle)
    X_train = X[train_shuffle]
    y_train = y[train_shuffle]
    X_test = X_test[test_shuffle]
    y_test = y_test[test_shuffle]
    if even_classes:
        X_train, y_train = __even_classes(X_train, y_train)
        X_test, y_test = __even_classes(X_test, y_test)
    return X_train, X_test, y_train, y_test

# Get paths to all files with valid experiments
def _get_filtered_file_paths(path: PathLike):
    files = os.listdir(path)
    filter = r".*R(01|02)\.edf$"
    filter = re.compile(filter)
    filtered = []
    for f in files:
        if not filter.match(f):
            filtered.append(os.path.join(path, f))
    filtered.sort()
    return filtered

def _read_single_experiment(path: PathLike, window, offset, label_replace):
    dc = EDFDatasetCreator(path, window, offset)
    X, y = dc.create_dataset()
    y = label_replace(np.array(y))
    return X, y

def _replace_labels(y: np.array):
    mapping_dict = {
            'BF': 'back',
            'BH': 'forward',
            'IBF': 'back',
            'IBH': 'forward',
            'ILH': 'left',
            'IRH': 'right',
            'LH': 'left',
            'R': 'relax',
            'RH': 'right'
            }
    ynew = np.vectorize(mapping_dict.get)(y)
    return ynew

def _replace_labels_binary(y: np.array):
    mapping_dict = {
            'BF': 'active',
            'BH': 'active',
            'IBF': 'active',
            'IBH': 'active',
            'ILH': 'active',
            'IRH': 'active',
            'LH': 'active',
            'R': 'relax',
            'RH': 'active'
            }
    ynew = np.vectorize(mapping_dict.get)(y)
    return ynew



def __even_classes(X, y):
    _, counts = np.unique(y, return_counts=True)
    min_represented = min(counts)
    idx_dict = defaultdict(list)
    for (idx, label) in enumerate(y):
        idx_dict[label].append(idx)
    selected = []
    for indices in idx_dict.values():
        curr_selected = np.random.choice(indices, min_represented, replace=False)
        selected.append(curr_selected)
    selected = np.array(selected).flatten()
    np.random.shuffle(selected)
    return X[selected], y[selected]
