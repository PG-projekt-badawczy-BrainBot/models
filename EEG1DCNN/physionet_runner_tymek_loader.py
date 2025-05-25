from physionet_data_loader import PhysionetDataLoader
import mne
import numpy as np
from EEG1DCNN import train_model
import os
from loader.loader import load_patient_dataset


mne.set_log_level('WARNING')
subject_id = 'S003'
data_dir = r"C:\Users\kasia\projekt_badawczy\eeg-motor-movementimagery-dataset-1.0.0\files"

data_loader = PhysionetDataLoader(data_dir)
raw = data_loader.load_subject_data(subject_id)


# Mapowanie: nazwa → index
channel_name_to_index = {ch: idx for idx, ch in enumerate(raw.ch_names)}

file_path = os.path.join(r"C:\Users\kasia\projekt_badawczy", 'processed_eeg_data', 'S003')
print('Loading dataset')
X_train, X_test, y_train, y_test = load_patient_dataset(
    file_path, window_offset=128, window_size=512, even_classes=True)

# Usuń "relax" ze zbioru treningowego
mask = np.array(y_train) != 'relax'
X_train = X_train[mask]
y_train = np.array(y_train)[mask]

mask_test = np.array(y_test) != 'relax'
X_test = X_test[mask_test]
y_test = np.array(y_test)[mask_test]



electrode_pairs = [
    ('FC5', 'FC6'), ('FC3', 'FC4'), ('FC1', 'FC2'),
    ('C5', 'C6'), ('C3', 'C4'), ('C1', 'C2'),
    ('CP5', 'CP6'), ('CP3', 'CP4'), ('CP1', 'CP2')
]


def preprocess_data(X, y, channel_name_to_index, electrode_pairs):
    """
    Convert EEG windows to CNN-ready format using electrode pairs.

    Parameters:
    - X: EEG data of shape (n_samples, window_size, n_channels)
    - y: labels of shape (n_samples,)
    - channel_name_to_index: dict mapping channel name to index
    - electrode_pairs: list of tuples (ch1, ch2)

    Returns:
    - X_out: shape (n_samples * n_pairs, window_size, 2, 1)
    - y_out: repeated labels to match X_out
    """
    X_out = []
    y_out = []

    for i in range(X.shape[0]):
        sample = X[i]  # shape: (window_size, n_channels)
        for ch1, ch2 in electrode_pairs:
            idx1 = channel_name_to_index[ch1]
            idx2 = channel_name_to_index[ch2]

            pair_data = np.stack([sample[:, idx1], sample[:, idx2]], axis=-1)  # (256, 2)
            pair_data = pair_data.astype(np.float32)
            X_out.append(pair_data[..., np.newaxis])
            y_out.append(y[i])  # repeat label for each pair

    X_out = np.array(X_out, dtype=np.float32)
    y_out = np.array(y_out)
    return X_out, y_out



X_train, y_train = preprocess_data(X_train, y_train, channel_name_to_index, electrode_pairs)
X_test, y_test = preprocess_data(X_test, y_test, channel_name_to_index, electrode_pairs)
print(f"Preprocessed train data shape: {X_train.shape}")
print(f"Preprocessed test data shape: {X_test.shape}")

from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# there is no window offseting, so we can shuffle data normally
# however i think this needs verification as signal can drift over time
# to check it uncomment the lines below and comapre the results
# if they are different, probably some preprocessing is needed

# split_point = int(0.9 * len(X))
# X_train, X_test = X[:split_point], X[split_point:]
# y_train, y_test = y_encoded[:split_point], y_encoded[split_point:]

print(f"Train data shape: {X_train.shape}")
print(f"Train labels shape: {y_train.shape}")
print(f"Sample of y_train: {y_train_encoded[0]}")


model, history = train_model(X_train, y_train_encoded, X_test, y_test_encoded)

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def evaluate_model(model, X_test, y_test):
    # Calculate accuracy
    y_pred = model.predict(X_test).argmax(axis=1)
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate AUC
    y_probs = model.predict(X_test)
    auc = roc_auc_score(y_test, y_probs, multi_class='ovr')

    # Classification report
    report = classification_report(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'auc': auc,
        'report': report
    }


results = evaluate_model(model, X_test, y_test)
print(f"Test Accuracy: {results['accuracy']:.2%}")
print(f"Test AUC: {results['auc']:.3f}")
print("\nClassification Report:\n", results['report'])