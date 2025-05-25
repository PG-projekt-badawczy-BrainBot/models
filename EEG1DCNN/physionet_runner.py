from physionet_data_loader import PhysionetDataLoader
import mne
import numpy as np
from EEG1DCNN import build_model, train_model


mne.set_log_level('WARNING')
subject_id = 'S003'
data_dir = r"C:\Users\kasia\projekt_badawczy\eeg-motor-movementimagery-dataset-1.0.0\files"

data_loader = PhysionetDataLoader(data_dir)
raw = data_loader.load_subject_data(subject_id)

events, event_id = mne.events_from_annotations(raw)
print(f"Events: {events}")
print(f"Event IDs: {event_id}")

labels_to_consider = ['ILH', 'IRH', 'IBH', 'IBF']
events_to_consider = [event_id[label] for label in labels_to_consider]
epochs = mne.Epochs(
    raw, events, event_id=events_to_consider,
    tmin=0.0, tmax=4.0,
    preload=True,
    baseline=None,
    reject_by_annotation=True
)


epochs_data = epochs.get_data()
print(f"Original epochs shape: {epochs_data.shape}")

# Trim the last sample to change shape from (4, 64, 641) to (4, 64, 640)
epochs_data = epochs_data[:, :, :-1]
print(f"Trimmed epochs shape: {epochs_data.shape}")

epochs = mne.EpochsArray(epochs_data, epochs.info,
                        events=epochs.events,
                        tmin=epochs.tmin,
                        event_id=epochs.event_id)

print(f"Epochs shape: {epochs.get_data().shape}")
for label in labels_to_consider:
    print(f"Number of epochs for {label}: {len(epochs[event_id[label]])}")
    print(f"Epochs shape for {label}: {epochs[event_id[label]].get_data().shape}")

# Inspect data
raw.plot(n_channels=10, duration=5, start=0, show=True)

# Plot topomaps of each class
for label in labels_to_consider:
    mi = event_id[label]
    print(f"Plotting topomap for {label} (event ID: {mi})")
    epochs[mi].compute_psd().plot_topomap()


electrode_pairs = [
    ('FC5', 'FC6'), ('FC3', 'FC4'), ('FC1', 'FC2'),
    ('C5', 'C6'), ('C3', 'C4'), ('C1', 'C2'),
    ('CP5', 'CP6'), ('CP3', 'CP4'), ('CP1', 'CP2')
]


def preprocess_data(epochs):
    """Convert MNE epochs to CNN-ready format with electrode pairs."""
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    labels = epochs.events[:, -1]

    X, y = [], []
    for epoch_idx in range(data.shape[0]):
        epoch_data = data[epoch_idx]  # (n_channels, 640)
        for ch1, ch2 in electrode_pairs:
            # Get channel indices
            idx1 = epochs.ch_names.index(ch1)
            idx2 = epochs.ch_names.index(ch2)

            # Concatenate and reshape to (640, 2)
            pair_data = np.hstack([epoch_data[idx1], epoch_data[idx2]]).reshape(640, 2)
            X.append(pair_data)
        y.extend([labels[epoch_idx]] * len(electrode_pairs))  # Same label for all pairs

    X = np.array(X)[..., np.newaxis]  # Add channel dimension (n_samples, 640, 2, 1)
    y = np.array(y)
    return X, y


X, y = preprocess_data(epochs)
print(f"Preprocessed data shape: {X.shape}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.1, stratify=y_encoded, random_state=11
)

# there is no window offseting, so we can shuffle data normally
# however i think this needs verification as signal can drift over time
# to check it uncomment the lines below and comapre the results
# if they are different, probably some preprocessing is needed

# split_point = int(0.9 * len(X))
# X_train, X_test = X[:split_point], X[split_point:]
# y_train, y_test = y_encoded[:split_point], y_encoded[split_point:]

print(f"Train data shape: {X_train.shape}, Train labels shape: {y_train.shape}")
print(f"Sample of X_train: {X_train[0]}")
print(f"Sample of y_train: {y_train[0]}")


model, history = train_model(X_train, y_train, X_test, y_test)

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def evaluate_model(model, X_test, y_test):
    # Calculate accuracy
    y_pred = model.predict(X_test).argmax(axis=1)
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate AUC
    y_probs = model.predict(X_test)
    auc = roc_auc_score(y_test, y_probs, multi_class='ovr')

    # Classification report
    report = classification_report(y_test, y_pred, target_names=['T1', 'T2', 'T3', 'T4'])

    return {
        'accuracy': accuracy,
        'auc': auc,
        'report': report
    }


results = evaluate_model(model, X_test, y_test)
print(f"Test Accuracy: {results['accuracy']:.2%}")
print(f"Test AUC: {results['auc']:.3f}")
print("\nClassification Report:\n", results['report'])