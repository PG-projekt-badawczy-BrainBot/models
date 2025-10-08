from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report


from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf
import os
from utilz import channels_32, channels_16

def load_eeg_data(subject, base_path=''):
    raw_fnames = []
    subject_folder = f"S{subject:03d}"
    subject_path = os.path.join(base_path, subject_folder)
    for file in os.listdir(subject_path):
        full_path = os.path.join(subject_path, file)
        print(full_path)
        if not file.endswith('.edf'):
            continue

        raw_fnames.append(full_path)

    return concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])


data_path = r"C:\Users\kasia\projekt_badawczy\processed_eeg_data"
print(__doc__)
tmin, tmax = 2.0, 4.0
subjects = 1
raw = load_eeg_data(subject=subjects,
                    base_path=data_path)


eegbci.standardize(raw)
montage = make_standard_montage("standard_1005")
raw.set_montage(montage)
raw.set_eeg_reference(projection=True)
raw.pick_channels(channels_16)
raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

epochs = Epochs(
    raw,
    event_id=["RH", "LH", "BH", "BF", "R"],
    tmin=tmin,
    tmax=tmax,
    proj=True,
    picks=picks,
    baseline=None,
    preload=True,
)
epochs_train = epochs.copy().crop(tmin=tmin, tmax=tmax)

event_names = [list(epochs.event_id.keys())[list(epochs.event_id.values()).index(num)]
              for num in epochs.events[:, -1]]

labels_mapping = {"RH": 4, "LH": 3, "BH": 2, "BF": 1, "R": 0}
labels_mapping_binary = {"RH": 1, "LH": 1, "BH": 1, "BF": 1, "R": 0}
labels = np.array([labels_mapping[name] for name in event_names])
labels_binary = np.array([labels_mapping_binary[name] for name in event_names])

epochs_data = epochs.get_data(copy=False)
epochs_data_train = epochs_train.get_data(copy=False)
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)

svm_binary = OneVsRestClassifier(SVC(kernel='rbf', probability=True))
csp_binary = CSP(n_components=2, reg=None, log=True, norm_trace=False)
clf_binary = Pipeline([("CSP", csp_binary), ("LDA", LinearDiscriminantAnalysis())])

svm = OneVsRestClassifier(SVC(kernel='rbf', probability=True))
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
clf = Pipeline([("CSP", csp), ("SVM", svm)])

n_classes = 5
n_binary_classes = 2
confusion_mat = np.zeros((n_classes, n_classes), dtype=int)
confusion_mat_binary = np.zeros((n_binary_classes, n_binary_classes), dtype=int)
all_y_pred = []
all_y_true = []
all_y_pred_binary = []
all_y_true_binary = []

for train_index, test_index in cv.split(epochs_data_train):
    X_train = epochs_data_train[train_index]
    X_test = epochs_data_train[test_index]
    y_train = labels[train_index]
    y_test = labels[test_index]
    y_train_binary = labels_binary[train_index]
    y_test_binary = labels_binary[test_index]

    clf_binary.fit(X_train, y_train_binary)
    clf.fit(X_train[y_train_binary == 1], y_train[y_train_binary == 1])

    y_binary_proba = clf_binary.predict_proba(X_test)[:, 1]
    y_binary_pred = (y_binary_proba > 0.5).astype(int)
    all_y_true_binary.extend(y_test_binary)
    all_y_pred_binary.extend(y_binary_pred)

    movement_indices = np.where(y_binary_pred == 1)[0]
    if len(movement_indices) == 0:
        continue
    X_test_movement = X_test[movement_indices]
    y_test_movement = y_test[movement_indices]
    y_pred_movement = clf.predict(X_test_movement)
    print(f"Detected movement: {len(movement_indices)} / {y_test_binary.sum()}")


    all_y_true.extend(y_test_movement)
    all_y_pred.extend(y_pred_movement)

    for true, pred in zip(y_test_binary, y_binary_pred):
        confusion_mat_binary[true, pred] += 1

    for true, pred in zip(y_test_movement, y_pred_movement):
        confusion_mat[true, pred] += 1

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

all_y_true_binary = np.array(all_y_true_binary)
all_y_pred_binary = np.array(all_y_pred_binary)
accuracy_binary = np.mean(all_y_true_binary == all_y_pred_binary)
print(f"Binary classification accuracy: {accuracy_binary}")
print("\nMacierz pomyłek dla klasyfikacji binarnej:")
print(confusion_mat_binary)
print("\nRaport klasyfikacji dla klasyfikacji binarnej:")
print(classification_report(all_y_true_binary, all_y_pred_binary, labels=[0, 1], target_names=["NR", "R"]))

accuracy = np.mean(all_y_true == all_y_pred)
print(f"Classification accuracy: {accuracy}")

print("\nMacierz pomyłek:")
print(confusion_mat)
print("\nRaport klasyfikacji:")
print(classification_report(all_y_true, all_y_pred, labels=[1, 2, 3, 4],
    target_names=["BF", "BH", "LH", "RH"]))
csp_binary.fit_transform(epochs_data, labels)

csp_binary.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)
plt.show()
