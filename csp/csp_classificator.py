from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from utilz import channels_32, channels_16


from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf
import os


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
tmin, tmax = -1.0, 4.0
subjects = 1
raw = load_eeg_data(subject=subjects,
                    base_path=data_path)

eegbci.standardize(raw)
montage = make_standard_montage("standard_1005")
raw.set_montage(montage)
raw.set_eeg_reference(projection=True)
#raw.pick_channels(channels_16)

raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

epochs = Epochs(
    raw,
    event_id=["RH", "LH", "BH", "BF"],
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

labels_mapping = {"RH": 3, "LH": 2, "BH": 1, "BF": 0}
labels = np.array([labels_mapping[name] for name in event_names])

epochs_data = epochs.get_data(copy=False)
epochs_data_train = epochs_train.get_data(copy=False)
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)
print("csv split")
print(cv_split)


lda = LinearDiscriminantAnalysis()
svm = OneVsRestClassifier(SVC(kernel='rbf', probability=True))
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
clf = Pipeline([("CSP", csp), ("SVM", svm)])

n_classes = len(np.unique(labels)) + 1
confusion_mat = np.zeros((n_classes, n_classes), dtype=int)
all_y_pred = []
all_y_true = []

for train_index, test_index in cv.split(epochs_data_train):
    X_train = epochs_data_train[train_index]
    X_test = epochs_data_train[test_index]
    y_train = labels[train_index]
    y_test = labels[test_index]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    for true, pred in zip(y_test, y_pred):
        confusion_mat[true, pred] += 1

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

accuracy = np.mean(all_y_true == all_y_pred)
print(f"Classification accuracy: {accuracy}")

print("\nMacierz pomyłek:")
print(confusion_mat)
print("\nRaport klasyfikacji:")
print(classification_report(all_y_true, all_y_pred, target_names=labels_mapping.keys()))
csp.fit_transform(epochs_data, labels)

csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)
plt.show()