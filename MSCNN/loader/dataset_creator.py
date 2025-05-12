import edfio
import numpy as np
import mne

class EDFDatasetCreator:
    def __init__(self, edf_path: str, window_size: int = 3000, overlap: int = 0):
        self.edf_path = edf_path
        self.window_size = window_size
        self.overlap = overlap
        self.signals, self.fs = self._read_edf()
        self.annots = self._read_mne_annotations()

    def _read_edf(self):
        reader = edfio.read_edf(self.edf_path)
        signals = [reader.signals[i].data for i in range(reader.num_signals)]
        signals = np.stack(signals, axis=-1)  # shape: (samples, channels)
        fs = reader.signals[0].sampling_frequency
        return signals, fs

    def _read_mne_annotations(self):
        raw = mne.io.read_raw_edf(self.edf_path, preload=False, verbose=False)
        annots = raw.annotations
        parsed = []
        for onset, duration, desc in zip(annots.onset, annots.duration, annots.description):
            parsed.append({
                "onset": int(onset * self.fs),       # convert to sample index
                "duration": int(duration * self.fs),
                "description": desc
            })
        return parsed

    def _get_segments(self):
        segments = []
        step = self.window_size - self.overlap
        num_samples = self.signals.shape[0]

        for start in range(0, num_samples - self.window_size + 1, step):
            end = start + self.window_size
            segment = self.signals[start:end, :]  # shape: (window_size, 32)
            label = self._label_segment(start, end)
            if label is not None:
                segments.append((segment, label))
        return segments

    def _label_segment(self, start: int, end: int) -> str | None:
        for annot in self.annots:
            onset = int(annot['onset'])
            duration = int(annot.get('duration', 0))
            if start >= onset and end <= onset + duration:
                return annot['description']
        return None

    def create_dataset(self):
        segments = self._get_segments()
        X = np.array([seg[0] for seg in segments])  # shape: (N, window_size, 32)
        y = [seg[1] for seg in segments]
        return X, y
