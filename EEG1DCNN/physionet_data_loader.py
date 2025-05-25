import mne
from pathlib import Path

class PhysionetDataLoader:
    """
    A class for loading and processing EEG data from the Physionet.
    """
    
    # Label mapping for the BCI Competition IV Dataset 2a
    LABEL_MAPPING = {
        'T0': 'R',      # Relaxation
        'T1': 'RH',     # Right hand squeeze
        'T2': 'LH',     # Left hand squeeze
        'T3': 'BH',     # Both hands
        'T4': 'BF',     # Both feet
        'T5': 'IRH',    # Imagined right hand squeeze
        'T6': 'ILH',    # Imagined left hand squeeze
        'T7': 'IBH',    # Imagined both hands
        'T8': 'IBF',    # Imagined both feet
        'T9': 'EO',     # Eyes open
        'T10': 'EC'     # Eyes closed
    }
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def load_subject_data(self, subject_id: str):

        self.current_subject = subject_id
        subject_dir = self.data_dir / subject_id
        edf_files = list(subject_dir.glob('*.edf'))

        if not edf_files:
            raise FileNotFoundError(f"No EDF files found for subject {subject_id}")
        
        raw_list = []
        for edf_file in edf_files:
            raw = mne.io.read_raw_edf(str(edf_file), preload=True)
            self._standardize_annotations(raw, str(edf_file))
            raw = self._setup_montage(raw)
            raw_list.append(raw)
        
        if len(raw_list) > 1:
            return mne.concatenate_raws(raw_list)
        else:
            return raw_list[0]

    def _standardize_annotations(self, raw: mne.io.Raw, file_path: str) -> None:
        """
        Standardize annotation descriptions to match our mapping, based on run information.
        
        According to the dataset paper, annotation codes have different meanings depending on the run:
        - T0 always corresponds to rest
        - T1 corresponds to:
          - Left fist (real/imagined) in runs 3, 4, 7, 8, 11, 12
          - Both fists (real/imagined) in runs 5, 6, 9, 10, 13, 14
        - T2 corresponds to:
          - Right fist (real/imagined) in runs 3, 4, 7, 8, 11, 12
          - Both feet (real/imagined) in runs 5, 6, 9, 10, 13, 14
          
        Imagined movements are in runs 7-14, real movements are in runs 3-6.
        """
        
        # Extract run number from file path (e.g., S001R03.edf -> run 3)
        import re
        run_match = re.search(r'R(\d+)\.edf', file_path, re.IGNORECASE)
        if run_match:
            run_num = int(run_match.group(1))
        else:
            print(f"Warning: Could not determine run number from file path: {file_path}")
            run_num = 0
        
        # Make a copy of the annotations to avoid modifying them during iteration
        new_annotations = []
        
        for annot in raw.annotations:
            desc = annot['description']
            new_desc = desc  # Default to keeping the original
            
            # T0 is always rest
            if desc == 'T0':
                new_desc = 'R'  # Rest/Relaxation
            
            # For T1 and T2, interpretation depends on the run number
            elif desc == 'T1':
                if 3 <= run_num <= 6:  # Real movements
                    if run_num in [3, 4]:
                        new_desc = 'LH'  # Left Hand
                    elif run_num in [5, 6]:
                        new_desc = 'BH'  # Both Hands
                elif 7 <= run_num <= 14:  # Imagined movements
                    if run_num in [7, 8, 11, 12]:
                        new_desc = 'ILH'  # Imagined Left Hand
                    elif run_num in [9, 10, 13, 14]:
                        new_desc = 'IBH'  # Imagined Both Hands
            
            elif desc == 'T2':
                if 3 <= run_num <= 6:  # Real movements
                    if run_num in [3, 4]:
                        new_desc = 'RH'  # Right Hand
                    elif run_num in [5, 6]:
                        new_desc = 'BF'  # Both Feet
                elif 7 <= run_num <= 14:  # Imagined movements
                    if run_num in [7, 8, 11, 12]:
                        new_desc = 'IRH'  # Imagined Right Hand
                    elif run_num in [9, 10, 13, 14]:
                        new_desc = 'IBF'  # Imagined Both Feet
            
            new_annotations.append({
                'onset': annot['onset'],
                'duration': annot['duration'],
                'description': new_desc
            })
        
        new_annot_obj = mne.Annotations(
            onset=[a['onset'] for a in new_annotations],
            duration=[a['duration'] for a in new_annotations],
            description=[a['description'] for a in new_annotations]
        )
        
        raw.set_annotations(new_annot_obj)

    def _setup_montage(self, raw_data: mne.io.Raw, on_missing: str = 'warn') -> mne.io.Raw:
        """
        Set up the correct electrode montage for the EEG data.
        
        This dataset uses a 64-channel setup following the international 10-10 system,
        excluding electrodes Nz, F9, F10, FT9, FT10, A1, A2, TP9, TP10, P9, and P10.
        
        Parameters:
        -----------
        raw_data : mne.io.Raw
            Raw EEG data to apply montage to
        on_missing : str
            How to handle missing channels: 'raise', 'warn', or 'ignore'
            
        Returns:
        --------
        mne.io.Raw
            Raw data with montage applied
        """
        # Use 10-05 montage as standard 10-10 is not available in MNE
        # 10-10 is a subset of 10-05, so we can use it for our montage
        montage = mne.channels.make_standard_montage('standard_1005')
        
        # Create a list of channel names from the raw data
        ch_names = raw_data.ch_names
        
        # Need to standardize the channel names to match the montage
        # 1. Remove trailing dots
        # 2. Convert to lowercase (montage names are all lowercase)
        clean_ch_names = [ch.rstrip('.').lower() for ch in ch_names]
        
        # Create a mapping from original names to standardized names
        # that match the montage's naming convention
        ch_mapping = {}
        montage_ch_names_lower = [ch.lower() for ch in montage.ch_names]
        
        for old_name, clean_name in zip(ch_names, clean_ch_names):
            if clean_name in montage_ch_names_lower:
                # Get the proper case from the montage
                idx = montage_ch_names_lower.index(clean_name)
                montage_name = montage.ch_names[idx]
                ch_mapping[old_name] = montage_name
        
        # Check if we found matches for all channels
        missing_channels = [ch for ch in ch_names if ch not in ch_mapping]
        if missing_channels:
            print(f"Warning: Could not find montage matches for {len(missing_channels)} channels: {missing_channels[:5]}...")
            print(f"Setting montage with on_missing='{on_missing}' to continue despite missing channels.")
        
        # Rename channels to match the montage
        raw_data.rename_channels(ch_mapping)
        
        # Apply the montage with specified on_missing parameter
        raw_data.set_montage(montage, on_missing=on_missing)
        
        print(f"Successfully applied 10-05 montage to {len(ch_mapping)} channels out of {len(ch_names)}")
        
        return raw_data