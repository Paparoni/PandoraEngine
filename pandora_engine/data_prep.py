# Simple Script to prepare wave data for model training
# Data from the Gravitational-Wave Open Science Center (GWOSC) - https://www.gw-openscience.org/
# Author: @paparoni (GitHub) Antwaun Tune Jr. (LinkedIn)

import h5py
import scipy

file = "G-G1_GWOSC_O3GK_16KHZ_R1-1270317056-4096.hdf5"  

# Load strain data into strain_data var
with h5py.File("data/"+file, 'r') as f:
    strain_data = f['strain'][:]

# Basic noise reduction
# We're going to use a bandpass filter since we only care for signals within a specific range of frequencies, we only want the frequencies where gravitational waves are expected betwen 10 and 1000 Hz
def bp_filter(data, low, high, fs, order=5):
    nyquist = 0.5 * fs
    low = low / nyquist
    high = high / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data =filtfilt(b, a, data)
    return filtered_data

# Segment data
# Here we are diving the the graviational wave signals into smaller segments to process more efficiently
window_size = 16384
def segment_data(data, window_size, overlap=0.5):
    step_size = int(window_size * (1 - overlap))
    segments = []
    for start in range(0, len(data) - window_size, step_size):
        window = data[start:start + window_size]
        segments.append(window)
    
    return segments
# Feature extraction (statistics)
def extract_statistics(data):
    return {
        'mean': np.mean(data),
        'variance': np.var(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }
segments = segment_data(strain_data, 16384)
features = [extract_statistics(segment) for segment in segments]

# Normalize data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(strain_data.reshape(-1, 1))
