# Simple script to visualize the raw strain data from a HDF5 file
# Data from the Gravitational-Wave Open Science Center (GWOSC) - https://www.gw-openscience.org/
# Author: @paparoni (GitHub) Antwaun Tune Jr. (LinkedIn)

import h5py
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

def plot_strain_data(filename, data_folder="data/"):
    file_path = f"{data_folder}{filename}"
    
    with h5py.File(file_path, 'r') as f:
        print("HDF5 file structure:")
        f.visit(print)
        strain = f['strain']['Strain'][:]
        sample_rate = f['strain']['Strain'].attrs['Xspacing']  # Time step between samples

    # Create time axis
    time = np.arange(len(strain)) * sample_rate
    strain_series = TimeSeries(strain, dt=sample_rate)

    # Plot the strain data
    plt.figure(figsize=(10, 4))
    plt.plot(time, strain, lw=0.5)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Strain")
    plt.title(f"Raw Gravitational Wave Strain Data ({filename})")
    plt.grid()
    plt.show()

