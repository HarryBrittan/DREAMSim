import ROOT
from collections import OrderedDict
import numpy as np
import h5py
import sys

print("Loading pulse shape from ROOT file...")
# Load pulse shape from ROOT file
p_data = "../../plotter/data/different_pulses.root"
pfile = ROOT.TFile(p_data)
h_pulse = pfile.Get("h_landau")  # Change to the desired pulse shape histogram

# Convert pulse shape to numpy array
pulses = np.zeros(h_pulse.GetNbinsX())
for i in range(h_pulse.GetNbinsX()):
    pulses[i] = h_pulse.GetBinContent(i+1)
pulse_len = len(pulses)
pulse_peak = np.argmax(pulses)

print("Reading ROOT file names from text file...")
# Read all ROOT file names from the text file (one per line)
try:
    with open("../../interactive/root_file_name.txt", "r") as f:
        root_files = [line.strip() for line in f if line.strip()]
    if not root_files:
        print("Error: No ROOT files listed in root_file_name.txt.")
        sys.exit()
    # Create a TChain and add all files
    tree = ROOT.TChain("tree")
    for fname in root_files:
        tree.Add(fname)
except FileNotFoundError:
    print("Error: root_file_name.txt not found. Please ensure the file exists and contains the correct ROOT file names.")
    sys.exit()
except Exception as e:
    print(f"Error: {e}")
    sys.exit()

# Parameters
time_max = 30.0
time_per_bin = 0.04  # 40 ps per bin
nBins = int(time_max / time_per_bin)
nevts = tree.GetEntries()  # Use the number of entries in the tree
#nevts = 50  # For testing, use a smaller number of events
x_min, x_max, n_xbins = -20, 20, 33
y_min, y_max, n_ybins = -20, 20, 25
x_bin_width = (x_max - x_min) / n_xbins
y_bin_width = (y_max - y_min) / n_ybins
NOISE_STD = 0.05

# Prepare arrays for HDF5 output
truth_list = []
convoluted_list = []

print(f"Starting event loop over {nevts} events...")
for ievt in range(nevts):
    tree.GetEntry(ievt)
    if ievt % 50 == 0:
        print(f"  Processing event {ievt+1}/{nevts}...")
    # Prepare 2D arrays for this event (flattened for storage)
    for ix in range(n_xbins):
        for iy in range(n_ybins):
            truth = np.zeros(nBins, dtype=np.float32)
            convoluted = np.zeros(nBins, dtype=np.float32)
            # Loop over photons in this event
            nPhotons = tree.nOPs
            for j in range(nPhotons):
                # Apply selection
                if not tree.OP_pos_final_z[j] > 50.0:
                    continue
                if not tree.OP_isCoreC[j]:
                    continue
                # Get photon properties
                t = tree.OP_time_final[j]
                x = tree.OP_pos_final_x[j]
                y = tree.OP_pos_final_y[j]
                # Determine spatial bin
                xbin = int((x - x_min) / x_bin_width)
                ybin = int((y - y_min) / y_bin_width)
                if xbin != ix or ybin != iy:
                    continue
                if xbin < 0 or xbin >= n_xbins or ybin < 0 or ybin >= n_ybins:
                    continue
                # Fill truth (delta function at photon time)
                t_bin = int(t / time_per_bin)
                if 0 <= t_bin < nBins:
                    truth[t_bin] += 1
                    # Overlay pulse shape (convolution)
                    start = t_bin - pulse_peak
                    for k in range(pulse_len):
                        bin_idx = start + k
                        if 0 <= bin_idx < nBins:
                            convoluted[bin_idx] += pulses[k]
            # Add noise to the convoluted signal (same as in pulse_gen.py)
            convoluted += np.random.normal(0, NOISE_STD, size=convoluted.shape)

            # Only save if there is at least one hit
            if np.any(truth):
                truth_list.append(truth)
                convoluted_list.append(convoluted)

# Convert to arrays
print("Converting lists to numpy arrays...")
truth_arr = np.array(truth_list, dtype=np.float32)
convoluted_arr = np.array(convoluted_list, dtype=np.float32)

# Save to HDF5
print("Saving arrays to mu20000evt_sim_pulse_data.h5...")
with h5py.File("mu20000evt_sim_pulse_data.h5", "w") as f:
    f.create_dataset("truth", data=truth_arr)
    f.create_dataset("convoluted", data=convoluted_arr)

print(f"Saved {truth_arr.shape[1]} samples to mu20000evt_sim_pulse_data.h5")



