import ROOT
from collections import OrderedDict
import numpy as np


#rootfiles to choose from (Sensl_FastOut_AveragePulse_1p8GHzBandwidth.root or different_pulses.root)
p_data = "../plotter/data/different_pulses.root"
pfile = ROOT.TFile(p_data)
# pulse shape file per photon (?)
h_pulse = pfile.Get("h_landau_noisy")  # Change to the desired pulse shape histogram

# convert pulse shape to numpy array
# not sure which one is faster: TH1 or numpy array
pulses = np.zeros(h_pulse.GetNbinsX())
for i in range(h_pulse.GetNbinsX()):
    pulses[i] = h_pulse.GetBinContent(i+1)

print("pulses: ", pulses)

try:
    with open("root_file_name.txt", "r") as f:
        sim_data = f.read().strip()
    ifile = ROOT.TFile(sim_data)
    tree = ifile.Get("tree")
except FileNotFoundError:
    print("Error: root_file_name.txt not found. Please ensure the file exists and contains the correct ROOT file name.")
    sys.exit()
except AttributeError:
    print(f"Error: Failed to open ROOT file {sim_data} or retrieve the tree. Please check the file path and contents.")
    sys.exit()

nFibers = 4
time_max = 30.0
time_per_bin = 0.04  # use 40 ps per bin
nBins = int(time_max / time_per_bin)
nevts = 10
# pulses for truth photons and reco with shapes
histos_truth = OrderedDict()
histos_reco = OrderedDict()
for ievt in range(nevts):
    histos_truth[ievt] = OrderedDict()
    histos_reco[ievt] = OrderedDict()
    for i in range(nFibers):
        histos_truth[ievt][i] = ROOT.TH1D(
            f"h_truth_C_{ievt}Evt_{i}", f"h_truth_C_{ievt}Evt_{i}", nBins, 0, time_max)
        histos_reco[ievt][i] = ROOT.TH1D(
            f"h_reco_C_{ievt}Evt_{i}", f"h_reco_C_{ievt}Evt_{i}", nBins, 0, time_max)


# Function to add a pulse to the reconstructed histogram
def AddPulse(h_pulse_reco, t0, pulses):
    t0_bin = h_pulse_reco.FindBin(t0)
    peak_index = np.argmax(pulses)  # Index of the pulse peak
    for i in range(len(pulses)):
        bin_index = t0_bin + (i - peak_index)  # Shift so peak aligns with t0_bin
        bin_index = int(bin_index)  # Ensure it's a Python int
        if 1 <= bin_index <= h_pulse_reco.GetNbinsX():
            val = h_pulse_reco.GetBinContent(bin_index)
            val += pulses[i]
            h_pulse_reco.SetBinContent(bin_index, val)


# Define spatial binning
x_min, x_max, n_xbins = -20, 20, 33  # X-axis range and number of bins
y_min, y_max, n_ybins = -20, 20, 25  # Y-axis range and number of bins
x_bin_width = (x_max - x_min) / n_xbins
y_bin_width = (y_max - y_min) / n_ybins

# Create histograms for each spatial bin and event
histos_truth = {}
histos_reco = {}
for ievt in range(nevts):  # Loop over events
    for ix in range(n_xbins):
        for iy in range(n_ybins):
            bin_key = (ievt, ix, iy)  # Include event index in the key
            histos_truth[bin_key] = ROOT.TH1D(
                f"h_truth_evt_{ievt}_bin_{ix}_{iy}",
                f"Truth Histogram for Event {ievt}, Bin ({ix}, {iy})",
                nBins, 0, time_max
            )
            histos_reco[bin_key] = ROOT.TH1D(
                f"h_reco_evt_{ievt}_bin_{ix}_{iy}",
                f"Reco Histogram for Event {ievt}, Bin ({ix}, {iy})",
                nBins, 0, time_max
            )

# Process events
nevts = 40
for ievt in range(nevts):
    tree.GetEntry(ievt)

    # Check if the branch `OP_time_final` has any entries for the event
    if len(tree.OP_time_final) == 0:
        print(f"Skipping event {ievt} as OP_time_final has no entries.")
        continue  # Skip this event if the branch is empty

    # Create a dictionary to track entries per bin
    bin_entries = {}

    nPhotons = tree.nOPs
    for j in range(nPhotons):
        # Make sure the photon reaches the end of the fiber
        if not tree.OP_pos_final_z[j] > 50.0:
            continue
        # Only consider photons from the core of the Cherenkov cone
        if not tree.OP_isCoreC[j]:
            continue

        # Get photon properties
        t = tree.OP_time_final[j]
        x = tree.OP_pos_final_x[j]
        y = tree.OP_pos_final_y[j]

        # Determine the spatial bin for the rod
        xbin = int((x - x_min) / x_bin_width)
        ybin = int((y - y_min) / y_bin_width)

        # Ensure the bin indices are within range
        if xbin < 0 or xbin >= n_xbins or ybin < 0 or ybin >= n_ybins:
            continue  # Skip if the photon is outside the defined spatial bins

        # Track entries for the bin
        bin_key = (ievt, xbin, ybin)
        if bin_key not in bin_entries:
            bin_entries[bin_key] = 0
        bin_entries[bin_key] += 1

        # Fill the truth histogram
        if bin_key not in histos_truth:
            histos_truth[bin_key] = ROOT.TH1D(
                f"h_truth_evt_{ievt}_bin_{xbin}_{ybin}",
                f"Truth Histogram for Event {ievt}, Bin ({xbin}, {ybin})",
                nBins, 0, time_max
            )
        histos_truth[bin_key].Fill(t)

        # Fill the reconstructed histogram with pulse reconstruction
        if bin_key not in histos_reco:
            histos_reco[bin_key] = ROOT.TH1D(
                f"h_reco_evt_{ievt}_bin_{xbin}_{ybin}",
                f"Reco Histogram for Event {ievt}, Bin ({xbin}, {ybin})",
                nBins, 0, time_max
            )
        AddPulse(histos_reco[bin_key], t, pulses)

# Save only histograms with entries
output_file = ROOT.TFile("output.root", "RECREATE")
for bin_key, hist in histos_truth.items():
    if hist.GetEntries() > 0:  # Save only if the histogram has entries
        hist.Write()
for bin_key, hist in histos_reco.items():
    if hist.GetEntries() > 0:  # Save only if the histogram has entries
        hist.Write()
output_file.Close()


