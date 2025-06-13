import ROOT
import h5py
import numpy as np

file = ROOT.TFile("../interactive/output.root", "READ")

X = []  # Input: convoluted pulse
Y = []  # Target: truth pulse

# Loop over all keys in the file
for key in file.GetListOfKeys():
    obj = key.ReadObj()
    if not isinstance(obj, ROOT.TH1D):
        continue
    name = obj.GetName()
    if "reco" in name:
        truth_name = name.replace("reco", "truth")
        truth_hist = file.Get(truth_name)
        if truth_hist and isinstance(truth_hist, ROOT.TH1D):
            print(f"Matched: {name} <-> {truth_name}")
            conv = np.array([obj.GetBinContent(i+1) for i in range(obj.GetNbinsX())])
            truth = np.array([truth_hist.GetBinContent(i+1) for i in range(truth_hist.GetNbinsX())])
            X.append(conv)
            Y.append(truth)

X = np.array(X)
Y = np.array(Y)

print(f"Number of matched events: {len(X)}")

with h5py.File("deconvolution_data.h5", "w") as f:
    f.create_dataset("convoluted", data=X)
    f.create_dataset("truth", data=Y)
