import ROOT
from collections import OrderedDict
import numpy as np

p_data = "data/Sensl_FastOut_AveragePulse_1p8GHzBandwidth.root"
pfile = ROOT.TFile(p_data)
# pulse shape file per photon (?)
h_pulse = pfile.Get("hist2")

# convert pulse shape to numpy array
# not sure which one is faster: TH1 or numpy array
pulses = np.zeros(h_pulse.GetNbinsX())
for i in range(h_pulse.GetNbinsX()):
    pulses[i] = h_pulse.GetBinContent(i+1)

print("pulses: ", pulses)

sim_data = "/Users/yfeng/Desktop/mc_testjob_run001_003_Test_20evt_pi+_100.0_100.0.root"
ifile = ROOT.TFile(sim_data)
tree = ifile.Get("tree")

nFibers = 4
time_max = 30.0
time_per_bin = 0.04  # use 40 ps per bin
nBins = int(time_max / time_per_bin)

# pulses for truth photons and reco with shapes
histos_truth = OrderedDict()
histos_reco = OrderedDict()
for ievt in range(20):
    histos_truth[ievt] = OrderedDict()
    histos_reco[ievt] = OrderedDict()
    for i in range(nFibers):
        histos_truth[ievt][i] = ROOT.TH1D(
            f"h_truth_C_{ievt}Evt_{i}", f"h_truth_C_{ievt}Evt_{i}", nBins, 0, time_max)
        histos_reco[ievt][i] = ROOT.TH1D(
            f"h_reco_C_{ievt}Evt_{i}", f"h_reco_C_{ievt}Evt_{i}", nBins, 0, time_max)


def AddPulse(h_pulse_reco, t0):
    t0_bin = h_pulse_reco.FindBin(t0)
    for i in range(pulses.size):
        val = h_pulse_reco.GetBinContent(t0_bin + i)
        val += pulses[i]
        h_pulse_reco.SetBinContent(t0_bin + i, val)


nevts = tree.GetEntries()
for ievt in range(nevts):
    tree.GetEntry(ievt)

    nPhotons = tree.nOPs
    for j in range(nPhotons):

        # make it to the end of the fiber
        if not tree.OP_pos_final_z[j] > 50.0:
            continue
        # look at photons from core of Cherenkov cone
        if not tree.OP_isCoreC[j]:
            continue

        t = tree.OP_time_final[j]

        ifiber = tree.OP_productionFiber[j]

        if ifiber >= nFibers:
            print("found possible bug, ifiber: ", ifiber)

        # truth photon arriving time
        histos_truth[ievt][ifiber].Fill(t, 1.0)
        # reco-ed pulse shape
        AddPulse(histos_reco[ievt][ifiber], t)

# save to file
ofile = ROOT.TFile("output.root", "RECREATE")
for ievt in range(20):
    for ifiber in range(nFibers):
        histos_truth[ievt][ifiber].Write()
        histos_reco[ievt][ifiber].Write()
ofile.Close()
