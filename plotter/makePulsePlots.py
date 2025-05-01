import sys
from collections import OrderedDict
import ROOT
sys.path.append("./CMSPLOTS")
from myFunction import DrawHistos  # noqa

ROOT.gROOT.SetBatch(True)

# Draw single pulse
p_data = "data/Sensl_FastOut_AveragePulse_1p8GHzBandwidth.root"
pfile = ROOT.TFile(p_data)
# pulse shape file per photon (?)
h_pulse = pfile.Get("hist2")

args = {
    'dology': True,
    'donormalize': False,
    'mycolors': [1],
    "MCOnly": True,
    'addOverflow': True,
    'addUnderflow': True,
    'mycolors': [2, 3, 4, 6]
}

DrawHistos([h_pulse], ["Single Pulse"], 0, 20, "Measured Time [ns]",
           1, 2.5e5, "Voltage [ns]", "SinglePulse_log", **args)
args['dology'] = False
DrawHistos([h_pulse], ["Single Pulse"], 0, 20, "Measured Time [ns]",
           -2e4, 2.5e5, "Voltage [ns]", "SinglePulse", **args)

fname_pulses = "output.root"
f_pulses = ROOT.TFile(fname_pulses)

args['dology'] = False

# truth photon time
histos_truth = OrderedDict()
for ievt in range(20):
    histos_truth[ievt] = OrderedDict()

    for iFiber in range(4):
        hname = f"h_truth_C_{ievt}Evt_{iFiber}"
        histos_truth[ievt][iFiber] = f_pulses.Get(hname)

    DrawHistos(list(histos_truth[ievt].values()), list(histos_truth[ievt].keys()), 5, 20, "Measured Time [ns]",
               0, 25, "Voltage [mV]", f"Truth_{ievt}", **args)

args['dology'] = True

# reco photon time
histos_reco = OrderedDict()
for ievt in range(20):
    histos_reco[ievt] = OrderedDict()

    for iFiber in range(4):
        hname = f"h_reco_C_{ievt}Evt_{iFiber}"
        histos_reco[ievt][iFiber] = f_pulses.Get(hname)

    DrawHistos(list(histos_reco[ievt].values()), list(histos_reco[ievt].keys()), 10, 30, "Measured Time [ns]",
               1, 3e10, "Voltage [mV]", f"Reco_{ievt}", **args)
