import ROOT
from collections import OrderedDict, defaultdict
import sys
import math
sys.path.append("./CMSPLOTS")
from myFunction import DrawHistos

doOP = True


print("Starting")

ROOT.gSystem.Load("macros/functions_cc.so")

ROOT.gROOT.SetBatch(True)
ROOT.ROOT.EnableImplicitMT(8)

elefile = "inputs/optics/electrons.txt"
pionfile = "inputs/optics/pions.txt"
opfile = "inputs/optics/ops.txt"

loops = [('ele', elefile), ('pion', pionfile)]
#loops = [('ele', elefile)]
if doOP:
    loops = [('op', opfile)]

rdfs = OrderedDict()
chains = OrderedDict()

for part, filename in loops:
    chains[part] = ROOT.TChain("tree")
    nfiles = 0
    with open(filename) as f:
        print(f"Reading {filename}")
        elefiles = f.readlines()
        for line in elefiles:
            line = line.strip()
            if line.startswith("#"):
                continue
            nfiles += 1
            print(f"{part} " + line)
            chains[part].Add(line)

            if nfiles > 100:
                break
    rdfs[part] = ROOT.RDataFrame(chains[part])

nEvts = OrderedDict()
for part, _ in loops:
    nEvts[part] = rdfs[part].Count().GetValue()
    nEvts[part] = float(nEvts[part])
    print(f"Number of events for {part}: ", nEvts[part])
    
    if doOP:
        # for OPs, it is always 1 OP per event
        # no need to normalize
        nEvts[part] = 1.0

    rdfs[part] = rdfs[part].Define("OP_passEnd", "OP_pos_final_z > 49.0")
    rdfs[part] = rdfs[part].Define("eWeight", " OP_passEnd / " + str(nEvts[part]))
    rdfs[part] = rdfs[part].Define("eWeightTotal", "1.0 / " + str(nEvts[part]))
    rdfs[part] = rdfs[part].Define("eWeight_long", " eWeight * (OP_pos_produced_z < 0.0)")
    rdfs[part] = rdfs[part].Define("eWeightTotal_long", " eWeightTotal * (OP_pos_produced_z < 0.0)")
    rdfs[part] = rdfs[part].Define("OP_time_delta", "OP_time_final - OP_time_produced") # in ns
    rdfs[part] = rdfs[part].Define("OP_pos_delta_z", "OP_pos_final_z - OP_pos_produced_z")  # in cm
    rdfs[part] = rdfs[part].Define("OP_time_per_meter", "OP_time_delta / OP_pos_delta_z * 100.0") # in ns/m
    rdfs[part] = rdfs[part].Define("OP_mom_produced", "sqrt(OP_mom_produced_x*OP_mom_produced_x + OP_mom_produced_y*OP_mom_produced_y + OP_mom_produced_z*OP_mom_produced_z)")
    rdfs[part] = rdfs[part].Define("OP_cosTheta_produced", "OP_mom_produced_z / OP_mom_produced")
    rdfs[part] = rdfs[part].Define("OP_cosThetaInv_produced", "1.0 / OP_cosTheta_produced")
    rdfs[part] = rdfs[part].Define("OP_time_per_meter_cosTheta_produced", "OP_time_per_meter * OP_cosTheta_produced")
    rdfs[part] = rdfs[part].Define("OP_pos_produced_r", "sqrt(OP_pos_produced_x*OP_pos_produced_x + OP_pos_produced_y*OP_pos_produced_y)")
    rdfs[part] = rdfs[part].Define("OP_pos_final_r", "sqrt(OP_pos_final_x*OP_pos_final_x + OP_pos_final_y*OP_pos_final_y)")

    # Get the number of entries in OP_time_final
    nEntries_OP_time_final = rdfs[part].Count().GetValue()
    
    # Define OP_time_final_normalized
    rdfs[part] = rdfs[part].Define("OP_time_final_normalized", f"OP_time_final / {nEntries_OP_time_final}")


    
    # cladding and core
    rdfs[part] = rdfs[part].Define("OP_pos_produced_core", "OP_pos_produced_r < 0.039")
    rdfs[part] = rdfs[part].Define("OP_pos_produced_clad", "OP_pos_produced_r > 0.039 && OP_pos_produced_r < 0.040")
    rdfs[part] = rdfs[part].Define("OP_pos_produced_out", "OP_pos_produced_r > 0.040")
    rdfs[part] = rdfs[part].Define("OP_pos_final_core", "OP_pos_final_r < 0.039")
    rdfs[part] = rdfs[part].Define("OP_pos_final_clad", "OP_pos_final_r > 0.039 && OP_pos_final_r < 0.040")
    rdfs[part] = rdfs[part].Define("OP_pos_final_out", "OP_pos_final_r > 0.040")

    for i,f in [(0,15), (0, 9), (0, 10), (0, 11), (0, 12)]:
        i_str = str(i).replace(".", "_")
        f_str = str(f).replace(".", "_")
        rdfs[part] = rdfs[part].Define(f"eWeight_time_between_{i_str}_and_{f_str}_ns", f"eWeight * (OP_time_delta > {i} && OP_time_delta < {f})")

    
    
    # sinAlpha
    # the angle between the momentum and the radial vector
    # incident angle at the core-cladding interface
    rdfs[part] = rdfs[part].Define("OP_pos_produced_sinAlpha", "SinTheta(OP_pos_produced_x,OP_pos_produced_y,OP_pos_produced_z,OP_mom_produced_x,OP_mom_produced_y,OP_mom_produced_z)")
    
    for place in ["clad", "core", "out"]:
        rdfs[part] = rdfs[part].Define(f"eWeight_{place}", f"eWeight * OP_pos_produced_{place}")
        rdfs[part] = rdfs[part].Define(f"eWeightTotal_{place}", f"eWeightTotal * OP_pos_produced_{place}")

histos = defaultdict(OrderedDict)

# for event displays
evtlist = [1, 3, 5, 10, 15]
evtlist = []

x_range = 0.045
nx_bins = 100
px_range = 1e-8
if doOP:
    px_range = 1e-6
t_range = 20

for part, rdf in rdfs.items():
    suffix = "_" + part

    
    for i, f in [(0, 13), (0, 8), (0, 6), (0, 5.8), (0, 5.5)]:
        i_str = str(i).replace(".", "_")
        f_str = str(f).replace(".", "_")
        histos[f"OP_pos_produced_x_vs_y_{i_str}_and_{f_str}_ns"][part] = rdf.Histo2D(
            (f"OP_pos_produced_x_vs_y_{i_str}_and_{f_str}_ns" + suffix, f"OP_pos_produced_x_vs_y_{i_str}_and_{f_str}_ns", nx_bins, -x_range, x_range, nx_bins, -x_range, x_range), "OP_pos_produced_x", "OP_pos_produced_y", f"eWeight_time_between_{i_str}_and_{f_str}_ns")
    
    histos['nOPs'][part] = rdf.Histo1D(
        ("nOPs" + suffix, "nOPs", 100, 0, 10000), "nOPs")
    histos['OP_time_produced'][part] = rdf.Histo1D(
        ("OP_time_produced" + suffix, "OP_time_produced", 70, 0, .15), "OP_time_produced", "eWeight")
    histos['OP_time_final'][part] = rdf.Histo1D(
        ("OP_time_final" + suffix, "OP_time_final", 100, 0, t_range), "OP_time_final", "eWeight")
    histos['OP_time_final_normalized'][part] = rdf.Histo1D(
        ("OP_time_final_normalized" + suffix, "OP_time_final_normalized", 200, 0, 0.002), "OP_time_final_normalized", "eWeight")
    histos['OP_time_delta'][part] = rdf.Histo1D(
        ("OP_time_delta" + suffix, "OP_time_delta", 100, 0, t_range), "OP_time_delta", "eWeight")
    histos["OP_pos_delta_z"][part] = rdf.Histo1D(
        ("OP_pos_delta_z" + suffix, "OP_pos_delta_z", 100, 0, 100.0), "OP_pos_delta_z", "eWeight")
    histos["OP_time_per_meter"][part] = rdf.Histo1D(
        ("OP_time_per_meter" + suffix, "OP_time_per_meter", 100, 0, 12.0), "OP_time_per_meter", "eWeight")
    histos["OP_cosTheta_produced"][part] = rdf.Histo1D(
        ("OP_cosTheta_produced" + suffix, "OP_cosTheta_produced", 100, 0, 1), "OP_cosTheta_produced", "eWeight")
    histos["OP_cosTheta_produced_total"][part] = rdf.Histo1D(
        ("OP_cosTheta_produced_total" + suffix, "OP_cosTheta_produced_total", 100, 0, 1), "OP_cosTheta_produced", "eWeightTotal")
    histos["OP_cosTheta_produced_clad"][part] = rdf.Histo1D(
        ("OP_cosTheta_produced_clad" + suffix, "OP_cosTheta_produced_clad", 100, 0, 1), "OP_cosTheta_produced", "eWeight_clad")
    histos["OP_cosTheta_produced_core"][part] = rdf.Histo1D(
        ("OP_cosTheta_produced_core" + suffix, "OP_cosTheta_produced_core", 100, 0, 1), "OP_cosTheta_produced", "eWeight_core")
    histos["OP_cosTheta_produced_clad_total"][part] = rdf.Histo1D(
        ("OP_cosTheta_produced_clad_total" + suffix, "OP_cosTheta_produced_clad_total", 100, 0, 1), "OP_cosTheta_produced", "eWeightTotal_clad")
    histos["OP_cosTheta_produced_core_total"][part] = rdf.Histo1D(
        ("OP_cosTheta_produced_core_total" + suffix, "OP_cosTheta_produced_core_total", 100, 0, 1), "OP_cosTheta_produced", "eWeightTotal_core")
    histos["OP_time_per_meter_cosTheta_produced"][part] = rdf.Histo1D(
        ("OP_time_per_meter_cosTheta_produced" + suffix, "OP_time_per_meter_cosTheta_produced", 100, 4.5, 6.5), "OP_time_per_meter_cosTheta_produced", "eWeight")
    histos["OP_pos_produced_r"][part]  = rdf.Histo1D(
        ("OP_pos_produced_r" + suffix, "OP_pos_produced_r", 100, 0, 0.04), "OP_pos_produced_r", "eWeight")
    histos["OP_pos_produced_r_total"][part]  = rdf.Histo1D(
        ("OP_pos_produced_r_total" + suffix, "OP_pos_produced_r_total", 100, 0, 0.04), "OP_pos_produced_r", "eWeightTotal")

    histos["OP_pos_produced_x_vs_y"][part] = rdf.Histo2D(
        ("OP_pos_produced_x_vs_y" + suffix, "OP_pos_produced_x_vs_y", nx_bins, -x_range, x_range, nx_bins, -x_range, x_range), "OP_pos_produced_x", "OP_pos_produced_y", "eWeight")
    histos["OP_pos_produced_x_vs_y_total"][part] = rdf.Histo2D(
        ("OP_pos_produced_x_vs_y_total" + suffix, "OP_pos_produced_x_vs_y_total", nx_bins, -x_range, x_range, nx_bins, -x_range, x_range), "OP_pos_produced_x", "OP_pos_produced_y", "eWeightTotal")
    histos["OP_pos_final_x_vs_y"][part] = rdf.Histo2D(
        ("OP_pos_final_x_vs_y" + suffix, "OP_pos_final_x_vs_y", nx_bins, -x_range, x_range, nx_bins, -x_range, x_range), "OP_pos_final_x", "OP_pos_final_y", "eWeight")
    histos["OP_mom_produced_x_vs_y"][part] = rdf.Histo2D(("OP_mom_produced_x_vs_y" + suffix, "OP_mom_produced_x_vs_y",
                                                         nx_bins, -px_range, px_range, nx_bins, -px_range, px_range), "OP_mom_produced_x", "OP_mom_produced_y", "eWeight")
    histos["OP_mom_final_x_vs_y"][part] = rdf.Histo2D(("OP_mom_final_x_vs_y" + suffix, "OP_mom_final_x_vs_y",
                                                      nx_bins, -px_range, px_range, nx_bins, -px_range, px_range), "OP_mom_final_x", "OP_mom_final_y", "eWeight")
    histos["OP_mom_produced_x_vs_y_total"][part] = rdf.Histo2D(("OP_mom_produced_x_vs_y_total" + suffix, "OP_mom_produced_x_vs_y_total",
                                                                nx_bins, -px_range, px_range, nx_bins, -px_range, px_range), "OP_mom_produced_x", "OP_mom_produced_y", "eWeightTotal")
    
    histos["OP_time_per_meter_vs_cosTheta_produced"][part] = rdf.Histo2D(("OP_time_per_meter_vs_cosTheta_produced" + suffix, "OP_time_per_meter_vs_cosTheta_produced",
                                                         100, 0, 12, 100, 1, 3.0), "OP_time_per_meter", "OP_cosThetaInv_produced", "eWeight")
    
    histos["OP_cosTheta_vs_r_produced"][part] = rdf.Histo2D(("OP_cosTheta_vs_r_produced" + suffix, "OP_cosTheta_vs_r_produced",
                                                         100, 0, 0.04, 100, 0, 1), "OP_pos_produced_r", "OP_cosTheta_produced", "eWeight")
    
    histos["OP_cosTheta_vs_r_produced_total"][part] = rdf.Histo2D(("OP_cosTheta_vs_r_total" + suffix, "OP_cosTheta_vs_r_produced_total", 100, 0, 0.04, 100, 0, 1), "OP_pos_produced_r", "OP_cosTheta_produced", "eWeightTotal")
    
    histos["OP_sinAlpha_vs_r_produced"][part] = rdf.Histo2D(("OP_sinAlpha_vs_r_produced" + suffix, "OP_sinAlpha_vs_r_produced",
                                                         100, 0, 0.04, 100, 0, 1), "OP_pos_produced_r", "OP_pos_produced_sinAlpha", "eWeight")
    
    histos["OP_sinAlpha_vs_r_produced_total"][part] = rdf.Histo2D(("OP_sinAlpha_vs_r_total" + suffix, "OP_sinAlpha_vs_r_produced_total", 100, 0, 0.04, 100, 0, 1), "OP_pos_produced_r", "OP_pos_produced_sinAlpha", "eWeightTotal")
    
    histos["OP_sinAlpha_vs_r_produced_long"][part] = rdf.Histo2D(("OP_sinAlpha_vs_r_produced_long" + suffix, "OP_sinAlpha_vs_r_produced_long", 100, 0, 0.04, 100, 0, 1), "OP_pos_produced_r", "OP_pos_produced_sinAlpha", "eWeight_long")
    
    histos["OP_sinAlpha_vs_r_produced_long_total"][part] = rdf.Histo2D(("OP_sinAlpha_vs_r_produced_long_total" + suffix, "OP_sinAlpha_vs_r_produced_long_total", 100, 0, 0.04, 100, 0, 1), "OP_pos_produced_r", "OP_pos_produced_sinAlpha", "eWeightTotal_long")
    
    histos["OP_profilePx_produced_x_vs_y"][part] = rdf.Profile2D(("OP_profilePx_produced_x_vs_y" + suffix, "OP_profilePx_produced_x_vs_y",nx_bins, -x_range, x_range, nx_bins, -x_range, x_range), "OP_pos_produced_x", "OP_pos_produced_y", "OP_mom_produced_x", "eWeight")
    histos["OP_profilePy_produced_x_vs_y"][part] = rdf.Profile2D(("OP_profilePy_produced_x_vs_y" + suffix, "OP_profilePy_produced_x_vs_y",nx_bins, -x_range, x_range, nx_bins, -x_range, x_range), "OP_pos_produced_x", "OP_pos_produced_y", "OP_mom_produced_y", "eWeight")
    histos["OP_profilePx_final_x_vs_y"][part] = rdf.Profile2D(("OP_profilePx_final_x_vs_y" + suffix, "OP_profilePx_final_x_vs_y",nx_bins, -x_range, x_range, nx_bins, -x_range, x_range), "OP_pos_final_x", "OP_pos_final_y", "OP_mom_final_x", "eWeight")
    histos["OP_profilePy_final_x_vs_y"][part] = rdf.Profile2D(("OP_profilePy_final_x_vs_y" + suffix, "OP_profilePy_final_x_vs_y",nx_bins, -x_range, x_range, nx_bins, -x_range, x_range), "OP_pos_final_x", "OP_pos_final_y", "OP_mom_final_y", "eWeight")
    
    histos["OP_profilePx_produced_x_vs_y_total"][part] = rdf.Profile2D(("OP_profilePx_produced_x_vs_y_total" + suffix, "OP_profilePx_produced_x_vs_y_total",nx_bins, -x_range, x_range, nx_bins, -x_range, x_range), "OP_pos_produced_x", "OP_pos_produced_y", "OP_mom_produced_x", "eWeightTotal")
    histos["OP_profilePy_produced_x_vs_y_total"][part] = rdf.Profile2D(("OP_profilePy_produced_x_vs_y_total" + suffix, "OP_profilePy_produced_x_vs_y_total",nx_bins, -x_range, x_range, nx_bins, -x_range, x_range), "OP_pos_produced_x", "OP_pos_produced_y", "OP_mom_produced_y", "eWeightTotal")
    
    # some event displays
    for i in evtlist:
        rdf_event = rdf.Filter(f"rdfentry_ == {i}")
        histos[f"event_{i}_OP_pos_produced_x_vs_y"][part] = rdf_event.Histo2D((f"event_{i}_produced_x_vs_y" + suffix,
                                                                               f"event_{i}_OP_pos_produced_x_vs_y", 50, -x_range, x_range, 50, -x_range, x_range), "OP_pos_produced_x", "OP_pos_produced_y")
        histos[f"event_{i}_OP_pos_final_x_vs_y"][part] = rdf_event.Histo2D((f"event_{i}_final_x_vs_y" + suffix,
                                                                            f"event_{i}_OP_pos_final_x_vs_y", 50, -x_range, x_range, 50, -x_range, x_range), "OP_pos_final_x", "OP_pos_final_y")
        histos[f"event_{i}_OP_mom_produced_x_vs_y"][part] = rdf_event.Histo2D((f"event_{i}_produced_px_vs_py" + suffix,
                                                                               f"event_{i}_OP_mom_produced_x_vs_y", 50, -px_range, px_range, 50, -px_range, px_range), "OP_mom_produced_x", "OP_mom_produced_y")
        histos[f"event_{i}_OP_mom_final_x_vs_y"][part] = rdf_event.Histo2D((f"event_{i}_final_px_vs_py" + suffix,
                                                                            f"event_{i}_OP_final_x_vs_y", 50, -px_range, px_range, 50, -px_range, px_range), "OP_mom_final_x", "OP_mom_final_y")

colormaps = {
    'ele': 2,
    'pion': 3,
    "op": 4,
}

def GetColors(ene_fracs):
    colors = []
    for str in ene_fracs.keys():
        part, _ = str.split("_")
        color = colormaps[part]
        colors.append(color)
    return colors

args = {
    'dology': True,
    'mycolors': [colormaps[part] for part in rdfs.keys()],
    "MCOnly": True,
    'addOverflow': True,
    'addUnderflow': True,
    'donormalize': False
}


print("Drawing")

DrawHistos(list(histos['nOPs'].values()), list(histos['nOPs'].keys(
)), 0, 110000, "Number of OPs", 1e-1, 1e4, "Fraction of OPs", "nOPs", **args)
DrawHistos(list(histos['OP_time_produced'].values()), list(histos['OP_time_produced'].keys(
)), 0, t_range, "Time [ns]", 1e-1, 1e7, "Fraction of OPs", "OP_time_produced", **args)
DrawHistos(list(histos['OP_time_final'].values()), list(histos['OP_time_final'].keys(
)), 0, t_range, "Time [ns]", 1e-1, 1e7, "Fraction of OPs", "OP_time_final", **args)
DrawHistos(list(histos['OP_time_delta'].values()), list(histos['OP_time_delta'].keys(
)), 0, t_range, "Time [ns]", 1e-1, 1e7, "Fraction of OPs", "OP_time_delta", **args)
DrawHistos(list(histos['OP_pos_delta_z'].values()), list(histos['OP_pos_delta_z'].keys(
)), 0, 100, "z [cm]", 1e-1, 1e7, "Fraction of OPs", "OP_pos_delta_z", **args)
DrawHistos(list(histos['OP_time_per_meter'].values()), list(histos['OP_time_per_meter'].keys(
)), 0, 12, "Time [ns/m]", 1e-1, 1e7, "Fraction of OPs", "OP_time_per_meter", **args)
DrawHistos(list(histos['OP_cosTheta_produced'].values()), list(histos['OP_cosTheta_produced'].keys(
)), 0, 1, "cos(#theta)", 1e-1, 1e7, "Fraction of OPs", "OP_cosTheta_produced", **args)
DrawHistos(list(histos['OP_cosTheta_produced_total'].values()), list(histos['OP_cosTheta_produced_total'].keys(
)), 0, 1, "cos(#theta)", 1e-1, 1e7, "Fraction of OPs", "OP_cosTheta_produced_total", **args)
DrawHistos(list(histos['OP_time_per_meter_cosTheta_produced'].values()), list(histos['OP_time_per_meter_cosTheta_produced'].keys(
)), 4.5, 6.5, "Time [ns/m]", 1e-1, 1e7, "Fraction of OPs", "OP_time_per_meter_cosTheta_produced", **args)
DrawHistos(list(histos['OP_pos_produced_r'].values()), list(histos['OP_pos_produced_r'].keys(
)), 0, 0.04, "r [cm]", 1e-1, 1e4, "Fraction of OPs", "OP_pos_produced_r", **args)

def GetRatio(histos, hname, parts=None):
    if hname not in histos:
        return None
    if hname + "_total" not in histos:
        return None
    
    if parts is None:
        parts = histos[hname].keys()
    
    h_ratios = []
    for part in parts:
        h = histos[hname][part].GetValue()
        hden = histos[hname + "_total"][part].GetValue()
    
        h_ratio = h.Clone(h.GetName() + "_ratio")
        h_ratio.Divide(hden)
        
        h_ratios.append(h_ratio)
    return h_ratios

h_ratios_cosTheta = GetRatio(histos, "OP_cosTheta_produced")
h_ratios_cosTheta_clad = GetRatio(histos, "OP_cosTheta_produced_clad")
h_ratios_cosTheta_core = GetRatio(histos, "OP_cosTheta_produced_core")
DrawHistos(h_ratios_cosTheta + h_ratios_cosTheta_clad + h_ratios_cosTheta_core, ["ele", "pion", "ele clad", "pion clad", "ele core", "ele clad"], 0, 1, "cos(#theta)", 1e-3, 10, "Fraction of OPs", "OP_cosTheta_produced_ratio", **{**args, 'mycolors': [2,3,4,5,6,7]})

h_ratios_r = GetRatio(histos, "OP_pos_produced_r")
DrawHistos(h_ratios_r, ["ele", "pion"], 0, 0.04, "r [cm]", 0.0, 0.4, "OP Trapping Rate", "OP_pos_produced_r_ratio", **{**args, 'dology': False})
DrawHistos(h_ratios_r, ["ele", "pion"], 0, 0.04, "r [cm]", 1e-4, 1e2, "OP Trapping Rate", "OP_pos_produced_r_ratio_log", **args)

output_file = ROOT.TFile("multiclad_output.root", "RECREATE")
for part in rdfs.keys():
    histos['OP_time_produced'][part].Write()
    histos['OP_time_final'][part].Write()
    histos['OP_pos_produced_r'][part].Write()
    histos['OP_cosTheta_produced'][part].Write()
    histos['OP_cosTheta_produced_total'][part].Write()
    histos['OP_time_per_meter_vs_cosTheta_produced'][part].Write()
    histos['OP_sinAlpha_vs_r_produced'][part].Write()
    histos['OP_sinAlpha_vs_r_produced_total'][part].Write()
    histos['OP_mom_produced_x_vs_y'][part].Write()
    histos['OP_mom_final_x_vs_y'][part].Write()
    histos["OP_pos_final_x_vs_y"][part].Write()
    histos["OP_pos_produced_x_vs_y"][part].Write()
    histos['OP_time_final_normalized'][part].Write()
    histos['OP_time_delta'][part].Write()
h_ratios_r[0].Write()
h_ratios_cosTheta[0].Write()
h_ratios_cosTheta_clad[0].Write()
h_ratios_cosTheta_core[0].Write()


def makeArrowPlots(hprof2d_x, hprof2d_y, min_entries=1, min_value= 1e-10, scale = 1e7):
    # assumes hprof2d_x and hprof2d_y have same binning
    arrows = []
    nbinsX = hprof2d_x.GetNbinsX()
    nbinsY = hprof2d_x.GetNbinsY()
    for i in range(1, nbinsX+1):
        for j in range(1, nbinsY+1):
            bin_ij = hprof2d_x.GetBin(i, j)
            if hprof2d_x.GetBinEffectiveEntries(bin_ij) <= min_entries:
                continue
            avg_px = hprof2d_x.GetBinContent(i, j)
            avg_py = hprof2d_y.GetBinContent(i, j)
            avg_pxy = (avg_px**2 + avg_py**2)**0.5
            if avg_pxy <= min_value:
                continue
            
            x = hprof2d_x.GetXaxis().GetBinCenter(i)
            y = hprof2d_x.GetYaxis().GetBinCenter(j)
            dx = avg_px * scale
            dy = avg_py * scale
            arrow = ROOT.TArrow(x, y, x+dx, y+dy, 0.01, "|>")
            arrow.SetLineColor(ROOT.kPink)
            arrow.SetFillColor(ROOT.kPink)
            arrows.append(arrow)
    return arrows

# 2D plots
args['dology'] = False
args['drawoptions'] = "colz"
args['dologz'] = True
args['zmax'] = 1e2
args['zmin'] = 1e-4
if doOP:
    args['zmax'] = 1e3
    args['zmin'] = 1.0
args['doth2'] = True
args['addOverflow'] = False
args['addUnderflow'] = False

scale = 1e7
if doOP:
    scale = 2e4
    
for part in rdfs.keys():
    for i, f in [(0, 13), (0, 8), (0, 6), (0, 5.8), (0, 5.5)]:
        i_str = str(i).replace(".", "_")
        f_str = str(f).replace(".", "_")
        DrawHistos([histos[f"OP_pos_produced_x_vs_y_{i_str}_and_{f_str}_ns"][part]], [], -x_range, x_range,
                   "x [cm]", -x_range, x_range, "y [cm]", f"OP_pos_produced_x_vs_y_{i_str}_and_{f_str}_ns_{part}", **args)
        

    DrawHistos([histos['OP_pos_produced_x_vs_y'][part]], [], -x_range, x_range,
               "x [cm]", -x_range, x_range, "y [cm]", f"OP_pos_produced_x_vs_y_{part}", **args)
    DrawHistos([histos['OP_pos_produced_x_vs_y_total'][part]], [], -x_range, x_range,
                "x [cm]", -x_range, x_range, "y [cm]", f"OP_pos_produced_x_vs_y_total_{part}", **args)
    hratio_x_vs_y = GetRatio(histos, "OP_pos_produced_x_vs_y", parts=[part])[0]
    DrawHistos([hratio_x_vs_y], [], -x_range, x_range, "x [cm]", -x_range, x_range, "y [cm]", f"OP_pos_produced_x_vs_y_ratio", **{**args, 'zmax': 1.0, 'zmin': 0.0})
    DrawHistos([histos['OP_pos_final_x_vs_y'][part]], [], -x_range, x_range,
               "x [cm]", -x_range, x_range, "y [cm]", f"OP_pos_final_x_vs_y_{part}", **args)
    DrawHistos([histos['OP_mom_produced_x_vs_y'][part]], [], -px_range, px_range,
               "px [GeV/c]", -px_range, px_range, "py [GeV/c]", f"OP_mom_produced_x_vs_y_{part}", **args)
    DrawHistos([histos['OP_mom_final_x_vs_y'][part]], [], -px_range, px_range,
               "px [GeV/c]", -px_range, px_range, "py [GeV/c]", f"OP_mom_final_x_vs_y_{part}", **args)
    DrawHistos([histos['OP_mom_produced_x_vs_y_total'][part]], [], -px_range, px_range,
                "px [GeV/c]", -px_range, px_range, "py [GeV/c]", f"OP_mom_produced_x_vs_y_total_{part}", **args)
    hratio_px_vs_py = GetRatio(histos, "OP_mom_produced_x_vs_y", parts=[part])[0]
    DrawHistos([hratio_px_vs_py], [], -px_range, px_range, "px [GeV/c]", -px_range, px_range, "py [GeV/c]", f"OP_mom_produced_x_vs_y_ratio", **{**args, 'zmax': 1.0, 'zmin': 0.0})
    
    DrawHistos([histos['OP_time_per_meter_vs_cosTheta_produced'][part]], [], 0, 12, "Time [ns/m]", 1, 2.5, "1.0/cos(#theta)", f"OP_time_per_meter_vs_cosTheta_produced_{part}", **args)
    
    DrawHistos([histos['OP_cosTheta_vs_r_produced'][part]], [], 0, 0.04, "r [cm]", 0, 1, "cos(#theta)", f"OP_cosTheta_vs_r_produced_{part}", **args)
    DrawHistos([histos['OP_cosTheta_vs_r_produced_total'][part]], [], 0, 0.04, "r [cm]", 0, 1, "cos(#theta)", f"OP_cosTheta_vs_r_produced_total_{part}", **args)
    
    hratio_cosTheta_vs_r = GetRatio(histos, "OP_cosTheta_vs_r_produced", parts=[part])[0]
    DrawHistos([hratio_cosTheta_vs_r], [], 0, 0.04, "r [cm]", 0, 1, "cos(#theta)", f"OP_cosTheta_vs_r_produced_ratio_{part}", **{**args, 'zmax': 1.0, 'zmin': 0.0})
    
    DrawHistos([histos['OP_sinAlpha_vs_r_produced'][part]], [], 0, 0.04, "r [cm]", 0, 1, "sin(#alpha)", f"OP_sinAlpha_vs_r_produced_{part}", **args)
    
    DrawHistos([histos['OP_sinAlpha_vs_r_produced_total'][part]], [], 0, 0.04, "r [cm]", 0, 1, "sin(#alpha)", f"OP_sinAlpha_vs_r_produced_total_{part}", **args)
    
    hratio_sinAlpha_vs_r = GetRatio(histos, "OP_sinAlpha_vs_r_produced", parts=[part])[0]
    DrawHistos([hratio_sinAlpha_vs_r], [], 0, 0.04, "r [cm]", 0, 1, "sin(#alpha)", f"OP_sinAlpha_vs_r_produced_ratio_{part}", **{**args, 'zmax': 1.0, 'zmin': 0.0})
    
    DrawHistos([histos['OP_sinAlpha_vs_r_produced_long'][part]], [], 0, 0.04, "r [cm]", 0, 1, "sin(#alpha)", f"OP_sinAlpha_vs_r_produced_long_{part}", **args)
    
    DrawHistos([histos['OP_sinAlpha_vs_r_produced_long_total'][part]], [], 0, 0.04, "r [cm]", 0, 1, "sin(#alpha)", f"OP_sinAlpha_vs_r_produced_long_total_{part}", **args)
    
    hratio_sinAlpha_vs_r_long = GetRatio(histos, "OP_sinAlpha_vs_r_produced_long", parts=[part])[0]
    DrawHistos([hratio_sinAlpha_vs_r_long], [], 0, 0.04, "r [cm]", 0, 1, "sin(#alpha)", f"OP_sinAlpha_vs_r_produced_ratio_long_{part}", **{**args, 'zmax': 1.0, 'zmin': 0.0})
    
    # profile plots
    arrows = makeArrowPlots(histos['OP_profilePx_produced_x_vs_y'][part].GetValue(), histos['OP_profilePy_produced_x_vs_y'][part].GetValue(), scale=scale)
    DrawHistos([histos['OP_pos_produced_x_vs_y'][part]], [], -x_range, x_range,"x [cm]", -x_range, x_range, "y [cm]", f"OP_produced_x_vs_y_{part}_withPXY", **args, extraToDraw = arrows)
    
    arrows = makeArrowPlots(histos['OP_profilePx_final_x_vs_y'][part].GetValue(), histos['OP_profilePy_final_x_vs_y'][part].GetValue(), scale=scale)
    DrawHistos([histos['OP_pos_final_x_vs_y'][part]], [], -x_range, x_range,"x [cm]", -x_range, x_range, "y [cm]", f"OP_final_x_vs_y_{part}_withPXY", **args, extraToDraw = arrows)
    
    arrows = makeArrowPlots(histos['OP_profilePx_produced_x_vs_y_total'][part].GetValue(), histos['OP_profilePy_produced_x_vs_y_total'][part].GetValue(), scale=scale)
    DrawHistos([histos['OP_pos_produced_x_vs_y_total'][part]], [], -x_range, x_range,"x [cm]", -x_range, x_range, "y [cm]", f"OP_produced_x_vs_y_total_{part}_withPXY", **args, extraToDraw = arrows)
    
    # event displays
    for i in evtlist:
        DrawHistos([histos[f"event_{i}_OP_pos_produced_x_vs_y"][part]], [], -x_range, x_range,
                   "x [cm]", -x_range, x_range, "y [cm]", f"event_{i}_OP_pos_produced_x_vs_y_{part}", **args)

        DrawHistos([histos[f"event_{i}_OP_pos_final_x_vs_y"][part]], [], -x_range, x_range,
                   "x [cm]", -x_range, x_range, "y [cm]", f"event_{i}_OP_pos_final_x_vs_y_{part}", **args)
        DrawHistos([histos[f"event_{i}_OP_mom_produced_x_vs_y"][part]], [], -px_range, px_range,
                   "px [GeV/c]", -px_range, px_range, "py [GeV/c]", f"event_{i}_OP_mom_produced_x_vs_y_{part}", **args)
        DrawHistos([histos[f"event_{i}_OP_mom_final_x_vs_y"][part]], [], -px_range, px_range,
                   "px [GeV/c]", -px_range, px_range, "py [GeV/c]", f"event_{i}_OP_final_x_vs_y_{part}", **args)
hratio_x_vs_y.Write()
output_file.Close()
print("Done")
