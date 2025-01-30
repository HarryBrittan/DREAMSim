import sys
from collections import OrderedDict
import ROOT
sys.path.append("./CMSPLOTS")

from myFunction import DrawHistos

import os

ROOT.ROOT.EnableImplicitMT(4)

partfile = "inputs/particles.txt"

chains = OrderedDict()
chains['par'] = ROOT.TChain("tree")

loops = [('par', partfile)]

rdfs = OrderedDict()

# Creating a dataframe for each path in the text file and adding each line in that file to the dataframe
for part, filename in loops:
    with open(filename) as f:
        #print(f"Reading {filename}")
        elefiles = f.readlines()
        for i, line in enumerate(elefiles):
            path = line.strip()
            if path.startswith("#"):
                continue
            #print(f"{part} " + path)
            if not os.path.isfile(path):
                print(f"File does not exist: {path}")
                continue
            rdfs[f"{part}_{i}"] = ROOT.RDataFrame("tree", path)

#Getting number of events from each dataframe, lines inside each txt files will accumulate into a dataframe per file
nEvts = OrderedDict()
for key, rdf in rdfs.items():
    nEvts[key] = rdf.Count().GetValue()
    print(f"Key: {key}, Number of entries: {rdf.Count().GetValue()}")

#Defining new values in each root dataframe  
rdfs_new = OrderedDict()
for part in rdfs.keys():
    rdfs_new[part] = rdfs[part].Define( "eTotaltruth", "eLeaktruth + eCalotruth + eWorldtruth + eInvisible") \
        .Define("eTotalGeant", "eLeaktruth + eCalotruth + eWorldtruth") \
        .Define("truthhit_r", f"sqrt((truthhit_x-2.5)*(truthhit_x-2.5) + (truthhit_y+2.5)*(truthhit_y+2.5))")\
        .Define("eweight", f"truthhit_edep/ {nEvts[part]}")
            
rdfs = rdfs_new

#rdfs['par'] = rdfs['par'].Define("eweight", f"truthhit_edep/ {nEvts['par']}")
histos = OrderedDict()
figures = ['eLeaktruth', 'eCalotruth', 'eTotaltruth', 'eTotalGeant',
           'eRodtruth', 'eCentruth', 'eScintruth',
           'truthhit_x', 'truthhit_y', 'truthhit_z', 'truthhit_r',
           'time', 'time_zoomed',
           'truthhit_x_vs_truthhit_y', 'truthhit_x_vs_truthhit_z', 'truthhit_r_vs_truthhit_z',
           'time_vs_truthhit_z', 'time_vs_truthhit_r']

#Adding figures through append here to keep track can remove and add directly to list after drawing graphs is complete
new_figs= ['eCentruth_vs_eRodtruth', 'eScintruth_vs_eRodtruth']
figures.extend(new_figs)
evtlist = [1, 10, 20, 30, 40]
for i in evtlist:
    figures.append(f"event_{i}_truthhit_x_vs_truthhit_y")
    figures.append(f"event_{i}_truthhit_x_vs_truthhit_z")
    figures.append(f"event_{i}_truthhit_r_vs_truthhit_z")
    figures.append(f"event_{i}_time_vs_truthhit_z")
    figures.append(f"event_{i}_time_vs_truthhit_r")

for fig in figures:
    histos[fig] = OrderedDict()

for part, rdf in rdfs.items():
    suffix = "_" + part

    histos['eLeaktruth'][part] = rdf.Histo1D(
        ("eLeaktruth" + suffix, "eLeaktruth", 50, 0, 11.0), "eLeaktruth")
    histos['eCalotruth'][part] = rdf.Histo1D(
        ("eCalotruth" + suffix, "eCalotruth", 50, 0., 102.0), "eCalotruth")
    histos['eTotaltruth'][part] = rdf.Histo1D(
        ("eTotaltruth" + suffix, "eTotaltruth", 50, 90., 110.0), "eTotaltruth")
    histos['eTotalGeant'][part] = rdf.Histo1D(
        ("eTotalGeant" + suffix, "eTotalGeant", 50, 80., 110.0), "eTotalGeant")
    histos['eRodtruth'][part] = rdf.Histo1D(
        ("eRodtruth" + suffix, "eRodtruth", 50, 50, 102.0), "eRodtruth")
    histos['eCentruth'][part] = rdf.Histo1D(
        ("eCentruth" + suffix, "eCentruth", 50, 0.0, 1.0), "eCentruth")
    histos['eScintruth'][part] = rdf.Histo1D(
        ("eScintruth" + suffix, "eScintruth", 50, 0.0, 1.0), "eScintruth")

    # energy weighted
    histos['truthhit_x'][part] = rdf.Histo1D(
        ("truthhit_x" + suffix, "truthhit_x", 50, -20, 20), "truthhit_x", "eweight")
    histos['truthhit_y'][part] = rdf.Histo1D(
        ("truthhit_y" + suffix, "truthhit_y", 50, -20, 20), "truthhit_y", "eweight")
    histos['truthhit_z'][part] = rdf.Histo1D(
        ("truthhit_z" + suffix, "truthhit_z", 100, -100, 100), "truthhit_z", "eweight")
    histos['truthhit_r'][part] = rdf.Histo1D(
        ("truthhit_r" + suffix, "truthhit_r", 50, 0, 30), "truthhit_r", "eweight")

    histos['time'][part] = rdf.Histo1D(
        ("time" + suffix, "time", 50, 0, 50), "truthhit_globaltime", "eweight")
    histos['time_zoomed'][part] = rdf.Histo1D(
        ("time_zoomed" + suffix, "time_zoomed", 50, 0, 10), "truthhit_globaltime", "eweight")

    histos['truthhit_x_vs_truthhit_y'][part] = rdf.Histo2D(
        ("truthhit_x_vs_truthhit_y" + suffix, "truthhit_x_vs_truthhit_y", 50, -20, 20, 50, -20, 20), "truthhit_x", "truthhit_y", "eweight")
    histos['truthhit_x_vs_truthhit_z'][part] = rdf.Histo2D(
        ("truthhit_x_vs_truthhit_z" + suffix, "truthhit_x_vs_truthhit_z", 50, -20, 20, 100, -100, 100), "truthhit_x", "truthhit_z", "eweight")
    histos['truthhit_r_vs_truthhit_z'][part] = rdf.Histo2D(
        ("truthhit_r_vs_truthhit_z" + suffix, "truthhit_r_vs_truthhit_z", 50, 0, 30, 100, -100, 100), "truthhit_r", "truthhit_z", "eweight")

    histos['time_vs_truthhit_z'][part] = rdf.Histo2D(
        ("time_vs_truthhit_z" + suffix, "time_vs_truthhit_z", 50, 0, 20, 100, -100, 100), "truthhit_globaltime", "truthhit_z", "eweight")
    histos['time_vs_truthhit_r'][part] = rdf.Histo2D(
        ("time_vs_truthhit_r" + suffix, "time_vs_truthhit_r", 50, 0, 20, 50, 0, 30), "truthhit_globaltime", "truthhit_r", "eweight")

    histos['eCentruth_vs_eRodtruth'][part] = rdf.Histo2D(
        ("eCentruth_vs_eRodtruth" + suffix, "eCentruth_vs_eRodtruth", 50, -20, 20, 50, -20, 20), "eCentruth", "eRodtruth", "eweight")
    histos['eScintruth_vs_eRodtruth'][part] = rdf.Histo2D(
        ("eScintruth_vs_eRodtruth" + suffix, "eScintruth_vs_eRodtruth", 50, -20, 20, 50, -20, 20), "eScintruth", "eRodtruth", "eweight")
    # some event displays
    for i in evtlist:
        rdf_event = rdf.Filter(f"rdfentry_ == {i}")
        histos[f"event_{i}_truthhit_x_vs_truthhit_y"][part] = rdf_event.Histo2D(
            (f"event_{i}_truthhit_x_vs_truthhit_y" + suffix, f"event_{i}_truthhit_x_vs_truthhit_y", 50, -20, 20, 50, -20, 20), "truthhit_x", "truthhit_y", "truthhit_edep")
        histos[f"event_{i}_truthhit_x_vs_truthhit_z"][part] = rdf_event.Histo2D(
            (f"event_{i}_truthhit_x_vs_truthhit_z" + suffix, f"event_{i}_truthhit_x_vs_truthhit_z", 50, -20, 20, 100, -100, 100), "truthhit_x", "truthhit_z", "truthhit_edep")
        histos[f"event_{i}_truthhit_r_vs_truthhit_z"][part] = rdf_event.Histo2D(
            (f"event_{i}_truthhit_r_vs_truthhit_z" + suffix, f"event_{i}_truthhit_r_vs_truthhit_z", 50, 0, 30, 100, -100, 100), "truthhit_r", "truthhit_z", "truthhit_edep")

        histos[f"event_{i}_time_vs_truthhit_z"][part] = rdf_event.Histo2D(
            (f"event_{i}_time_vs_truthhit_z" + suffix, f"event_{i}_time_vs_truthhit_z", 50, 0, 20, 100, -100, 100), "truthhit_globaltime", "truthhit_z", "truthhit_edep")
        histos[f"event_{i}_time_vs_truthhit_r"][part] = rdf_event.Histo2D(
            (f"event_{i}_time_vs_truthhit_r" + suffix, f"event_{i}_time_vs_truthhit_r", 50, 0, 20, 50, 0, 30), "truthhit_globaltime", "truthhit_r", "truthhit_edep")

colormaps = {
    'par_0': 2,
    'par_1': 3,
    'par_2': 4,
    'par_3': 9
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
    'donormalize': True,
    'mycolors': [colormaps[part] for part in rdfs.keys()],
    "MCOnly": True,
    'addOverflow': True,
    'addUnderflow': True
}

print("Drawing")

DrawHistos(list(histos['eLeaktruth'].values()), list(histos['eLeaktruth'].keys(
)), 0, 11, "Leakage Energy [GeV]", 1e-3, 1e2, "Fraction of events", "eLeaktruth", **args)
DrawHistos(list(histos['eCalotruth'].values()), list(histos['eCalotruth'].keys(
)), 0., 102, "Calo Energy [GeV]", 1e-3, 1e2, "Fraction of events", "eCalotruth", **args)
DrawHistos(list(histos['eTotaltruth'].values()), list(histos['eTotaltruth'].keys(
)), 80., 110, "Total Energy [GeV]", 1e-3, 1e2, "Fraction of events", "eTotaltruth", **args)
DrawHistos(list(histos['eTotalGeant'].values()), list(histos['eTotalGeant'].keys(
)), 80., 110, "Total 'Visible' Energy [GeV]", 1e-3, 1e2, "Fraction of events", "eTotalGeant", **args)
DrawHistos(list(histos['eRodtruth'].values()), list(histos['eRodtruth'].keys(
)), 50, 102, "Rod Energy [GeV]", 1e-3, 1e2, "Fraction of events", "eRodtruth", **args)
DrawHistos(list(histos['eCentruth'].values()), list(histos['eCentruth'].keys(
)), 0, 1, "CFiber Energy [GeV]", 1e-3, 1e2, "Fraction of events", "eCentruth", **args)
DrawHistos(list(histos['eScintruth'].values()), list(histos['eScintruth'].keys(
)), 0, 1, "SFiber Energy [GeV]", 1e-3, 1e2, "Fraction of events", "eScintruth", **args)

args['donormalize'] = False
DrawHistos(list(histos['truthhit_x'].values()), list(histos['truthhit_x'].keys(
)), -20, 20, "x [cm]", 1e-3, 1e2, "Deposited Energy [GeV]", "truthhit_x", **args)
DrawHistos(list(histos['truthhit_y'].values()), list(histos['truthhit_y'].keys(
)), -20, 20, "y [cm]", 1e-3, 1e2, "Deposited Energy [GeV]", "truthhit_y", **args)
DrawHistos(list(histos['truthhit_z'].values()), list(histos['truthhit_z'].keys(
)), -100, 100, "z [cm]", 1e-3, 1e2, "Deposited Energy [GeV]", "truthhit_z", **args)
DrawHistos(list(histos['truthhit_r'].values()), list(histos['truthhit_r'].keys(
)), 0, 30, "r [cm]", 1e-3, 1e2, "Deposited Energy [GeV]", "truthhit_r", **args)

DrawHistos(list(histos['time'].values()), list(histos['time'].keys(
)), 0, 100, "Time [ns]", 1e-3, 1e2, "Deposited Energy [GeV]", "time", **args)
DrawHistos(list(histos['time_zoomed'].values()), list(histos['time_zoomed'].keys(
)), 0, 10, "Time [ns]", 1e-3, 1e2, "Deposited Energy [GeV]", "time_zoomed", **args)

# 2D plots
args['dology'] = False
args['drawoptions'] = "colz"
args['dologz'] = True
args['zmax'] = 1e0
args['zmin'] = 1e-6
args['doth2'] = True
args['addOverflow'] = False
args['addUnderflow'] = False
for part in rdfs.keys():
    DrawHistos([histos['truthhit_x_vs_truthhit_y'][part]], [], -20, 20,
               "x [cm]", -20, 20, "y [cm]", f"truthhit_x_vs_truthhit_y_{part}", **args)
    DrawHistos([histos['truthhit_x_vs_truthhit_z'][part]], [], -20, 20,
               "x [cm]", -100, 100, "z [cm]", f"truthhit_x_vs_truthhit_z_{part}", **args)
    DrawHistos([histos['truthhit_r_vs_truthhit_z'][part]], [], 0, 30,
               "r [cm]", -100, 100, "z [cm]", f"truthhit_r_vs_truthhit_z_{part}", **args)

    DrawHistos([histos['time_vs_truthhit_z'][part]], [], 0, 20,
               "Time [ns]", -100, 100, "z [cm]", f"time_vs_truthhit_z_{part}", **args)
    DrawHistos([histos['time_vs_truthhit_r'][part]], [], 0, 20,
               "Time [ns]", 0, 30, "r [cm]", f"time_vs_truthhit_r_{part}", **args)
    
    DrawHistos([histos['eCentruth_vs_eRodtruth'][part]], [], 1, 5,
                "Cfiber Energy [GeV]", 0, 5, "Rod Energy [GeV]", f"eCentruth_vs_eRodtruth_{part}", **args)
    DrawHistos([histos['eScintruth_vs_eRodtruth'][part]], [], 1, 5,
                "Sfiber Energy [GeV]", 0, 5, "Rod Energy [GeV]", f"eScintruth_vs_eRodtruth_{part}", **args)
    # event displays
    for i in evtlist:
        DrawHistos([histos[f"event_{i}_truthhit_x_vs_truthhit_y"][part]], [
        ], -20, 20, "x [cm]", -20, 20, "y [cm]", f"event_{i}_truthhit_x_vs_truthhit_y_{part}", **args)
        DrawHistos([histos[f"event_{i}_truthhit_x_vs_truthhit_z"][part]], [
        ], -20, 20, "x [cm]", -100, 100, "z [cm]", f"event_{i}_truthhit_x_vs_truthhit_z_{part}", **args)
        DrawHistos([histos[f"event_{i}_truthhit_r_vs_truthhit_z"][part]], [
        ], 0, 30, "r [cm]", -100, 100, "z [cm]", f"event_{i}_truthhit_r_vs_truthhit_z_{part}", **args)

        DrawHistos([histos[f"event_{i}_time_vs_truthhit_z"][part]], [
        ], 0, 20, "Time [ns]", -100, 100, "z [cm]", f"event_{i}_time_vs_truthhit_z_{part}", **args)
        DrawHistos([histos[f"event_{i}_time_vs_truthhit_r"][part]], [
        ], 0, 20, "Time [ns]", 0, 30, "r [cm]", f"event_{i}_time_vs_truthhit_r_{part}", **args)


print("Done")