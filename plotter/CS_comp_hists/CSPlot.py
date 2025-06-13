import os
import ROOT
import numpy as np

# Define the destination directory for PNG files
output_dir = "./CShistograms"

# Ensure the destination directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# List of ROOT files to process
input_files = [
    "/lustre/work/hbrittan/dreamsimoutputs/2025_06_10/mc_dreamsim_e+_0_100_100.1_run1_0_Test_1000evt_e+_100_100.1.root",
    "/lustre/work/hbrittan/dreamsimoutputs/2025_06_10/mc_dreamsim_e+_1_100_100.1_run1_1_Test_1000evt_e+_100_100.1.root",
    "/lustre/work/hbrittan/dreamsimoutputs/2025_06_10/mc_dreamsim_e+_2_100_100.1_run1_2_Test_1000evt_e+_100_100.1.root",
    "/lustre/work/hbrittan/dreamsimoutputs/2025_06_10/mc_dreamsim_e+_3_100_100.1_run1_3_Test_1000evt_e+_100_100.1.root",
    "/lustre/work/hbrittan/dreamsimoutputs/2025_06_10/mc_dreamsim_e+_4_100_100.1_run1_4_Test_1000evt_e+_100_100.1.root",
    "/lustre/work/hbrittan/dreamsimoutputs/2025_06_10/mc_dreamsim_e+_5_100_100.1_run1_5_Test_1000evt_e+_100_100.1.root",
    "/lustre/work/hbrittan/dreamsimoutputs/2025_06_10/mc_dreamsim_e+_6_100_100.1_run1_6_Test_1000evt_e+_100_100.1.root",
    "/lustre/work/hbrittan/dreamsimoutputs/2025_06_10/mc_dreamsim_e+_7_100_100.1_run1_7_Test_1000evt_e+_100_100.1.root",
    "/lustre/work/hbrittan/dreamsimoutputs/2025_06_10/mc_dreamsim_e+_8_100_100.1_run1_8_Test_1000evt_e+_100_100.1.root",
    "/lustre/work/hbrittan/dreamsimoutputs/2025_06_10/mc_dreamsim_e+_9_100_100.1_run1_9_Test_1000evt_e+_100_100.1.root",

    #"../jobs/mc_DREAM_mu+_0_run2_0_Test_100evt_mu+_100_100.root",
    # "../jobs/mc_DREAM_mu+_1_run2_1_Test_100evt_mu+_100_100.root",
    # "../jobs/mc_DREAM_mu+_2_run2_2_Test_100evt_mu+_100_100.root",
    # "../jobs/mc_DREAM_mu+_3_run2_3_Test_100evt_mu+_100_100.root",
    # "../jobs/mc_DREAM_mu+_4_run2_4_Test_100evt_mu+_100_100.root",
    # "../jobs/mc_DREAM_mu+_5_run2_5_Test_100evt_mu+_100_100.root",
    # "../jobs/mc_DREAM_mu+_6_run2_6_Test_100evt_mu+_100_100.root",
    # "../jobs/mc_DREAM_mu+_7_run2_7_Test_100evt_mu+_100_100.root",
    # "../jobs/mc_DREAM_mu+_8_run2_8_Test_100evt_mu+_100_100.root",
    # "../jobs/mc_DREAM_mu+_9_run2_9_Test_100evt_mu+_100_100.root",
    # Add more files as needed

    # Add paths here
]

# Create a TChain to combine all files
chain = ROOT.TChain("tree")  # Replace "tree" with the name of your TTree
for file in input_files:
    chain.Add(file)

# Prepare arrays to store energy depositions
hadronic_edep = []
electromagnetic_edep = []
muon_edep = []  # Initialize muon energy if needed
total_edep = []
ios_edep = []  # Ionization energy deposition, if needed

# Loop over events in the TChain
print("Processing events to categorize energy deposition...")
for event in chain:

    # Initialize energy deposition counters
    hadronic_energy = 0
    electromagnetic_energy = 0
    muon_energy = 0  # Initialize muon energy if needed
    tot_edep = 0
    tot_ion_edep = 0

    # Loop through the truthhit_pid and corresponding truthhit_edep values
    for particle_id, edep, edep_non_ion, calo in zip(event.truthhit_pid, event.truthhit_edep, event.truthhit_edepNonIon, event.truthhit_calotype):
        if not calo == 2:
            continue
        # Categorize the energy deposition based on the particle ID
        if abs(particle_id) in [11, 22]:  # Electromagnetic particles (e.g., electrons, photons)
            electromagnetic_energy += edep
        elif abs(particle_id) == 13:  # Muons
            muon_energy += edep
        else:  # Hadronic particles (e.g., protons, neutrons, etc.)
            hadronic_energy += edep
        # Check if the energy deposition is not non-ionizing
        if edep_non_ion == 0:
            tot_ion_edep += edep

    # Store the energy deposition values for this event
    hadronic_edep.append(hadronic_energy)
    electromagnetic_edep.append(electromagnetic_energy)
    muon_edep.append(muon_energy)
    total_edep.append(sum(event.truthhit_edep))  # Sum all energy depositions for the event
    ios_edep.append(tot_ion_edep)


# Convert lists to numpy arrays for easier manipulation
hadronic_edep = np.array(hadronic_edep)
electromagnetic_edep = np.array(electromagnetic_edep)
muon_edep = np.array(muon_edep)  # Assuming muon_edep contains all muon energy depositions
total_edep = np.array(total_edep)  # Assuming truthhit_edep contains all energy depositions
total_ion_edep = np.array(ios_edep)  # Assuming truthhit_edepNonIon contains ionization energy depositions

input_energy = 100
total_edep_per_Energy = np.array(total_edep/input_energy)
total_ion_edep_per_Energy = np.array(total_ion_edep/input_energy)
# Check if arrays are empty before creating histograms
if len(hadronic_edep) > 0:
    hadronic_hist = ROOT.TH1F(
        "hadronic_hist",
        "Hadronic Energy Deposition;Hadronic Energy (MeV);Counts",
        50, 0, max(hadronic_edep)  # Adjust bins and range as needed
    )
    for h_edep in hadronic_edep:
        hadronic_hist.Fill(h_edep)
else:
    print("Warning: hadronic_edep is empty. Skipping Hadronic Energy Deposition histogram.")

if len(electromagnetic_edep) > 0:
    electromagnetic_hist = ROOT.TH1F(
        "electromagnetic_hist",
        "Electromagnetic Energy Deposition;Electromagnetic Energy (MeV);Counts",
        50, 0, max(electromagnetic_edep)  # Adjust bins and range as needed
    )
    for em_edep in electromagnetic_edep:
        electromagnetic_hist.Fill(em_edep)
else:
    print("Warning: electromagnetic_edep is empty. Skipping Electromagnetic Energy Deposition histogram.")

if len(muon_edep) > 0:
    muon_hist = ROOT.TH1F(
        "muon_hist",
        "Muon Energy Deposition;Muon Energy (MeV);Counts",
        50, 0, max(muon_edep)  # Adjust bins and range as needed
    )
    for muon_energy in muon_edep:
        muon_hist.Fill(muon_energy)
else:
    print("Warning: muon_edep is empty. Skipping Muon Energy Deposition histogram.")

if len(total_edep) > 0:
    total_hist = ROOT.TH1F(
        "total_hist",
        "Total Energy Deposition;Total Energy (MeV);Counts",
        50, 0, max(total_edep)  # Adjust bins and range as needed
    )
    for edep in total_edep:
        total_hist.Fill(edep)
else:
    print("Warning: total_edep is empty. Skipping Total Energy Deposition histogram.")

# Create a 1D histogram for ionization energy deposition if needed
total_ion_hist = ROOT.TH1F(
    "total_ion_hist",
    "Total Ionization Energy Deposition;Ionization Energy (MeV);Counts",
    50, 0, max(total_ion_edep)  # Adjust bins and range as needed
)

# Fill the 1D histograms
for ion_edep in total_ion_edep:
    total_ion_hist.Fill(ion_edep)



# Create a 2D histogram for energy depositions
histogram_2d = ROOT.TH2F(
    "edep_histogram",
    "Electromagnetic Edep/BeamE vs. Total Edep/BeamE;Tot_Edep/E;Electromagetic Edep/E",
    50, 0, max(total_edep_per_Energy),  # Adjust bins and range as needed
    50, 0, max(electromagnetic_edep/input_energy)  # Adjust bins and range as needed
)

total_hist_2d = ROOT.TH2F(
    "total_edep_histogram",
    "Electromagnetic Edep/BeamE vs. Ion_Edep/BeamE;Ion_Edep/E; Electromagnetic Edep/E",
    50, 0, max(total_ion_edep_per_Energy),  # Adjust bins and range as needed
    50, 0, max(electromagnetic_edep/input_energy)  # Adjust bins and range as needed
)

# Fill the 2D histogram
for tot_edep, em_edep in zip(total_edep_per_Energy, electromagnetic_edep/input_energy):
    histogram_2d.Fill(tot_edep, em_edep)

# Fill the total energy deposition 2D histogram
for tot_edep, em_edep in zip(total_ion_edep_per_Energy, electromagnetic_edep/input_energy):
    total_hist_2d.Fill(tot_edep, em_edep)


# Save the histograms to a ROOT file
output_file = ROOT.TFile("CSroottoday/CS_comparison_pidtruth_10000evt_e+_100GeV.root", "RECREATE")
histogram_2d.Write()
hadronic_hist.Write()
electromagnetic_hist.Write()
muon_hist.Write()
total_hist_2d.Write()
output_file.Close()

print("Histograms saved successfully.")

# Create a canvas and divide it into three pads
canvas = ROOT.TCanvas("canvas", "Energy Deposition", 1200, 800)
canvas.Divide(2, 2)  # Divide the canvas into a 2x2 grid

# Plot the 2D histogram in the first pad
canvas.cd(1)
histogram_2d.SetStats(0)  # Disable stats box for better visibility
histogram_2d.Draw("COLZ")

# Plot the hadronic histogram in the second pad
canvas.cd(2)
ROOT.gPad.SetLogy()  # Set logarithmic scale for better visibility
hadronic_hist.Draw("HIST")

# Plot the electromagnetic histogram in the third pad
canvas.cd(3)
ROOT.gPad.SetLogy()  # Set logarithmic scale for better visibility
electromagnetic_hist.Draw("HIST")

#Plot the total energy 2D histogram in the fourth pad
canvas.cd(4)
total_hist_2d.SetStats(0)  # Disable stats box for better visibility
total_hist_2d.Draw("COLZ")



# Save the combined plot as a single PNG file in the output directory
canvas.SaveAs(f"{output_dir}/CS_comparison_pidtruth_e+_100GeV.png")

print(f"Combined plot saved successfully in {output_dir}.")
