import ROOT
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import os
import sys
import subprocess
from PIL import Image
import matplotlib.colors as mcolors


# Read the ROOT file name from the shared text file
try:
    with open("root_file_name.txt", "r") as f:
        degreefile = f.read().strip()
except FileNotFoundError:
    print("Error: root_file_name.txt not found. Please run makeInteractPulse.py first.")
    sys.exit()

# Open the ROOT file
file = ROOT.TFile.Open(degreefile)
tree = file.Get("tree")

# Ask the user for the event number to filter
event_number = int(input("Enter the event number to filter (0-based index): "))

# Check if the event number is valid
if event_number < 0 or event_number >= tree.GetEntries():
    print(f"Invalid event number. Please enter a number between 0 and {tree.GetEntries() - 1}.")
    exit()

# Create a 2D histogram
h2 = ROOT.TH2F("h2", f"2D Histogram for Event {event_number};OP_pos_produced_x;OP_pos_produced_y",
               33, -20, 20, 25, -20, 20)

# Loop through the tree and process only the specified event
for i, event in enumerate(tree):
    if i != event_number:
        continue  # Skip all events except the specified one

    # Loop over the elements in the vectors for the selected event
    for x, y, is_cherenkov, z in zip(event.OP_pos_final_x, event.OP_pos_final_y, event.OP_isCoreC, event.OP_pos_final_z):
        if not is_cherenkov or z <= 49.0:
            continue  # Skip invalid particles
        h2.Fill(x, y)

# Convert ROOT histogram to numpy arrays
counts = np.array([[h2.GetBinContent(i, j) for j in range(1, h2.GetNbinsY() + 1)] for i in range(1, h2.GetNbinsX() + 1)])
xedges = np.linspace(h2.GetXaxis().GetXmin(), h2.GetXaxis().GetXmax(), h2.GetNbinsX() + 1)
yedges = np.linspace(h2.GetYaxis().GetXmin(), h2.GetYaxis().GetXmax(), h2.GetNbinsY() + 1)

# Create a mapping of bins to image file paths
image_directory = "images"  # Directory where images are stored
bin_to_image = {}
for i in range(1, h2.GetNbinsX() + 1):
    for j in range(1, h2.GetNbinsY() + 1):
        bin_to_image[(i, j)] = os.path.join(image_directory, f"bin_x{i}_y{j}.png")

# Plot the histogram with a logarithmic color scale
fig, ax = plt.subplots()
im = ax.imshow(
    counts.T,
    origin='lower',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    aspect='auto',
    norm=mcolors.LogNorm(vmin=1, vmax=counts.max())  # Apply logarithmic normalization
)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title(f"2D Histogram for Event {event_number}")
plt.colorbar(im, ax=ax)  # Add a colorbar to show the scale

# Add click interactivity (disable hover behavior)
cursor = mplcursors.cursor(im, hover=False)  # Disable hover behavior

@cursor.connect("add")
def onclick(sel):
    # Get the clicked coordinates (xpos, ypos) in data space
    xpos, ypos = sel.target
    print("xpos:", xpos, "ypos:", ypos)

    # Calculate bin indices based on xpos and ypos
    xbin = np.digitize([xpos], xedges)[0] - 1  # Convert xpos to xbin index
    ybin = np.digitize([ypos], yedges)[0] - 1  # Convert ypos to ybin index

    print(f"Bin indices: xbin={xbin}, ybin={ybin}")

    # Ensure the bin indices are within range
    if xbin < 0 or xbin >= len(xedges) - 1 or ybin < 0 or ybin >= len(yedges) - 1:
        print("Clicked outside valid bin range.")
        return

    # Call makePulsePlots.py to generate the plot for the selected bin
    subprocess.run(["python", "makeInteractPulsePlots.py", str(event_number), str(xbin), str(ybin)])

    filename = f"plots/Reco_Event_{event_number}_Bin_{xbin}_{ybin}.pdf"
    # Check if file exists and show it
    if os.path.exists(filename):
        print(f"Opening file : {filename}")
        if sys.platform =="darwin":
            subprocess.run(["open", filename])
        else:
            subprocess.run(["xdg-open", filename])
    else:
        print(f"Image not found for bin ({xbin}, {ybin})")
plt.show()
