import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load the generated pulses
with h5py.File("mu20000evt_0.5shift_sim_pulse_data.h5", "r") as f:
    truth = np.array(f["truth"])
    convoluted = np.array(f["convoluted"])

# Plot a few random examples and save to PDF
n_show = 100  # Number of pulses to show
indices = np.random.choice(len(truth), n_show, replace=False)

with PdfPages("pulse_examples.pdf") as pdf:
    for i, idx in enumerate(indices):
        plt.figure()
        plt.plot(truth[idx], label="Truth Pulse")
        plt.plot(convoluted[idx], label="Convoluted Pulse")
        plt.xlabel("Time Bin")
        plt.ylabel("Amplitude")
        plt.title(f"Pulse Example {idx}")
        plt.legend()
        pdf.savefig()
        plt.close()

print("Saved pulse examples to pulse_examples.pdf")
