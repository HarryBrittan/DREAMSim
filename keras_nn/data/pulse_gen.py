import numpy as np
import h5py
import time
import ROOT

# Parameters
N_EVENTS = 50000
N_BINS = 200  # Number of time bins per event
NOISE_STD = 0.05  # Standard deviation of noise
MAX_HITS_PER_EVENT = 4  # Max number of true hits per event

# Load pulse template from ROOT file
pulse_root_file = "../../plotter/data/different_pulses.root"
pulse_hist_name = "h_landau"  # Change if needed

pfile = ROOT.TFile(pulse_root_file)
h_pulse = pfile.Get(pulse_hist_name)
pulse_template = np.array([h_pulse.GetBinContent(i+1) for i in range(h_pulse.GetNbinsX())], dtype=np.float32)
pulse_len = len(pulse_template)
pulse_peak = np.argmax(pulse_template)

# Start timing
start_time = time.time()

# Generate data
truth_hits = []
convoluted_pulses = []

for _ in range(N_EVENTS):
    n_hits = np.random.randint(1, MAX_HITS_PER_EVENT + 1)
    truth = np.zeros(N_BINS)
    convoluted = np.zeros(N_BINS)
    for _ in range(n_hits):
        hit_bin = np.random.randint(20, N_BINS - 20)
        n_particles = np.random.randint(1, 4)
        truth[hit_bin] += n_particles
        for _ in range(n_particles):
            # Overlay the ROOT pulse template at the hit_bin
            start = hit_bin - pulse_peak
            for k in range(pulse_len):
                bin_idx = start + k
                if 0 <= bin_idx < N_BINS:
                    convoluted[bin_idx] += pulse_template[k]
    # Add noise to the convoluted signal
    convoluted += np.random.normal(0, NOISE_STD, size=convoluted.shape)
    truth_hits.append(truth)
    convoluted_pulses.append(convoluted)

truth_hits = np.array(truth_hits, dtype=np.float32)
convoluted_pulses = np.array(convoluted_pulses, dtype=np.float32)

# Save to HDF5
with h5py.File("pulse_data.h5", "w") as f:
    f.create_dataset("truth", data=truth_hits)
    f.create_dataset("convoluted", data=convoluted_pulses)

print(f"Saved {N_EVENTS} events (with up to {MAX_HITS_PER_EVENT} hits per event, multiple particles per bin possible) to pulse_data.h5")

# Print elapsed time
elapsed = time.time() - start_time
print(f"Generation finished in {elapsed:.2f} seconds.")
