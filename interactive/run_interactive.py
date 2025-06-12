import subprocess
import sys
import os

# Define the paths to the scripts
make_pulse_script = "makeInteractPulse.py"
event_interact_script = "eventinteract2.py"

# Step 1: Run makeInteractPulse.py
print("Running makeInteractPulse.py...")
try:
    result = subprocess.run(["python", make_pulse_script], check=True)
    print("makeInteractPulse.py completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error: makeInteractPulse.py failed with exit code {e.returncode}.")
    sys.exit(1)

# Step 2: Check if root_file_name.txt exists and contains a valid file path
try:
    with open("root_file_name.txt", "r") as f:
        root_file_name = f.read().strip()
        if not os.path.exists(root_file_name):
            print(f"Error: The ROOT file specified in root_file_name.txt does not exist: {root_file_name}")
            sys.exit(1)
except FileNotFoundError:
    print("Error: root_file_name.txt not found. Please ensure it exists and contains the correct ROOT file name.")
    sys.exit(1)

# Step 3: Run eventinteract2.py
print("Running eventinteract2.py...")
try:
    result = subprocess.run(["python", event_interact_script], check=True)
    print("eventinteract2.py completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error: eventinteract2.py failed with exit code {e.returncode}.")
    sys.exit(1)
