# main.py
import subprocess

# Step 1: Preprocess the data
subprocess.run(["python3", "scripts/preprocess.py"])

# Step 2: Train the model
subprocess.run(["python3", "scripts/train.py"])

# To predict on a new image, you can run the predict script separately.
# Example: subprocess.run(["python3", "scripts/predict.py", "path_to_new_image.jpg"])
