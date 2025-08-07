import pandas as pd
import numpy as np

# Load your collected hand data
df = pd.read_csv("hand_data.csv", header=None)

# Separate features and labels
features = df.iloc[:, :-1].values  # All landmark coordinates
labels = df.iloc[:, -1].values     # Finger count

augmented_data = []

# How many new rows per original?
N_AUG = 5

for i in range(len(features)):
    base = features[i]
    label = labels[i]
    
    for _ in range(N_AUG):
        # Add slight noise to each coordinate (mean=0, std=0.01)
        noise = np.random.normal(loc=0, scale=0.01, size=base.shape)
        augmented = base + noise
        augmented_data.append(np.append(augmented, label))

# Combine original + augmented
augmented_df = pd.DataFrame(np.vstack([df.values, augmented_data]))

# Save to CSV
augmented_df.to_csv("hand_data_augmented.csv", index=False, header=False)
print("✅ Augmented dataset saved as 'hand_data_augmented.csv'")
