import pandas as pd
import numpy as np
import os

# Load the original dataset
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), 'obesity_dataset.csv') 
df= pd.read_csv(DATA_FILE_PATH)

# Generate new columns
np.random.seed(42)  # For reproducibility
df["Age"] = np.random.randint(18, 65, size=len(df))  # Age between 18 and 65
df["Exercise_Hours_Per_Week"] = np.random.randint(0, 10, size=len(df))  # Exercise between 0 and 10 hours

# Save the modified dataset
new_file_path = "obesity_dataset_extended.csv"
df.to_csv(new_file_path, index=False)

# Return the new file path
new_file_path
