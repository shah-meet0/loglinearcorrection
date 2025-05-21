import pandas as pd
import numpy as np
import os
import sys
import pickle
from data_processing import main as process_data

# Updated master_SZ.py that includes data processing step

# Clear variables (Python doesn't need explicit clearing like MATLAB)
# Add paths to sys.path (equivalent to MATLAB's addpath)
sys.path.append(os.path.join(os.getcwd(), 'toolbox'))
sys.path.append(os.path.join(os.getcwd(), 'programs'))

#%% Step 1: Process raw data and generate Excel files
print("Step 1: Processing raw Stata data...")
data_path = "./raw-data/suarezserratoWhoBenefitsState2016/ILS_3_shocks_6_parm_housing.dta"  # Path to your Stata file
estimates_dir = "./estimates"

# Run data processing (equivalent to the Stata code)
if not os.path.exists(estimates_dir):
    os.makedirs(estimates_dir)

# Check if Excel files already exist, if not, process the data
excel_files = ['base_est.xlsx', 'bartik_est.xlsx', 'full_est.xlsx']
excel_paths = [os.path.join(estimates_dir, f) for f in excel_files]

if not all(os.path.exists(path) for path in excel_paths):
    print("Excel files not found. Processing raw data...")
    if os.path.exists(data_path):
        process_data(data_path, estimates_dir)
        print("Data processing complete!")
    else:
        print(f"Error: Could not find {data_path}")
        print("Please make sure the Stata file is in the current directory.")
        sys.exit(1)
else:
    print("Excel files already exist. Skipping data processing.")

#%% Step 2: Load and format estimates from generated Excel files
print("\nStep 2: Loading processed estimates...")

# Read first Excel file
T1 = pd.read_excel(os.path.join(estimates_dir, 'base_est.xlsx'), header=None)
beta_Mod1 = T1.iloc[0, :].values
cov_Mod1 = T1.iloc[1:, :].values  # Take all rows below the first one

# Read second Excel file  
T2 = pd.read_excel(os.path.join(estimates_dir, 'bartik_est.xlsx'), header=None)
beta_Mod2 = T2.iloc[0, :].values
cov_Mod2 = T2.iloc[1:, :].values

# Read third Excel file
T3 = pd.read_excel(os.path.join(estimates_dir, 'full_est.xlsx'), header=None)
beta_Mod3 = T3.iloc[0, :].values
cov_Mod3 = T3.iloc[1:, :].values

# Create data dictionary
data_dict = {
    'beta_Mod1': beta_Mod1,
    'cov_Mod1': cov_Mod1,
    'beta_Mod2': beta_Mod2,
    'cov_Mod2': cov_Mod2,
    'beta_Mod3': beta_Mod3,
    'cov_Mod3': cov_Mod3
}

# Save using pandas serialization (pickle format)
print("Saving processed data...")
with open(os.path.join(estimates_dir, 'RFmoments.pkl'), 'wb') as f:
    pickle.dump(data_dict, f)

#%% Step 3: Generate tables
print("\nStep 3: Generating tables...")

# Import and run table functions, passing data directly
from table1 import table1
from table2 import table2

# Execute the table functions with data dictionary
table1(data_dict)
table2(data_dict)

print("\nAll steps completed successfully!")
print("="*50)
print("Generated files:")
print(f"- {os.path.join(estimates_dir, 'base_est.xlsx')}")
print(f"- {os.path.join(estimates_dir, 'bartik_est.xlsx')}")
print(f"- {os.path.join(estimates_dir, 'full_est.xlsx')}")
print(f"- {os.path.join(estimates_dir, 'RFmoments.pkl')}")
print("- ./tables/table1.tex")
print("- ./tables/table2.tex")
print("="*50)
