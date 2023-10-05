import os
import pandas as pd
import re
from paths import phone_seg_anno_log_path
from misc_progress_bar import draw_progress_bar


def cleanplusone(): 
    # Directory containing CSV files
    directory_path = phone_seg_anno_log_path

    # Iterate through CSV files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            # Read the CSV file into a DataFrame
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            
            # Check and update the "token" column
            if 'token' in df.columns:
                df['token'] = df['token'].apply(lambda x: re.sub(r'\+1$', '', x) if isinstance(x, str) else x)
                
                # Save the updated DataFrame back to the same file
                df.to_csv(file_path, index=False)
                print(f"Processed and updated: {filename}")
            else:
                print(f"No 'token' column found in {filename}")

    print("Processing complete.")

if __name__ == "__main__": 
    cleanplusone()