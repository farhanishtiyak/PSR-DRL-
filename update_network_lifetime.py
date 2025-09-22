import csv
import os
import glob

def process_csv_file(file_path):
    print(f"Processing file: {file_path}")
    
    # Read the CSV file
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)  # Get the header row
            data = list(reader)
        except StopIteration:
            print(f"Warning: File {file_path} appears to be empty or has no valid data")
            return
    
    # Find the indices for live_nodes and live_percentage columns
    try:
        live_nodes_idx = header.index('live_nodes')
        live_percentage_idx = header.index('live_percentage')
    except ValueError:
        print(f"Error: File {file_path} does not contain required columns")
        return
    
    # Track changes
    changes_made = 0
    
    # Process the data rows
    for i in range(1, len(data)):
        # Check live_nodes
        try:
            prev_live_nodes = int(data[i-1][live_nodes_idx])
            curr_live_nodes = int(data[i][live_nodes_idx])
            
            if curr_live_nodes > prev_live_nodes:
                print(f"Fixing live_nodes at time {data[i][0]}: {curr_live_nodes} -> {prev_live_nodes}")
                data[i][live_nodes_idx] = str(prev_live_nodes)
                changes_made += 1
        except (ValueError, IndexError):
            print(f"Error processing live_nodes at row {i+1}")
        
        # Check live_percentage
        try:
            prev_live_percentage = float(data[i-1][live_percentage_idx])
            curr_live_percentage = float(data[i][live_percentage_idx])
            
            if curr_live_percentage > prev_live_percentage:
                print(f"Fixing live_percentage at time {data[i][0]}: {curr_live_percentage} -> {prev_live_percentage}")
                data[i][live_percentage_idx] = str(prev_live_percentage)
                changes_made += 1
        except (ValueError, IndexError):
            print(f"Error processing live_percentage at row {i+1}")
    
    if changes_made > 0:
        # Create backup of the original file
        backup_path = file_path + '.bak'
        if not os.path.exists(backup_path):
            os.rename(file_path, backup_path)
            print(f"Created backup of original file at: {backup_path}")
        
        # Write the corrected data
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)
        
        print(f"Made {changes_made} corrections to the file")
    else:
        print("No corrections needed")

def process_all_comparison_folders():
    base_path = "/home/ishtiyak/Desktop/Thesis/MatLab/Comparison"
    subdirs = ["10", "30", "50", "100"]
    total_files_processed = 0
    total_corrections = 0
    
    for subdir in subdirs:
        folder_path = os.path.join(base_path, subdir)
        if os.path.isdir(folder_path):
            print(f"\nProcessing folder: {folder_path}")
            # Look for CSV files containing "network_lifetime" or "simulation_data"
            csv_pattern = os.path.join(folder_path, "*network_lifetime*.csv")
            csv_files = glob.glob(csv_pattern)
            
            # Also search for simulation data files
            sim_pattern = os.path.join(folder_path, "*simulation_data*.csv")
            csv_files.extend(glob.glob(sim_pattern))
            
            if not csv_files:
                print(f"No matching CSV files found in {folder_path}")
                continue
                
            print(f"Found {len(csv_files)} file(s) to process")
            for csv_file in csv_files:
                print(f"\n{'='*50}")
                process_csv_file(csv_file)
                total_files_processed += 1
    
    print(f"\n{'='*50}")
    print(f"Total files processed: {total_files_processed}")

if __name__ == "__main__":
    process_all_comparison_folders()
