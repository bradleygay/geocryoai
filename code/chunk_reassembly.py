import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

def reconstruct_h5(chunk_dir, output_path):
    """Reconstruct H5 file from chunks"""
    # Get all chunk files sorted numerically
    chunk_files = sorted(glob(os.path.join(chunk_dir, '*.h5')))
    
    if not chunk_files:
        print("No H5 chunks found")
        return
        
    print(f"Found {len(chunk_files)} H5 chunks")
    
    # Read first chunk to get dataset info
    with h5py.File(chunk_files[0], 'r') as f:
        dataset_names = list(f.keys())
    
    # Create output file and datasets
    with h5py.File(output_path, 'w') as out_file:
        # Process each dataset
        for dataset_name in dataset_names:
            # Get data shape and type from first chunk
            with h5py.File(chunk_files[0], 'r') as f:
                chunk_data = f[dataset_name][:]
                data_type = chunk_data.dtype
            
            # Calculate total length
            total_length = 0
            for chunk_path in chunk_files:
                with h5py.File(chunk_path, 'r') as f:
                    total_length += len(f[dataset_name])
            
            # Create dataset in output file
            dataset_shape = list(chunk_data.shape)
            dataset_shape[0] = total_length
            out_dataset = out_file.create_dataset(
                dataset_name,
                shape=tuple(dataset_shape),
                dtype=data_type,
                compression='gzip'
            )
            
            # Copy data from chunks
            current_idx = 0
            for i, chunk_path in enumerate(chunk_files):
                print(f"Processing chunk {i+1}/{len(chunk_files)}")
                with h5py.File(chunk_path, 'r') as f:
                    chunk_data = f[dataset_name][:]
                    chunk_length = len(chunk_data)
                    out_dataset[current_idx:current_idx+chunk_length] = chunk_data
                    current_idx += chunk_length

def reconstruct_parquet(chunk_dir, output_path):
    """Reconstruct Parquet file from chunks"""
    # Get all chunk files sorted numerically
    chunk_files = sorted(glob(os.path.join(chunk_dir, '*.parquet')))
    
    if not chunk_files:
        print("No Parquet chunks found")
        return
        
    print(f"Found {len(chunk_files)} Parquet chunks")
    
    # Read and concatenate all chunks
    dfs = []
    for i, chunk_path in enumerate(chunk_files):
        if i % 100 == 0:
            print(f"Processing chunk {i+1}/{len(chunk_files)}")
        df_chunk = pd.read_parquet(chunk_path)
        dfs.append(df_chunk)
    
    print("Concatenating chunks...")
    final_df = pd.concat(dfs, ignore_index=True)
    
    print("Saving reconstructed file...")
    final_df.to_parquet(output_path, compression='gzip')

def main():
    # Set directories
    base_dir = "data/JGRMLC"
    h5_chunks_dir = os.path.join(base_dir, "h5_chunks")
    parquet_chunks_dir = os.path.join(base_dir, "parquet_chunks")
    
    # Reconstruct H5 file
    if os.path.exists(h5_chunks_dir):
        print("Reconstructing H5 file...")
        output_h5 = os.path.join(base_dir, "ensemble_tensor_reconstructed.h5")
        reconstruct_h5(h5_chunks_dir, output_h5)
        
    # Reconstruct Parquet file
    if os.path.exists(parquet_chunks_dir):
        print("Reconstructing Parquet file...")
        output_parquet = os.path.join(base_dir, "final_fcfch4alt_monthly_1km_ds_reconstructed.parquet")
        reconstruct_parquet(parquet_chunks_dir, output_parquet)

if __name__ == "__main__":
    main()
