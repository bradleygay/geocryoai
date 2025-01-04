import os
import h5py
import pandas as pd
import numpy as np
from pathlib import Path

def chunk_h5(input_path, output_dir, chunk_size_gb=0.025):  # 25MB chunks
    """Chunk an H5 file into smaller pieces."""
    chunk_size = chunk_size_gb * 1024 * 1024 * 1024  # Convert to bytes
    with h5py.File(input_path, 'r') as f:
        datasets = list(f.keys())
        
        for dataset_name in datasets:
            dataset = f[dataset_name]
            total_size = dataset.size * dataset.dtype.itemsize
            num_chunks = int(np.ceil(total_size / chunk_size))
            
            chunk_length = max(1, len(dataset) // num_chunks)
            
            for i in range(0, len(dataset), chunk_length):
                end_idx = min(i + chunk_length, len(dataset))
                
                chunk_filename = f"{Path(input_path).stem}_chunk_{i//chunk_length+1:03d}.h5"
                output_path = os.path.join(output_dir, chunk_filename)
                
                with h5py.File(output_path, 'w') as chunk_file:
                    chunk_file.create_dataset(
                        dataset_name,
                        data=dataset[i:end_idx],
                        compression='gzip',
                        compression_opts=9
                    )

def chunk_parquet(input_path, output_dir, chunk_size_gb=0.025):  # 25MB chunks
    """Chunk a Parquet file into smaller pieces."""
    chunk_size = chunk_size_gb * 1024 * 1024 * 1024  # Convert to bytes
    df = pd.read_parquet(input_path)
    
    # Calculate approximate size per row
    sample_size = df.memory_usage(deep=True).sum() / len(df)
    rows_per_chunk = int(chunk_size / sample_size)
    
    # Ensure we have at least one row per chunk
    rows_per_chunk = max(1, rows_per_chunk)
    
    for i in range(0, len(df), rows_per_chunk):
        end_idx = min(i + rows_per_chunk, len(df))
        
        chunk_filename = f"{Path(input_path).stem}_chunk_{i//rows_per_chunk+1:03d}.parquet"
        output_path = os.path.join(output_dir, chunk_filename)
        
        df.iloc[i:end_idx].to_parquet(
            output_path,
            compression='gzip',
            compression_level=9
        )

def main():
    # Set paths
    # base_dir = "/Users/bgay/JGRMLC"  # Update this path as needed
    base_dir = "/Users/bgay/geocryoai/data/JGRMLC"
    h5_output_dir = os.path.join(base_dir, "h5_chunks")
    parquet_output_dir = os.path.join(base_dir, "parquet_chunks")
    
    os.makedirs(h5_output_dir, exist_ok=True)
    os.makedirs(parquet_output_dir, exist_ok=True)
    
    # Process H5 file
    h5_path = "/Volumes/earthdata/github/geocryoai/data/JGRMLC/ensemble_tensor.h5"
    # h5_path = os.path.join(base_dir, "ensemble_tensor.h5")
    if os.path.exists(h5_path):
        print(f"Chunking {h5_path}...")
        chunk_h5(h5_path, h5_output_dir)
    
    # Process Parquet file
    parquet_path = "/Volumes/earthdata/github/geocryoai/data/JGRMLC/final_fcfch4alt_monthly_1km_ds.parquet"
    # parquet_path = os.path.join(base_dir, "final_fcfch4alt_monthly_1km_ds.parquet")
    if os.path.exists(parquet_path):
        print(f"Chunking {parquet_path}...")
        chunk_parquet(parquet_path, parquet_output_dir)

if __name__ == "__main__":
    main()
