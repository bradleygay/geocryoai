import os
import h5py
import numpy as np
from pathlib import Path
from glob import glob

def reassemble_h5(chunks_dir, output_path):
    """Reassemble H5 chunks into a single file."""
    print(f"Scanning directory: {chunks_dir}")
    
    # Get all chunk files and sort them
    chunk_files = glob(os.path.join(chunks_dir, "*.h5"))
    chunk_files.sort()
    
    if not chunk_files:
        print("No chunk files found!")
        return
        
    print(f"Found {len(chunk_files)} chunk files")
    
    # Get dataset information from first chunk
    with h5py.File(chunk_files[0], 'r') as f:
        datasets = list(f.keys())
        
    # Process each dataset separately
    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Get total shape and dtype from chunks
        chunks_data = []
        dtype = None
        
        for chunk_file in chunk_files:
            with h5py.File(chunk_file, 'r') as f:
                if dataset_name in f:
                    chunk = f[dataset_name][:]
                    chunks_data.append(chunk)
                    if dtype is None:
                        dtype = chunk.dtype
        
        if chunks_data:
            # Concatenate all chunks
            print("Concatenating chunks...")
            full_data = np.concatenate(chunks_data, axis=0)
            
            # Create or update the output file
            print(f"Saving to {output_path}")
            with h5py.File(output_path, 'a') as f:
                if dataset_name in f:
                    del f[dataset_name]
                f.create_dataset(
                    dataset_name,
                    data=full_data,
                    compression='gzip',
                    compression_opts=9
                )
            
            print(f"Successfully reassembled dataset {dataset_name}")
            print(f"Final shape: {full_data.shape}")

def main():
    chunks_dir = "/Volumes/earthdata/geocryoai/results/pcf_results_chunks"
    output_path = "/Volumes/earthdata/geocryoai/results/pcf_results_reassembled.h5"
    
    print(f"Starting reassembly of H5 chunks...")
    reassemble_h5(chunks_dir, output_path)

if __name__ == "__main__":
    main()
