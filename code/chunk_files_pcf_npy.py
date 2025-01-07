import os
import numpy as np
from pathlib import Path

def chunk_npy(input_path, output_dir, chunk_size_gb=0.025):
    """Chunk a NumPy .npy file into smaller pieces using memory mapping."""
    chunk_size = chunk_size_gb * 1024 * 1024 * 1024  # Convert to bytes
    
    print(f"Memory mapping data from: {input_path}")
    # Load the array with memory mapping
    data = np.load(input_path, mmap_mode='r')
    
    print(f"Array shape: {data.shape}")
    print(f"Array dtype: {data.dtype}")
    print(f"Total size in bytes: {data.nbytes}")
    
    # Calculate total size and number of chunks
    total_size = data.nbytes
    num_chunks = int(np.ceil(total_size / chunk_size))
    
    print(f"Will create {num_chunks} chunks")
    
    # Calculate chunk length based on the first dimension
    chunk_length = max(1, len(data) // num_chunks)
    print(f"Each chunk will contain {chunk_length} elements from the first dimension")
    
    # Create chunks
    for i in range(0, len(data), chunk_length):
        end_idx = min(i + chunk_length, len(data))
        
        # Create chunk filename
        chunk_filename = f"{Path(input_path).stem}_chunk_{i//chunk_length+1:03d}.npy"
        output_path = os.path.join(output_dir, chunk_filename)
        
        print(f"\nProcessing chunk {i//chunk_length+1}/{num_chunks}")
        print(f"Indices {i}:{end_idx}")
        
        # Save chunk
        try:
            chunk_data = data[i:end_idx].copy()  # Copy needed for mmap_mode
            np.save(output_path, chunk_data)
            chunk_size = os.path.getsize(output_path)
            print(f"Saved chunk: {chunk_size/1024/1024:.2f} MB")
        except Exception as e:
            print(f"Error saving chunk: {e}")
            continue

def main():
    # Set paths
    base_dir = "/Volumes/earthdata/pcf/pcf_model_2"
    npy_output_dir = os.path.join(base_dir, "predictions_chunks")
    
    os.makedirs(npy_output_dir, exist_ok=True)
    
    # Process NPY file
    npy_path = os.path.join(base_dir, "predictions.npy")
    
    file_size_gb = os.path.getsize(npy_path) / (1024**3)
    print(f"Input file size: {file_size_gb:.2f} GB")
    
    if os.path.exists(npy_path):
        print(f"Chunking {npy_path}...")
        chunk_npy(npy_path, npy_output_dir)
    else:
        print(f"Error: File not found: {npy_path}")

if __name__ == "__main__":
    main()
