import os
import h5py
import numpy as np
from pathlib import Path

def get_dataset_size(dataset):
    """Calculate the size of a dataset in bytes."""
    return np.prod(dataset.shape) * dataset.dtype.itemsize

def chunk_h5(input_path, output_dir, chunk_size_gb=0.025):
    """Chunk an H5 file into smaller pieces."""
    chunk_size = chunk_size_gb * 1024 * 1024 * 1024  # Convert to bytes
    
    with h5py.File(input_path, 'r') as f:
        def process_item(name, obj):
            print(f"Found item: {name}, type: {type(obj)}")
            
            if isinstance(obj, h5py.Dataset):
                print(f"Processing dataset: {name}")
                print(f"Shape: {obj.shape}")
                print(f"dtype: {obj.dtype}")
                
                total_size = get_dataset_size(obj)
                num_chunks = int(np.ceil(total_size / chunk_size))
                
                # Only chunk if the dataset has a non-zero first dimension
                if len(obj.shape) > 0 and obj.shape[0] > 0:
                    chunk_length = max(1, obj.shape[0] // num_chunks)
                    
                    print(f"Dataset size: {total_size/1024/1024:.2f} MB")
                    print(f"Creating {num_chunks} chunks of {chunk_length} elements each")
                    
                    for i in range(0, obj.shape[0], chunk_length):
                        end_idx = min(i + chunk_length, obj.shape[0])
                        
                        # Create safe filename by replacing '/' with '_'
                        safe_name = name.replace('/', '_')
                        chunk_filename = f"{Path(input_path).stem}_{safe_name}_chunk_{i//chunk_length+1:03d}.h5"
                        output_path = os.path.join(output_dir, chunk_filename)
                        
                        print(f"Writing chunk {i//chunk_length+1}/{num_chunks} to {output_path}")
                        
                        try:
                            with h5py.File(output_path, 'w') as chunk_file:
                                chunk_data = obj[i:end_idx]
                                chunk_file.create_dataset(
                                    name,
                                    data=chunk_data,
                                    compression='gzip',
                                    compression_opts=9
                                )
                            print(f"Successfully wrote chunk of size {os.path.getsize(output_path)/1024/1024:.2f} MB")
                        except Exception as e:
                            print(f"Error writing chunk: {e}")
                else:
                    print(f"Skipping dataset {name} - not suitable for chunking")
            
        print(f"Examining file structure of {input_path}")
        f.visititems(process_item)

def main():
    # Set paths
    base_dir = "/Volumes/earthdata/pcf/pcf_results"
    h5_output_dir = os.path.join(base_dir, "pcf_results_chunks")
    
    print(f"Creating output directory: {h5_output_dir}")
    os.makedirs(h5_output_dir, exist_ok=True)
    
    # Process H5 file
    h5_path = "/Volumes/earthdata/pcf_results_2.h5"
    
    if not os.path.exists(h5_path):
        print(f"Error: Input file not found: {h5_path}")
        return
        
    print(f"Input file size: {os.path.getsize(h5_path)/1024/1024:.2f} MB")
    print(f"Chunking {h5_path}...")
    
    try:
        chunk_h5(h5_path, h5_output_dir)
        print("Chunking completed successfully")
    except Exception as e:
        print(f"Error during chunking: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
