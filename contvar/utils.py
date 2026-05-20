import numpy as np
import h5py
from tqdm import tqdm


def load_all_embeddings(path):
    """Load ESM2 embeddings from h5 file into RAM."""
    embeddings = {}
    print(f"Loading embeddings from {path} into RAM once...")
    with h5py.File(path, "r") as h5_file:
        for key in tqdm(h5_file.keys(), desc="Loading H5"):
            embeddings[key.lower()] = np.array(h5_file[key])
    return embeddings
