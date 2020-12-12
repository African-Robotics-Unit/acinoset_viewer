import pickle
from typing import Dict

def load_pickle(pickle_file) -> Dict:
    """
    Loads a skeleton dictionary from a saved skeleton .pickle file
    """
    with open(pickle_file, 'rb') as handle:
        data = pickle.load(handle)

    return(data)