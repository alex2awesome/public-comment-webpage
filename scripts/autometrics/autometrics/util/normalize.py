import numpy as np
from bisect import bisect_right

def find_distinct_quintiles_with_min_max(dataset, target_variable):
    # Get unique, sorted values
    df = dataset.get_dataframe()
    target_col = df[target_variable].drop_duplicates().sort_values()
    values = target_col.values
    n = len(values)
    
    if n < 5:
        # If fewer than 5 distinct values, just return as many as you have, plus min/max if possible
        # For example, if you have only [0,1,2], you'll just return [0,1,2]
        # The user wants stable behavior: just return what you can
        return values.tolist()
    
    # Direct indexing for quartiles:
    # Indices: Q1, Q2, Q3 are chosen based on their position in the sorted array
    q1_idx = int(np.floor(0.25 * (n - 1)))
    q2_idx = int(np.floor(0.50 * (n - 1)))
    q3_idx = int(np.floor(0.75 * (n - 1)))
    
    quartiles = [
        values[0],          # min
        values[q1_idx],     # Q1
        values[q2_idx],     # Q2 (median)
        values[q3_idx],     # Q3
        values[-1]          # max
    ]
    
    return quartiles

def map_to_bucket(value, boundaries):
    # If boundaries = [0,1,2,3,4], buckets: 
    #   0 -> bucket 0 (0 to 1)
    #   1 -> bucket 1 (1 to 2)
    #   ...
    idx = bisect_right(boundaries, value) - 1
    # Clamp the index to valid range
    idx = max(0, min(idx, len(boundaries) - 1))
    return idx