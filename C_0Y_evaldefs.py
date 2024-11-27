import dis
import numpy as np
import scipy
import torch
from colorama import Fore, Style

def filter_data_by_tags(data, tags, select):
    # Convert tag_list to a set for faster membership testing
    tag_set = set(select)

    # Create a boolean mask where True indicates the tag is in tag_list
    mask = np.isin(tags, list(tag_set))

    # Use the mask to filter the data and tags arrays
    filtered_data = data[mask]
    filtered_tags = tags[mask]

    return filtered_data, filtered_tags

def filter_data_by_tags_to_list(data, tags, tag_list):
    result = []

    for tag in tag_list:
        # Create a boolean mask where True indicates the tag matches the current tag
        mask = (tags == tag)
        # Use the mask to filter the data array for the current tag
        filtered_data = data[mask]
        # Append the filtered data to the result list
        result.append(filtered_data)

    return result

def postproc_standardize(data, tags, outlier_ratio=0, denan=True):
    if denan: 
        # Remove NaN values
        mask = ~np.isnan(data).any(axis=1)
        data = data[mask]
        tags = tags[mask] 
        nannum = np.sum(~mask)

    if outlier_ratio > 0: 
        # Step 1: Remove outliers
        low_percentile = np.percentile(data, outlier_ratio, axis=0)
        high_percentile = np.percentile(data, 100-outlier_ratio, axis=0)

        # Keep rows where all elements are within the percentiles
        mask = (data > low_percentile) & (data < high_percentile)
        filtered_data = data[mask.all(axis=1)]
        filtered_tags = tags[mask.all(axis=1)]
    else: 
        filtered_data = data
        filtered_tags = tags

    # Step 2: Standardization
    # Calculate mean and std only from the filtered data
    mean = np.mean(filtered_data, axis=0)
    std = np.std(filtered_data, axis=0)
    eps = 1e-9

    # Z-score standardization
    standardized_data = (filtered_data - mean) / (std + eps)
    # in s-score normalization here, we added an epsilon to avoid 0 std
    if denan: 
        return standardized_data, filtered_tags, nannum
    else:
        return standardized_data, filtered_tags

def indicator_function(condition):
    return 1 if condition else 0

def unsym_abx_error(cap_delta, cap_ksi, distance):
    n_delta = cap_delta.shape[0]
    n_ksi = cap_ksi.shape[0]
    sum_value = 0

    for a in range(n_delta):
        for b in range(n_ksi):
            for x in range(n_delta):
                if x != a:
                    sum_value += (indicator_function(
                        distance(cap_delta[x], cap_ksi[b]) < distance(cap_delta[a], cap_delta[x])
                    ) + 0.5 * indicator_function(
                        distance(cap_delta[x], cap_ksi[b]) == distance(cap_delta[a], cap_delta[x])
                    ))

    return 1 / (n_delta * (n_delta - 1) * n_ksi) * (sum_value)

def sym_abx_error(cap_delta, cap_ksi, distance):
    return 0.5 * (unsym_abx_error(cap_delta, cap_ksi, distance) + unsym_abx_error(cap_ksi, cap_delta, distance))

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def cosine_distance(x, y): 
    return scipy.spatial.distance.cosine(x, y)


# Optimized unsymmetrical ABX error
def parallel_unsym_abx_error(cap_delta, cap_ksi, distance_fn):
    n_delta = cap_delta.shape[0]
    dist_delta_delta = distance_fn(cap_delta.unsqueeze(1), cap_delta.unsqueeze(0)).squeeze()
    dist_delta_ksi = distance_fn(cap_delta.unsqueeze(1), cap_ksi.unsqueeze(0)).squeeze()
    mask = torch.eye(n_delta, device=cap_delta.device).bool()
    valid_indices = ~mask

    delta_to_ksi = dist_delta_ksi.unsqueeze(1).expand(-1, n_delta, -1)
    delta_to_delta = dist_delta_delta.unsqueeze(2).expand(-1, -1, cap_ksi.shape[0])
    comparisons = delta_to_ksi[valid_indices] < delta_to_delta[valid_indices]
    ties = delta_to_ksi[valid_indices] == delta_to_delta[valid_indices]

    sum_value = comparisons.sum().float() + 0.5 * ties.sum().float()
    return sum_value / (n_delta * (n_delta - 1) * cap_ksi.shape[0])

def parallel_sym_abx_error(cap_delta, cap_ksi, distance_fn):
    return 0.5 * (
        parallel_unsym_abx_error(cap_delta, cap_ksi, distance_fn) + 
        parallel_unsym_abx_error(cap_ksi, cap_delta, distance_fn)
    )

# Euclidean distance function for PyTorch
# def euclidean_distance(x, y): 
#     # overwrite the np based distance calculation
#     return torch.cdist(x, y, p=2)

def euclidean_distance(x, y):
    # return torch.cdist(x, y, p=2)
    x1, x2 = x, y
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment
    x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

    # Compute squared distance matrix using quadratic expansion
    # But be clever and do it with a single matmul call
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))

    # Zero out negative values
    res.clamp_min_(1e-30).sqrt_()
    return res

def sym_abx_error(cap_delta, cap_ksi, distance): 
    # device = "cpu"
    cap_delta_gpu = torch.tensor(cap_delta, dtype=torch.float32)
    cap_ksi_gpu = torch.tensor(cap_ksi, dtype=torch.float32)
    try: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cap_delta_gpu = cap_delta_gpu.to(device)
        cap_ksi_gpu = cap_ksi_gpu.to(device)
        abx_score = parallel_sym_abx_error(cap_delta=cap_delta_gpu, 
                                    cap_ksi=cap_ksi_gpu, 
                                    distance_fn=distance).item()
    except RuntimeError as e: 
        print(Fore.RED + "GPU error detected, falling back to CPU...")
        print(Style.RESET_ALL)
        device = torch.device("cpu")
        cap_delta_gpu = cap_delta_gpu.to(device)
        cap_ksi_gpu = cap_ksi_gpu.to(device)
        abx_score = parallel_sym_abx_error(cap_delta=cap_delta_gpu, 
                                    cap_ksi=cap_ksi_gpu, 
                                    distance_fn=distance).item()
    del cap_delta_gpu  # Delete unused variables
    del cap_ksi_gpu
    torch.cuda.empty_cache()  # Clear cached memory
    return abx_score
    