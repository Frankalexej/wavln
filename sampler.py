import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, poisson
from mio import *

# --------------------------
# Gamma Sampling
# --------------------------
def generate_gamma_samples_loaded(shape, loc, scale, size=10000, shift=0.0): 
    """
    Generate gamma-distributed random samples with the given parameters.

    Parameters:
    shape (float): Shape parameter of the gamma distribution.
    loc (float): Location parameter of the gamma distribution.
    scale (float): Scale parameter of the gamma distribution.
    size (int, optional): Number of samples to generate. Defaults to 10000.
    size (float, optional): If the samples has a minimum requirement, set this. Defaults to 0.0.

    Returns:
    numpy.ndarray: Array of gamma-distributed random samples.

    Example:
    >>> shape = 2
    >>> loc = 1
    >>> scale = 3
    >>> size = 1000
    >>> generate_gamma_samples_loaded(shape, loc, scale, size).shape
    (1000,)
    """
    # Load gamma parameters for word and sound lengths
    samples = gamma.rvs(shape, loc, scale, size=size) + shift
    return samples

# --------------------------
# Gammas to Sum
# --------------------------
def gamma_samples_sum(total_num, params, shift=0.0): 
    """
    Generate gamma-distributed random samples with the given parameters, 
    such that sum of samples excepts last sample is approximately equal to total_num.

    Parameters:
    total_num (float): The target sum of the generated samples.
    params (tuple): A tuple of shape, loc, and scale parameters for the gamma distribution.

    Returns:
    numpy.ndarray: An array of gamma-distributed random samples whose sum excepts last sample 
                    is approximately equal to total_num.

    Example:
    >>> params = (2, 1, 3)
    >>> total_num = 10
    >>> samples = gamma_samples_sum(total_num, params)
    >>> assert abs(sum(samples) - total_num) < 0.1
    """
    
    # Explain why no failure control implemented.
    # The gamma distribution is not a discrete distribution, 
    # so we aim to sum floats rather than integers, making it difficult to achieve an exact sum. 
    # Therefore, no need to waste time trying and still risk failure.
    # Hence, we proceed without any failure control.
    shape, loc, scale = params
    samples = []
    total = 0
    
    while total < total_num:
        sample = generate_gamma_samples_loaded(shape, loc, scale, size=1, shift=shift)
        samples.append(sample)
        total += sample
        
        if total > total_num:
            last_sample = samples.pop()
            total -= last_sample
            samples.append(total_num - total)
            break
    return np.array(samples)


# --------------------------
# Do Stats
# --------------------------
def statlize(samples, histtitle="", x_name="", range=None): 
    """
    Plot a histogram of the given samples and print basic statistics of the distribution.

    Parameters:
    -----------
    samples : numpy.ndarray
        Samples to plot histogram of and calculate statistics.
    histtitle : str, optional
        Title of the histogram plot. Default is an empty string.
    x_name : str, optional
        Name of the x-axis in the histogram plot. Default is an empty string.

    Returns:
    --------
    None
    """
    # Calculate some basic statistics on the numbers
    mean_samples = np.mean(samples)
    median_samples = np.median(samples)
    std_samples = np.std(samples)
    min_sample = np.min(samples)
    max_sample = np.max(samples)

    # Plot a histogram of the durations
    plt.hist(samples, color='blue', range=range)
    plt.axvline(mean_samples, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(median_samples, color='green', linestyle='dashed', linewidth=2, label='Median')
    plt.legend()
    plt.title("Histogram of " + histtitle)
    plt.xlabel(x_name)
    plt.ylabel("Frequency")
    plt.show()

    # Print the statistics
    print("Mean number: {:.2f}".format(mean_samples))
    print("Median number: {:.2f}".format(median_samples))
    print("Standard deviation of number: {:.2f}".format(std_samples))
    print("Minimum number: {:.2f}".format(min_sample))
    print("Maximum number: {:.2f}".format(max_sample))

# --------------------------
# Sample (phones/word) to phone indices
# --------------------------
def samples2idx(samples): 
    """
    Given a numpy array of sample counts per segment, computes the starting and ending index
    for each segment based on the cumulative sum of samples. The output only contains the
    ending index for each segment.
    
    Args:
    - samples: A 1D numpy array of integers representing the number of samples per segment.
    
    Returns:
    - endidx: A 1D numpy array of integers representing the ending index for each segment.
    """
    endidx = np.cumsum(samples)
    return endidx[:-1]


# --------------------------
# Sample (phones/word) to phone indices (with SE)
# --------------------------
def samples2idx_with_se(samples): 
    """
    Convert an array of sample sizes to an array of start and end indices.

    Parameters:
    samples (numpy.ndarray): Array of sample sizes.

    Returns:
    tuple: A tuple of two numpy.ndarrays - start indices and end indices.
           The start indices are obtained by adding a 0 at the beginning of the
           cumsum of samples array, and removing the last element.
           The end indices are obtained by taking the cumsum of samples array.

    Example:
    >>> samples = np.array([2, 3, 4])
    >>> samples2idx_with_se(samples)
    (array([0, 2, 5]), array([2, 5, 9]))
    """
    endidx = np.cumsum(samples)
    return np.concatenate(([0], endidx[:-1])), endidx


# --------------------------
# Filtered Start-End Pairs
# --------------------------
def idx2se(df):
    """
    Extracts start and end sequence from a pandas dataframe containing start_time and end_time columns.

    Args:
        df (pd.DataFrame): pandas dataframe containing start_time and end_time columns.

    Returns:
        Tuple of two numpy arrays containing start and end sequence respectively.
    """
    start_seq = df["start_time"].to_numpy()
    end_seq = df["end_time"].to_numpy()
    return start_seq, end_seq

def generate_phone_sample(phones_length_gamma, size=1): 
    """
    Generate a random sample of sound lengths drawn from a gamma distribution specified by the given parameters.
    
    Parameters:
    phones_length_gamma (str): The filename of the gamma distribution parameters for sound lengths.
    size (int): The number of samples to generate.
    
    Returns:
    numpy.ndarray: The generated sample of sound lengths.
    """
    # Load the gamma distribution parameters for sound lengths
    shape, loc, scale = load_gamma_params(phones_length_gamma)
    
    # Generate a random sample of sound lengths
    sound_lengths = gamma.rvs(shape, loc, scale, size=size)
    
    # Compute the mean and standard deviation of the sample
    mean = np.mean(sound_lengths)
    std_dev = np.std(sound_lengths)

    # Print the results
    print(f"Fitted shape parameter: {shape}")
    print(f"Fitted scale parameter: {scale}")
    print(f"Sample mean: {mean}")
    print(f"Sample standard deviation: {std_dev}")
    
    return sound_lengths
