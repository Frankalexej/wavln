import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

def plot_attention_comparison_and_significance(p2s, s2p, v2p, p2v, save_path):
    n_steps = 100
    segment_length = int(n_steps * 0.2)  # Calculate 20% segment length

    # Calculate the means for the first and last 20% segments for each interaction
    first_20_p2s_mean = np.mean(p2s[:, :segment_length], axis=1).mean()
    last_20_s2p_mean = np.mean(s2p[:, -segment_length:], axis=1).mean()
    first_20_v2p_mean = np.mean(v2p[:, :segment_length], axis=1).mean()
    last_20_p2v_mean = np.mean(p2v[:, -segment_length:], axis=1).mean()

    # Perform statistical tests between the three specified pairs
    _, p_val_s2p_vs_p2s = ttest_rel(s2p[:, -segment_length:].flatten(), p2s[:, :segment_length].flatten())
    _, p_val_p2s_vs_p2v = ttest_rel(p2s[:, :segment_length].flatten(), p2v[:, -segment_length:].flatten())
    _, p_val_p2v_vs_v2p = ttest_rel(p2v[:, -segment_length:].flatten(), v2p[:, :segment_length].flatten())

    # Plotting
    labels = ['First 20% P->S', 'Last 20% S->P', 'First 20% V->P', 'Last 20% P->V']
    means = [first_20_p2s_mean, last_20_s2p_mean, first_20_v2p_mean, last_20_p2v_mean]
    x = np.arange(len(labels))  # Label locations
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, means, color=['blue', 'green', 'red', 'orange'])

    # Mark significance directly on the bar graph
    significance_threshold = 0.05
    signif_positions = [1, 0, 3]  # Positions for text annotations for the three comparisons
    p_values = [p_val_s2p_vs_p2s, p_val_p2s_vs_p2v, p_val_p2v_vs_v2p]

    for pos, p_val in zip(signif_positions, p_values):
        if p_val < significance_threshold:
            ax.text(pos, max(means) * 1.05, '*', ha='center', fontsize=12, color='black')

    ax.set_ylabel('Average Attention')
    ax.set_title('Attention Averages and Significance')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(means) * 1.2)  # Adjust y-axis limit for visibility

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
