import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker as ticker  # Required for controlling ticks
import seaborn as sns

def get_bounds(data):
    """Calculates median and 90% credible intervals for labels."""
    data = np.array(data)
    med = np.median(data)
    # Using np.percentile for cleaner quantile calculation
    upper_lim = np.percentile(data, 95)
    lower_lim = np.percentile(data, 5)
    return med, upper_lim - med, med - lower_lim

def make_production_corner(samples, labels, color=None, figsize=(20, 20), 
                           bins=20, hist_alpha=0.7, custom_bounds=None):
    """
    General wrapper to generate a high-quality corner plot with clean ticks.
    """
    # Set default color to c[4] from colorblind palette
    if color is None:
        color = sns.color_palette('colorblind')[4]

    # Apply production styling
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['text.usetex'] = False
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_context('talk')
    sns.set_style('ticks')

    ndim = samples.shape[1]
    
    fig = plt.figure(figsize=figsize)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", color])
    
    for i in range(ndim):
        # --- 1D Histograms (Diagonal) ---
        ax = fig.add_subplot(ndim, ndim, int(1 + (ndim + 1) * i))
        
        data_i = samples[:, i]
        p_bounds_i = custom_bounds[i] if custom_bounds else (np.min(data_i), np.max(data_i))
        edges = np.linspace(p_bounds_i[0], p_bounds_i[1], bins)
        
        ax.hist(data_i, bins=edges, color=color, alpha=hist_alpha, density=True, zorder=0, rasterized=True)
        ax.hist(data_i, bins=edges, histtype='step', color='black', density=True, zorder=2)
        
        ax.grid(True, dashes=(1, 3))
        ax.set_xlim(p_bounds_i)

        # THE FIX: Limit x-ticks on the diagonal as well for the bottom-right parameter
        ax.xaxis.set_major_locator(ticker.MaxNLocator(2))
        
        med, up, lo = get_bounds(data_i)
        ax.axvline(med + up, color=color, ls='--')
        ax.axvline(med - lo, color=color, ls='--')
        ax.set_title(fr"${med:.2f}^{{+{up:.2f}}}_{{-{lo:.2f}}}$", fontsize=20)

        if i != 0: ax.set_yticklabels([])
        if i == ndim - 1: ax.set_xlabel(labels[i], fontsize=24)
        else: ax.set_xticklabels([])

        # --- 2D Hexbins (Lower Triangle) ---
        for j in range(i + 1, ndim):
            ax_2d = fig.add_subplot(ndim, ndim, int(1 + i + j * ndim))
            
            data_j = samples[:, j]
            p_bounds_j = custom_bounds[j] if custom_bounds else (np.min(data_j), np.max(data_j))
            
            ax_2d.hexbin(data_i, data_j, 
                         cmap=cmap, mincnt=1, gridsize=bins, rasterized=True,
                         extent=(p_bounds_i[0], p_bounds_i[1], p_bounds_j[0], p_bounds_j[1]),
                         linewidths=(0,), zorder=0)
            
            ax_2d.set_xlim(p_bounds_i)
            ax_2d.set_ylim(p_bounds_j)
            ax_2d.grid(True, dashes=(1, 3))
            
            # THE FIX: Force only two intervals (max 3 ticks) to stop overlap
            ax_2d.xaxis.set_major_locator(ticker.MaxNLocator(2, prune=None))
            ax_2d.yaxis.set_major_locator(ticker.MaxNLocator(2, prune=None))
            
            if i == 0: ax_2d.set_ylabel(labels[j], fontsize=24)
            else: ax_2d.set_yticklabels([])
                
            if j == ndim - 1: ax_2d.set_xlabel(labels[i], fontsize=24)
            else: ax_2d.set_xticklabels([])

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    return fig