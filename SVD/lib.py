import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, NMF, TruncatedSVD, KernelPCA
from scipy import stats
import umap
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Colorblind-friendly palette
cb_palette = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]


def quantile_normalize(data):
    sorted_data = np.sort(data, axis=0)
    rank_mean = np.mean(sorted_data, axis=1)
    ranks = np.apply_along_axis(stats.rankdata, 0, data).astype(int) - 1
    return rank_mean[ranks]


def center_data(data, method):
    if method == "median":
        return data - np.mean(data, axis=0)
    elif method == "variance":
        return data / np.std(data, axis=0)
    elif method == "min-max":
        return (data - np.min(data, axis=0)) / (
            np.max(data, axis=0) - np.min(data, axis=0)
        )
    elif method == "log":
        return np.log1p(data)
    elif method == "percent":
        return data - np.percentile(data, 50, axis=0)
    elif method == "quantile":
        return quantile_normalize(data)
    else:
        raise ValueError(f"Unknown centering method: {method}")


def apply_dimensionality_reduction(data, method):
    if method == "SVD":
        U, _, _ = np.linalg.svd(data, full_matrices=False)
        return U
    elif method == "t-SNE":
        tsne = TSNE(n_components=2)
        return tsne.fit_transform(data)
    elif method == "UMAP":
        reducer = umap.UMAP()
        return reducer.fit_transform(data)
    elif method == "PCA":
        pca = PCA(n_components=2)
        return pca.fit_transform(data)
    elif method == "PolynomialPCA":
        degree = 2
        poly_pca = KernelPCA(
            n_components=2, kernel="precomputed", degree=degree
        )  # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’}
        return poly_pca.fit_transform(data)
    elif method == "NMF":
        nmf = NMF(n_components=2, init="random", random_state=0)
        return nmf.fit_transform(data)
    elif method == "SparseSVD":
        svd = TruncatedSVD(n_components=2)
        return svd.fit_transform(data)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")


def scaling_data(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data


def plot_data(components, n_type1, method, i):
    plt.figure(figsize=(12, 12))
    plt.scatter(
        components[:n_type1, i],
        components[:n_type1, i + 1],
        label="Type 1",
        alpha=0.7,
        color=cb_palette[0],
    )
    plt.scatter(
        components[n_type1:, i],
        components[n_type1:, i + 1],
        label="Type 2",
        alpha=0.7,
        color=cb_palette[1],
    )
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.title(f"{method}-based Distinction between Type 1 and Type 2")
    plt.show()


def plot_data_separate(components, n_type1, method, difference, i):
    plt.scatter(
        components[:n_type1, i],
        components[:n_type1, i + 1],
        label="Type 1",
        alpha=0.7,
        color=cb_palette[0],
    )
    plt.scatter(
        components[n_type1:, i],
        components[n_type1:, i + 1],
        label="Type 2",
        alpha=0.7,
        color=cb_palette[1],
    )
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.title(f"{method}-based Distinction between Type 1 and Type 2")
    plt.show()

    # Plot the difference against the 659 components
    plt.figure(figsize=(12, 6))
    plt.plot(difference, color=cb_palette[2])
    plt.xlabel("Component")
    plt.ylabel("Difference in Signal Amplitude")
    plt.title("Difference between Mean Signals of Type 1 and Type 2")
    plt.show()


def plot_difference_unnormalized(type1_data, type2_data, spectrum):
    mean_type1 = np.mean(type1_data, axis=0)
    mean_type2 = np.mean(type2_data, axis=0)
    difference = abs(mean_type1 - mean_type2)

    plt.plot(spectrum, difference, color=cb_palette[3])
    plt.xlabel("Component")
    plt.ylabel("Difference in Signal Amplitude")
    plt.title("Difference between Mean Signals of Type 1 and Type 2")
    plt.tight_layout()
    plt.show()


def plot_difference(type1_data, type2_data, spectrum):
    # Normalize type1_data
    type1_data = (type1_data - np.min(type1_data, axis=0)) / (
        np.max(type1_data, axis=0) - np.min(type1_data, axis=0)
    )

    # Normalize type2_data
    type2_data = (type2_data - np.min(type2_data, axis=0)) / (
        np.max(type2_data, axis=0) - np.min(type2_data, axis=0)
    )

    mean_type1 = np.mean(type1_data, axis=0)
    mean_type2 = np.mean(type2_data, axis=0)
    difference = abs(mean_type1 - mean_type2)

    std_type1 = np.std(type1_data, axis=0)
    std_type2 = np.std(type2_data, axis=0)
    difference_error = np.sqrt(
        std_type1 ** 2 / len(type1_data) + std_type2 ** 2 / len(type2_data)
    )
    relative_error = (difference_error / difference) * 100

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot difference on the primary y-axis
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Difference in Signal Amplitude", color=cb_palette[0])
    (line1,) = ax1.plot(spectrum, difference, label="Difference", color=cb_palette[0])
    fill = ax1.fill_between(
        spectrum,
        difference - difference_error,
        difference + difference_error,
        color=cb_palette[1],
        alpha=0.5,
        label="Difference Error",
    )
    ax1.tick_params(axis="y", labelcolor=cb_palette[0])

    # Create secondary y-axis for relative error
    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.set_ylabel("Relative Error (%)", color=cb_palette[2])
    (line2,) = ax2.plot(
        spectrum,
        relative_error,
        label="Relative Error (%)",
        color=cb_palette[2],
        linestyle="--",
    )
    ax2.tick_params(axis="y", labelcolor=cb_palette[2])

    # Combine legends from both y-axes
    lines = [line1, fill, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc=2)

    plt.title(
        "Difference and Relative Error between Mean Signals of Type 1 and Type 2 after Normalization"
    )
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

    def on_xlims_change(axes):
        ax2.relim()
        ax2.autoscale_view(scaley=True)

    ax1.callbacks.connect("xlim_changed", on_xlims_change)

    fig.tight_layout()
    plt.show()
