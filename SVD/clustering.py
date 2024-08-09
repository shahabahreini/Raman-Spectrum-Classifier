import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from setting import updated_settings

# Hierarchical clustering explained
# https://towardsdatascience.com/hierarchical-clustering-explained-e59b13846da8

def plot_dendrogram(data, labels):
    """
    Plots a dendrogram using the given data.

    Parameters:
    - data: The data to be used for hierarchical clustering.
    - labels: Labels for each data point for visualization on the dendrogram.
    """
    linked = linkage(data, "single")

    plt.figure(figsize=(12, 7))
    dendrogram(
        linked,
        orientation="top",
        labels=labels,
        distance_sort="descending",
        show_leaf_counts=True,
    )
    plt.show()


def main(csv1_path, csv2_path):
    # Load the two CSVs
    data1 = pd.read_csv(csv1_path, header=None).T
    data2 = pd.read_csv(csv2_path, header=None).T

    # Combine the data from both CSVs
    combined_data = np.vstack([data1, data2])

    # Create labels for the data points
    labels = ["Type1_{}".format(i) for i in range(data1.shape[0])] + [
        "Type2_{}".format(i) for i in range(data2.shape[0])
    ]

    # Plot the dendrogram
    plot_dendrogram(combined_data, labels)


# Example usage
csv1_path = updated_settings["csv1_path"]
csv2_path = updated_settings["csv2_path"]
spectrum_list = pd.read_csv(updated_settings["csv_spectrum_path"], header=None)
spectrum_list = spectrum_list.iloc[:, 0].tolist()

main(csv1_path, csv2_path)
