from lib import *
from setting import modify_config

# Usage example:
new_settings = {
    "i": 0,  # nth component of SVD
    "enable_scaling": False,  # Scaling/Normalization
    "method": "PolynomialPCA",  # 'SVD', 't-SNE', 'PCA', "NMF", "SparseSVD", "PolynomialPCA", or 'UMAP'
    "centering_method": "variance",  # median|variance|min-max|log|percent|quantile
    "csv1_path": "Rab2.csv",
    "csv2_path": "Wt3.csv",
    "csv_spectrum_path": "Spectrum_650.csv",
}
updated_settings = modify_config("config.ini", new_settings)


def main():
    type1_data = pd.read_csv(updated_settings["csv1_path"], header=None).T
    type2_data = pd.read_csv(updated_settings["csv2_path"], header=None).T
    spectrum_list = pd.read_csv(updated_settings["csv_spectrum_path"], header=None)
    spectrum_list = spectrum_list.iloc[:, 0].tolist()
    n_type1 = type1_data.shape[0]

    i = int(updated_settings["i"])
    enable_scaling = bool(updated_settings["enable_scaling"])
    method = updated_settings["method"]
    centering_method = updated_settings["centering_method"]

    data = np.vstack([type1_data, type2_data])

    if enable_scaling:
        data = scaling_data(data)

    data = data if (centering_method in ['SVD', 'PCA', 'SparseSVD']) else center_data(data, centering_method)
    components = apply_dimensionality_reduction(data, method)
    plot_data(components, n_type1, method, i if (centering_method in ['SVD', 'PCA', 'SparseSVD', 'PolynomialPCA']) else 0)
    #plot_difference(type1_data, type2_data, spectrum_list)


if __name__ == "__main__":
    main()
