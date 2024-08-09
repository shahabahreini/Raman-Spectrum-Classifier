import configparser


def modify_config(file_path, new_settings):
    """
    Reads and modifies configuration variables from a config file.

    Parameters:
    - file_path (str): Path to the config file.
    - new_settings (dict): Dictionary containing new settings.

    Returns:
    - dict: Dictionary containing updated settings.
    """
    config = configparser.ConfigParser()
    config.read(file_path)

    for key, value in new_settings.items():
        config["SETTINGS"][key] = str(value)

    # Save the updated configuration back to the file
    with open(file_path, "w") as config_file:
        config.write(config_file)

    return {key: config["SETTINGS"][key] for key in config["SETTINGS"]}


# Usage example:
new_settings = {
    "i": 0,
    "enable_scaling": False,
    "method": "SVD",
    "centering_method": "median",
    "csv1_path": "Rab2.csv",
    "csv2_path": "Wt3.csv",
    "csv_spectrum_path": "Spectrum_650.csv",
}

updated_settings = modify_config("config.ini", new_settings)
