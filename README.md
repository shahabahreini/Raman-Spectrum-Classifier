# Raman Spectrum Classifier

This is an opensource project to identify Raman spectrum by using Machine Learning Classification method. The program receives at least two training sets to learn and classifies groups. The core of the classification method is **RandomForestClassifier** in **SKLearn** (please check [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) for more information) Python library. Please note **Python 3.x** above is required to run the program.<br/>
FYI: The GUI is designed in PyQT5 environment.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required packages. Please download the depository as a zip run the following command in extracted folder:

```bash
pip install -r requirements.txt
pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
```

## Usage
Please put all data files into "data" folder. If the folder does not exist create one.
```python
python3 GUI.py
```
or (depends on Python configuration and the operating system)
```python
python GUI.py
```

## Screenshot

![Main Window](/screenshots/screenshot_AccuracyTest.png)


