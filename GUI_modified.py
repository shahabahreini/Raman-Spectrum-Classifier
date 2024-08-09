# Version 2.2.2

import copy
import os
import sys
from os.path import relpath
from multiprocessing import Pool, Value

# PyQt5 GUI
from PyQt5 import uic
from PyQt5.QtWidgets import (
    QDialog,
    QApplication,
    QLabel,
    QPushButton,
    QLCDNumber,
    QFileDialog,
    QListWidget,
    QProgressBar,
    QCheckBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread

# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
import scikitplot as skplt
import matplotlib.pyplot as plt
import seaborn as sns

# Enable High-DPI scaling
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)  # Use high-DPI icons


class Worker(QThread):
    progress_updated = pyqtSignal(int)  # Signal to update progress
    finished = pyqtSignal()  # Signal to indicate that the thread has finished

    def run(self):
        with Pool() as pool:
            # Use imap to get results as they are ready
            for i, result in enumerate(
                pool.imap(call_accuracy_test, range(len(acc_check.X))), 1
            ):
                self.progress_updated.emit(i)  # Emit signal with progress
        self.finished.emit()  # Emit signal to indicate that the thread has finished


class UI(QDialog):
    # Define a signal that takes an integer parameter
    progress_updated = pyqtSignal(int)

    def __init__(self):
        super(UI, self).__init__()
        self.initialize_ui()

    def initialize_ui(self):
        # Load the UI file and define widgets
        uic.loadUi("loadui.ui", self)
        self.setup_widgets()
        self.connect_signals()
        self.show()

    def setup_widgets(self):
        # Define the widgets used in the UI
        self.identifying_status = self.findChild(QLabel, "label_identifying_status")
        self.button_browse = self.findChild(QPushButton, "pushButton_browse")
        self.button_samplebrowse = self.findChild(
            QPushButton, "pushButton_browse_samplefile"
        )
        self.button_featureList = self.findChild(
            QPushButton, "pushButton_browse_featureList"
        )
        self.button_run = self.findChild(QPushButton, "pushButton_accuracytest")
        self.button_identify = self.findChild(QPushButton, "pushButton_identify")
        self.lcdnumber_wrong = self.findChild(QLCDNumber, "lcdNumber_wrong_detection")
        self.lcdnumber_step = self.findChild(QLCDNumber, "lcdNumber_checks")
        self.lcdnumber_accuracy = self.findChild(QLCDNumber, "lcdNumber_accuracy")
        self.flist = self.findChild(QListWidget, "listWidget_files")
        self.sfilelist = self.findChild(QListWidget, "listWidget_samplefile")
        self.ffilelist = self.findChild(QListWidget, "listWidget_featureList")
        self.identified_result = self.findChild(
            QListWidget, "listWidget_identifiedlist"
        )
        self.suspicious_data_lst = self.findChild(
            QListWidget, "listWidget_suspiciousdata"
        )
        self.summary = self.findChild(QListWidget, "listWidget_summary")
        self.progressbar = self.findChild(QProgressBar, "loading")
        self.do_save_to_file = self.findChild(QCheckBox, "checkBox_SaveToFile")

    def connect_signals(self):
        # Connect widget signals to corresponding slots
        self.button_browse.clicked.connect(self.browse_trainingfiles)
        self.button_samplebrowse.clicked.connect(self.browse_samplefile)
        self.button_featureList.clicked.connect(self.browse_featureList)
        self.button_run.clicked.connect(self.accuracytest_serial)
        self.button_identify.clicked.connect(self.sample_identify)

    def browse_trainingfiles(self):
        self.flist.clear()
        acc_check.fnames = []
        fopath = os.path.join(os.path.dirname(__file__), "data")
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "Open CSV", fopath, "CSV Files (*.csv)"
        )
        for path in filepaths:
            fname = os.path.basename(path)
            acc_check.fpaths[(fname.replace(".csv", ""))] = path
            self.flist.addItem(fname)
            acc_check.fnames.append(fname.replace(".csv", ""))
            acc_check.number_of_files = self.flist.count()

    def browse_samplefile(self):
        self.sfilelist.clear()
        fopath = os.path.join(os.path.dirname(__file__), "data")
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open CSV", fopath, "CSV Files (*.csv)"
        )
        acc_check.sfpath = filepath
        fname = os.path.basename(filepath)
        if fname != "":
            self.sfilelist.addItem(fname)
            acc_check.sfname = fname.replace(".csv", "")
        else:
            acc_check.error = True
            UIWindow.identifying_status.setText("No sample file is chosen.")

    def browse_featureList(self):
        self.ffilelist.clear()
        fopath = os.path.join(os.path.dirname(__file__), "data")
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open CSV", fopath, "CSV Files (*.csv)"
        )
        acc_check.ffpath = filepath
        fname = os.path.basename(filepath)
        if fname != "":
            self.ffilelist.addItem(fname)
            acc_check.ffname = fname.replace(".csv", "")
        else:
            acc_check.error = True
            UIWindow.identifying_status.setText("No sample file is chosen.")

    def accuracytest_parallel(self):
        self.suspicious_data_lst.clear()
        acc_check.reset()
        acc_check.load_training_files()
        if not acc_check.error:
            self.worker = Worker()
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.finished.connect(
                self.on_worker_finished
            )  # Connect to the finished signal
            self.worker.start()

    def accuracytest_serial(self):
        self.suspicious_data_lst.clear()
        acc_check.reset()
        acc_check.load_training_files()
        if not acc_check.error:
            self.summary.clear()
            self.summary_update()
            self.identifying_status.setText("")
            self.progressbar.setMaximum(len(acc_check.X))
            self.lcdnumber_accuracy.display(round(0.0, 2))
            self.progressbar.setEnabled(True)
            while acc_check.checks <= len(acc_check.X):
                call_accuracy_test(1)
                QApplication.processEvents()  # Prevents GUI freezes in loops
                self.lcdnumber_wrong.display(acc_check.wrong_detection)
                self.lcdnumber_step.display(acc_check.checks)
                self.progressbar.setValue(acc_check.checks)
                acc_check.checks += 1
            self.update_lcd()
            self.progressbar.setEnabled(False)
            self.progressbar.setValue(0)
        acc_check.error = False

    def on_worker_finished(self):
        # Code to execute after the worker thread has finished
        self.update_lcd()
        self.progressbar.setEnabled(False)
        self.progressbar.setValue(0)
        acc_check.error = False
        # ...

    def update_progress(self, value):
        self.progressbar.setValue(value)

    def sample_identify(self):
        self.identifying_status.setText("Identifying process started ...")
        self.suspicious_data_lst.clear()
        self.identified_result.clear()
        self.summary.clear()
        QApplication.processEvents()  # Prevents GUI freezes in loops
        acc_check.reset()
        acc_check.load_training_files()
        if not acc_check.error:
            result = sample_identifer()
        if not acc_check.error:
            self.summary_update()

            # Writing to file
            if self.do_save_to_file.isChecked():
                with open("result.txt", "w") as file1:
                    for i, item in enumerate(result):
                        self.identified_result.addItem(f"Sample #{i + 1} -> {item} ")
                        # Writing data to a file
                        items = [
                            self.flist.item(x).text().replace(".csv", "")
                            for x in range(self.flist.count())
                        ]
                        file1.write(f"{items.index(item)}\n")
                        # file1.write(f'sample #{i + 1} -> {item}\n')

            else:
                for i, item in enumerate(result):
                    self.identified_result.addItem(f"Sample #{i + 1} -> {item} ")
            self.identifying_status.setText("Done!")
        acc_check.error = False

    def update_lcd(self):
        acc_check.accuracy_calc()
        self.lcdnumber_accuracy.display(round(acc_check.calculation_accuracy, 2))

    def summary_update(self):
        self.summary.addItem(f"Total samples: {len(acc_check.X)}")
        self.summary.addItem(
            f"Detected features: {acc_check.cols_length[acc_check.fnames[0]]}"
        )


class AccuracyCheck:
    def __init__(self):
        # Training-related attributes
        self.fnames = []
        self.fpaths = {}
        self.sfpath = ""
        self.ffpath = ""
        self.relative_paths = {}
        self.number_of_samples = []
        self.number_of_files = 0
        self.files_data = {}
        self.row_length = {}
        self.cols_length = {}
        self.X = []
        self.Y = []

        # Prediction-related attributes
        self.sfname = ""

        # Accuracy calculation attributes
        self.error = False
        self.wrong_detection = Value("i", 0)  # 'i' indicates an integer type
        self.checks = 1
        self.calculation_accuracy = 0
        self.suspecious_data = []

    def reset(self):
        """Reset the accuracy calculation attributes."""
        self.wrong_detection = 0
        self.checks = 1
        self.calculation_accuracy = 0
        self.files_data = {}
        self.row_length = {}
        self.cols_length = {}
        self.X = []
        self.Y = []
        self.suspecious_data = []

    def accuracy_calc(self):
        """Calculate the accuracy percentage."""
        self.calculation_accuracy = 100 - self.wrong_detection / self.checks * 100

    def load_training_files(self):
        """Load training files and preprocess the data."""
        if len(self.fnames) <= 1:
            self.error = True
            UIWindow.identifying_status.setText(
                "At least two training files are required."
            )
            return

        for fname in self.fnames:
            this_dir = os.path.dirname(__file__)
            self.relative_paths[fname] = relpath(self.fpaths[fname], this_dir)
            train_data = pd.read_csv(self.fpaths[fname], header=None)
            self.cols_length[fname], self.row_length[fname] = train_data.shape
            self.number_of_samples.append(self.row_length[fname] - 2)
            self.files_data[fname] = train_data

            self.load_samples(fname)

    def load_samples(self, fname):
        """Load samples from the given file name."""
        num_features = self.cols_length[fname]
        num_samples = self.row_length[fname]
        for i in range(1, num_samples):
            x = [self.files_data[fname].iloc[j, i] for j in range(num_features)]
            self.X.append(x)
            self.Y.append(fname)

    def suspicious_data_tracker(self):
        """Track and manage suspicious data."""
        group = self.Y[self.checks - 1]
        idx = (
            self.checks
            if self.row_length[group] >= self.checks
            else self.checks - self.row_length[group]
        )
        self.suspecious_data.append([group, idx])
        UIWindow.suspicious_data_lst.addItem(f"{group}\t-> {idx}")


def accuracy_sequential_checker(fnames):
    X = copy.deepcopy(acc_check.X)
    Y = copy.deepcopy(acc_check.Y)

    # Pick the test sample
    picked_for_test_X = [X.pop(acc_check.checks - 1)]
    picked_for_test_Y = [Y.pop(acc_check.checks - 1)]

    return X, Y, picked_for_test_X, picked_for_test_Y


def read_unknown_samples():
    unknown_sample = pd.read_csv(acc_check.sfpath, header=None)
    unX = unknown_sample.T.values.tolist()
    num_features = unknown_sample.shape[0]
    return unX, num_features


def handle_feature_importance(clf):
    feature_importance = clf.feature_importances_
    feature_name = pd.read_csv(acc_check.ffpath, header=None).iloc[:, 0].tolist()
    df = pd.DataFrame({"Spectrum": feature_name, "Gini Value": feature_importance})
    df.to_csv("data/gini_values.csv", index=False)
    plt.barh(df["Spectrum"], df["Gini Value"], height=1)
    plt.show()


def split_data():
    return train_test_split(acc_check.X, acc_check.Y, test_size=0.33, random_state=4)


def train_classifier(X_train, y_train):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf


def predict_and_evaluate(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return y_pred, accuracy


def train_and_predict():
    X_train, X_test, y_train, y_test = split_data()
    clf = train_classifier(X_train, y_train)
    y_pred, accuracy = predict_and_evaluate(clf, X_test, y_test)
    return clf, y_pred, y_test


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.show()
    skplt.metrics.plot_confusion_matrix(
        y_test, y_pred, normalize=False, title="Confusion Matrix"
    )
    plt.savefig("data/raw/results/confusion_matrix.png")
    print(cm)


def train_model():
    clf = RandomForestClassifier()
    clf.fit(acc_check.X, acc_check.Y)
    return clf


def validate_sample(unX, num_features):
    if num_features != acc_check.cols_length[acc_check.fnames[0]]:
        UIWindow.identifying_status.setText(
            "Sample and training set doesn't match! ..."
        )
        acc_check.error = True
        return False
    return True


def sample_identifer():
    plt.clf()
    result = []

    if acc_check.sfname == "":
        acc_check.error = True
        UIWindow.identifying_status.setText("No sample file is chosen.")
        return result

    unX, num_features = read_unknown_samples()
    if not validate_sample(unX, num_features):
        return result

    clf = train_model()
    handle_feature_importance(clf)
    clf, y_pred, y_test = train_and_predict()
    plot_confusion_matrix(y_test, y_pred)
    result = clf.predict(unX)

    return result


def compare_parallel(lst1, lst2):
    print(lst1[0], " = ", lst2[0])
    if lst1[0] != lst2[0]:
        with acc_check.wrong_detection.get_lock():  # Acquire the lock before updating
            acc_check.wrong_detection.value += 1
        acc_check.suspicious_data_tracker()


def compare_serial(lst1, lst2):
    if lst1[0] != lst2[0]:
        acc_check.wrong_detection += 1
        acc_check.suspicious_data_tracker()


def call_accuracy_test(check):
    preprocessed_output = accuracy_sequential_checker(acc_check.fnames)
    # Define what needs to be parallelized here
    return machine_learning_accuracytest_serial(preprocessed_output)  # Adjust as needed


def machine_learning_accuracytest_parallel(preprocessed_output):
    features, categories, samples, samples_cats = preprocessed_output
    clf.fit(features, categories)
    result = clf.predict(samples)
    compare_parallel(result, samples_cats)


def machine_learning_accuracytest_serial(preprocessed_output):
    features, categories, samples, samples_cats = preprocessed_output
    clf.fit(features, categories)
    result = clf.predict(samples)
    compare_serial(result, samples_cats)


# Initialize the GUI App
acc_check = AccuracyCheck()
# Preprocess the accuracy_sequential_checker if possible

# Initialize the classifier outside the function if it's being reused
clf = RandomForestClassifier()
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
