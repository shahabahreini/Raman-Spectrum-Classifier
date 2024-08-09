# Version 2.2.2

import copy
import os
import sys
from os.path import relpath

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
from PyQt5.QtCore import Qt

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
    classification_report,
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


class UI(QDialog):
    def __init__(self):
        super(UI, self).__init__()

        # load the ui file
        uic.loadUi("loadui.ui", self)

        # define our widgets
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

        # let's do something
        self.button_browse.clicked.connect(self.browse_trainingfiles)
        self.button_samplebrowse.clicked.connect(self.browse_samplefile)
        self.button_featureList.clicked.connect(self.browse_featureList)
        self.button_run.clicked.connect(self.accuracytest)
        self.button_identify.clicked.connect(self.sample_identify)

        # show the App
        self.show()

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

    def accuracytest(self):
        self.suspicious_data_lst.clear()
        acc_check.reset()
        acc_check.load_trainignfiles()
        if not acc_check.error:
            self.summary.clear()
            self.summary_update()
            self.identifying_status.setText("")
            self.progressbar.setMaximum(len(acc_check.X))
            self.lcdnumber_accuracy.display(round(0.0, 2))
            self.progressbar.setEnabled(True)
            while acc_check.checks <= len(acc_check.X):
                machine_learning_accuracytest()
                QApplication.processEvents()  # Prevents GUI freezes in loops
                self.lcdnumber_wrong.display(acc_check.wrong_detection)
                self.lcdnumber_step.display(acc_check.checks)
                self.progressbar.setValue(acc_check.checks)
                acc_check.checks += 1
            self.update_lcd()
            self.progressbar.setEnabled(False)
            self.progressbar.setValue(0)
        acc_check.error = False

    def sample_identify(self):
        self.identifying_status.setText("Identifying process started ...")
        self.suspicious_data_lst.clear()
        self.identified_result.clear()
        self.summary.clear()
        QApplication.processEvents()  # Prevents GUI freezes in loops
        acc_check.reset()
        acc_check.load_trainignfiles()
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
        # Regards to training part
        self.fnames = []
        self.fpaths = {}
        self.sfpath = ""
        self.ffpath = ""
        self.relative_paths = {}
        self.error = False
        self.wrong_detection = 0
        self.number_of_samples = []
        self.number_of_files = 0
        self.checks = 1
        self.calculation_accuracy = 0
        self.files_data = {}
        self.row_length = {}
        self.cols_length = {}
        self.X = []
        self.Y = []
        self.suspecious_data = []

        # Regards to sample prediction part
        self.sfname = ""

    def reset(self):
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
        self.calculation_accuracy = 100 - self.wrong_detection / self.checks * 100

    def load_trainignfiles(self):
        if len(self.fnames) > 1:
            for fname in self.fnames:
                this_dir = os.path.dirname(__file__)
                self.relative_paths[fname] = relpath(self.fpaths[fname], this_dir)
                train_data = pd.read_csv(self.fpaths[fname], header=None)
                self.cols_length[fname], self.row_length[fname] = train_data.shape
                self.number_of_samples.append(self.row_length[fname])
                self.files_data[fname] = train_data
                num_features = acc_check.cols_length[fname]
                num_samples = acc_check.row_length[fname]
                for i in range(0, num_samples):
                    x = []
                    for j in range(num_features):
                        x.append(self.files_data[fname].iloc[j, i])
                    self.X.append(x)
                    self.Y.append(fname)
        else:
            self.error = True
            UIWindow.identifying_status.setText(
                "At least two training files are required."
            )

    def suspecious_data_tracker(self):
        group = self.Y[self.checks - 1]
        idx = (
            self.checks
            if self.row_length[group] >= self.checks
            else self.checks - self.row_length[group]
        )
        self.suspecious_data.append([group, idx])
        UIWindow.suspicious_data_lst.addItem(f"{group}\t-> {idx}")


def accuracy_sequential_checker(fnames):
    picked_for_test_X = []
    picked_for_test_Y = []
    X = []
    Y = []
    X = copy.deepcopy(acc_check.X)
    Y = copy.deepcopy(acc_check.Y)

    picked_for_test_X.append(X[acc_check.checks - 1])
    picked_for_test_Y.append(Y[acc_check.checks - 1])
    # print(picked_for_test_Y, '----', picked_for_test_X)
    X.pop(acc_check.checks - 1)
    Y.pop(acc_check.checks - 1)

    return X, Y, picked_for_test_X, picked_for_test_Y


def sample_identifer():
    unX = []  # This contains all unknown samples
    plt.clf()

    if acc_check.sfname != "":
        unknown_sample = pd.read_csv(acc_check.sfpath, header=None)

        # ---------------------------- Correlation Matrix ---------------------------- #
        """shahab = pd.DataFrame(acc_check.X).T
        matrix = shahab.corr().round(2)
        sns.heatmap(matrix)
        plt.show()"""
        # ----------------------------------- ---- ----------------------------------- #

        num_features, num_samples = unknown_sample.shape
        for i in range(0, num_samples):
            x = []
            for j in range(num_features):
                x.append(unknown_sample.iloc[j, i])
            unX.append(x)
        if num_features == acc_check.cols_length[acc_check.fnames[0]]:
            clf = RandomForestClassifier()
            clf.fit(acc_check.X, acc_check.Y)

            # ---------------------------- feature importance ---------------------------- #
            """print()
            feature_name = pd.read_csv('data/WT/Raman_Spectrum.csv', header=None)
            feature_name_ = [i for i in range(len(clf.feature_importances_))]
            fig, ax = plt.subplots()
            ax.barh(feature_name_, clf.feature_importances_)
            ax.set_yticklabels(feature_name_)
            plt.show()"""

            feature_importance = clf.feature_importances_
            sorted_idx = sorted(
                range(len(feature_importance)), key=lambda k: feature_importance[k]
            )

            feature_name = pd.read_csv(acc_check.ffpath, header=None)
            feature_name = feature_name.iloc[:, 0].tolist()
            # feature_name_ = [str(i) for i in feature_name]

            # feature_importance_sorted = [feature_importance[i] for i in sorted_idx]

            imp_dict = {"Spectrum": feature_name, "Gini Value": feature_importance}
            df = pd.DataFrame(imp_dict)
            df.to_csv("data/gini_values.csv", index=False)

            plt.barh(feature_name, list(feature_importance), height=1)
            # sns.barplot(x = "Gini Value", y = "Spectrum", data = df)
            # plt.xlabel("Spectrum Importance")
            # plt.ylabel("Spectrum")
            plt.show()

            # ------------------------------------ -- ------------------------------------ #

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                acc_check.X, acc_check.Y, test_size=0.33, random_state=4
            )
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy:", accuracy)

            # Create the confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            ConfusionMatrixDisplay(confusion_matrix=cm).plot()
            plt.show()

            skplt.metrics.plot_confusion_matrix(
                y_test, y_pred, normalize=False, title="Confusion Matrix"
            )
            plt.savefig("data/raw/results/confusion_matrix.png")

            print(cm)
            print("\nClassification Report:\n", classification_report(y_test, y_pred))

            result = clf.predict(unX)
        else:
            UIWindow.identifying_status.setText(
                "Sample and training set doesn't match! ..."
            )
            acc_check.error = True
            result = []
    else:
        acc_check.error = True
        UIWindow.identifying_status.setText("No sample file is chosen.")
        result = []
    return result


def compare(lst1, lst2):
    if lst1[0] != lst2[0]:
        acc_check.wrong_detection += 1
        acc_check.suspecious_data_tracker()


def machine_learning_accuracytest():
    features, categories, samples, samples_cats = accuracy_sequential_checker(
        acc_check.fnames
    )
    clf = RandomForestClassifier()
    clf.fit(features, categories)
    result = clf.predict(samples)
    compare(result, samples_cats)


# Initialize the GUI App
acc_check = AccuracyCheck()
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
