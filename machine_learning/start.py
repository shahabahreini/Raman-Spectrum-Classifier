import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from setting import modify_config
import multiprocessing
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Move worker function to the top level
def worker(args):
    instance, train_test_indices, clf = args
    return instance.worker(train_test_indices, clf)


class ClassifierValidation:
    def __init__(self, settings):
        self.settings = settings
        self.load_data()

    def load_data(self):
        type1_data = pd.read_csv(self.settings["csv1_path"], header=None).T
        type2_data = pd.read_csv(self.settings["csv2_path"], header=None).T
        spectrum = pd.read_csv(updated_settings["csv_spectrum_path"], header=None)
        spectrum = spectrum.iloc[:, 0].tolist()

        self.data = pd.concat([type1_data, type2_data], axis=0)
        self.labels = [self.settings["csv1_path"]] * type1_data.shape[0] + [
            self.settings["csv2_path"]
        ] * type2_data.shape[0]
        self.data = StandardScaler().fit_transform(self.data)
        self.spectrum = spectrum

    def clf_maker(self, classifier="random_forest"):
        if classifier == "random_forest":
            clf = RandomForestClassifier()
        elif classifier == "svm":
            clf = SVC()
        elif classifier == "svm_linear":
            clf = SVC(kernel="linear")
        elif classifier == "lightgbm":
            clf = LGBMClassifier()
        else:
            raise ValueError(
                "Invalid classifier. Choose from: random_forest, svm, lightgbm"
            )
        return clf

    def validate(self, classifier="random_forest", method="single"):
        clf = self.clf_maker(classifier)
        if method == "single":
            self.single_process_validation(clf)
        elif method == "multi":
            self.multi_process_validation(clf)
        else:
            raise ValueError("Invalid method. Choose from: single, multi")
        clf.fit(self.data, self.labels)
        return clf

    def multi_process_validation(self, clf):
        loo = LeaveOneOut()
        train_test_indices = list(loo.split(self.data))
        with multiprocessing.Pool() as pool:
            accuracies = pool.map(
                worker, [(self, indices, clf) for indices in train_test_indices]
            )
        print(np.mean(accuracies))
        return

    def single_process_validation(self, clf):
        loo = LeaveOneOut()
        accuracies = []
        for train_index, test_index in loo.split(self.data):
            X_train, X_test = self.data[train_index], self.data[test_index]
            y_train, y_test = (
                np.array(self.labels)[train_index],
                np.array(self.labels)[test_index],
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
        print(np.mean(accuracies))
        return

    def worker(self, train_test_indices, clf):
        train_index, test_index = train_test_indices
        X_train, X_test = self.data[train_index], self.data[test_index]
        y_train, y_test = (
            np.array(self.labels)[train_index],
            np.array(self.labels)[test_index],
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def train_test_split_validation(self, classifier="random_forest", test_size=0.2):
        clf = self.clf_maker(classifier)
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=42
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
        self.plot_confusion_matrix(y_test, y_pred)
        return accuracy, clf  # Return the fitted classifier

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt="d")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.show()

    def save_feature_importances_to_csv(self, feature_importances):
        feature_importance_df = pd.DataFrame(
            {"spectrum": self.spectrum, "importance": feature_importances}
        )
        feature_importance_df.to_csv("feature_importances.csv", index=False)
        print("Feature importances saved to feature_importances.csv")

    def show_feature_importances(self, clf):
        if hasattr(clf, "feature_importances_"):
            feature_importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            feature_importances = np.abs(clf.coef_[0])
        else:
            raise ValueError(
                "This classifier does not have feature_importances_ or coef_ attribute."
            )
        # feature_indices = np.argsort(feature_importances)  # top 10 features
        feature_indices = np.arange(len(feature_importances))
        self.plot_feature_importances(feature_importances, feature_indices)
        return feature_importances

    def plot_feature_importances(self, feature_importances, feature_indices):
        plt.figure(figsize=(12, 6))
        plt.bar(
            range(len(feature_importances)),
            feature_importances[feature_indices],
            align="center",
            color="blue",
        )

        sorted_spectrum = [self.spectrum[i] for i in feature_indices]
        plt.xticks(
            range(len(feature_importances)),
            sorted_spectrum,
            rotation=60,
            ha="right",
            fontsize=8,
        )
        plt.xlabel("Feature Index")
        plt.ylabel("Feature Importance")
        plt.title("Top 10 Important Features")

        # Use colorblind-friendly palette and set background color
        sns.set_palette("colorblind")
        plt.gca().set_facecolor("white")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    new_settings = {
        "csv1_path": "Rab2.csv",
        "csv2_path": "Wt3.csv",
        "csv_spectrum_path": "Spectrum_650.csv",
    }
    updated_settings = modify_config("config.ini", new_settings)
    model = ClassifierValidation(updated_settings)
    classifier = "svm_linear"  # lightgbm|svm|random_forest|svm_linear
    method = "multi"
    fitted_clf = model.validate(classifier, method)
    # _, fitted_clf = model.train_test_split_validation(classifier)  # Get the fitted classifier
    feature_importances = model.show_feature_importances(fitted_clf)
    model.save_feature_importances_to_csv(feature_importances)
