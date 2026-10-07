import pandas as pd
import matplotlib
import json
from sklearn.utils import compute_class_weight
import numpy as np
import seaborn as sns

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout
from keras.src.metrics import Precision, Recall, Accuracy, AUC
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


# My CNN model working with 2D data
class MyModel:
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.model = self.create_model()

    def create_model(self):
        self.model = Sequential()
        self.model.add(Conv1D(16, 2, activation=LeakyReLU(), input_shape=(X_train.shape[1], 1)))
        self.model.add(Conv1D(16, 2, activation=LeakyReLU()))
        self.model.add(Conv1D(16, 2, activation=LeakyReLU()))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(learning_rate=0.04,
                         beta_1=0.9,
                         beta_2=0.999,
                         epsilon=1e-07,
                         amsgrad=False)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy',
                           metrics=[Precision(), Recall(), Accuracy(), AUC()])
        return self.model

    def fit(self, X_train, y_train, epochs=50, batch_size=2, class_weight=None, save_best_model_path='best_model.h5'):
        checkpoint = ModelCheckpoint(save_best_model_path, monitor='precision', mode='max', save_best_only=True,
                                     verbose=1)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=class_weight,
                       callbacks=[checkpoint])
        self.model.load_weights(save_best_model_path)
        # return self.model precision, recall, accuracy
        return self.model

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def save(self, file_path):
        self.model.save(file_path)


# Code Settings
"""
    Uncomment the following lines to display all the columns and rows
"""
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

Data_path = "Data/"
Results_path = "ML_Results/"


def read_csv(file_path):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        data = pd.read_excel(file_path)
    return data


# Data Summary Function
def Data_Summary(data):
    print("Data Summary")
    print("=====================================")
    print("First 5 rows")
    print(data.describe())
    print("=====================================")
    print("Data types")
    print(data.info())
    print("=====================================")
    print("Data count")
    print(data.count())
    print("=====================================")
    print("Missing values")
    print(data.isnull().sum())
    print("=====================================")
    print("Data shape")
    print(data.shape)
    print("=====================================")
    print("Unique values in each column")
    print(data.nunique())
    print("=====================================")

    # describe the last column
    print("Last column description")
    print(data.iloc[:, -1].describe())
    # Visualize the last column
    temp_data = data.drop(data.index[0])
    plt.figure(figsize=(7, 6))
    plt.bar(temp_data.iloc[:, -1].unique(), temp_data.iloc[:, -1].value_counts())
    plt.show(block=True)

    print("---------------------------------------------------------------------------------")


def Visualize_Data(data):
    print("Visualizing Data")
    print("=====================================")
    # delete the first row as it only contains the column names
    new_data = data.drop(data.index[0])

    # print(new_data.iloc[:, 1:-1])
    viz_data = new_data.iloc[:, 1:-1]

    # Visualizing the data, x is Class(last column), y is the rest of the columns
    for column in viz_data.columns:
        plt.figure(figsize=(7, 6))
        plt.bar(new_data.iloc[:, -1], viz_data[column])
        plt.show(block=True)
        wait = input("Press Enter to continue")


def Preprocessing(data, train_data=True, test_size=0.2):
    if train_data:
        print("Splitting the data into training and testing data")
        print("=====================================")
    else:
        print("Preprocessing Testing Data")
        print("=====================================")
    std_scaler = StandardScaler()

    # if data contains Time and Amount columns, scale them
    if 'Time' in data.columns and 'Amount' in data.columns:
        scaled_time = std_scaler.fit_transform(data['Time'].values.reshape(-1, 1))
        scaled_amount = std_scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

        # drop the original Time and Amount column
        data.drop(['Time'], axis=1, inplace=True)
        data.drop(['Amount'], axis=1, inplace=True)

        # insert the scaled Time and scaled Amount column
        data.insert(0, 'Scaled Time', scaled_time)
        data.insert(1, 'Scaled Amount', scaled_amount)

    if train_data:
        X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=test_size,
                                                            random_state=42, stratify=data.iloc[:, -1])

        print("---------------------------------------------------------------------------------")
        return X_train, X_test, y_train, y_test
    else:
        X_train = data.iloc[:, :-1]
        y_train = data.iloc[:, -1]

        print("---------------------------------------------------------------------------------")
        return X_train, y_train


# This outputs a json file with the results of the machine learning models for the REST API
def Machine_Learning_Json_output(X_train, X_test, y_train, y_test, results_file_name):
    print("Machine Learning")
    print("=====================================")

    # Define a dictionary to store results
    results = {}

    # TODO: Check which one is better the ones with the if's or doing all at once in the ned
    models = ["Logistic Regression", "Decision Tree", "Naive Bayes", "KNN", "SVM"]
    for model_name in models:
        if model_name == "Logistic Regression":
            model = lm.LogisticRegression()
            model.fit(X_train, y_train)
            model_score = model.score(X_test, y_test)
            try:
                model_report = classification_report(y_test, model.predict(X_test))
            except:
                model_report = "None"
            results[model_name] = {"accuracy": model_score, "report": model_report}

        elif model_name == "Decision Tree":
            # Decision Tree
            # TODO: Look for other criterion
            Decision_tree = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=None,
                                                   min_samples_leaf=5)
            Decision_tree.fit(X_train, y_train)
            Decision_tree_score = Decision_tree.score(X_test, y_test)
            try:
                Decision_tree_report = classification_report(y_test, Decision_tree.predict(X_test))
            except:
                Decision_tree_report = "None"
            results[model_name] = {"accuracy": Decision_tree_score, "report": Decision_tree_report}

        elif model_name == "Naive Bayes":
            # Naive Bayes
            Naive_Bayes = GaussianNB()
            Naive_Bayes.fit(X_train, y_train)
            Naive_Bayes_score = Naive_Bayes.score(X_test, y_test)
            try:
                Naive_Bayes_report = classification_report(y_test, Naive_Bayes.predict(X_test))
            except:
                Naive_Bayes_report = "None"
            results[model_name] = {"accuracy": Naive_Bayes_score, "report": Naive_Bayes_report}

        elif model_name == "KNN":
            # KNN
            KNN = KNeighborsClassifier(n_neighbors=5)
            KNN.fit(X_train, y_train)
            KNN_score = KNN.score(X_test, y_test)
            try:
                KNN_report = classification_report(y_test, KNN.predict(X_test))
            except:
                KNN_report = "None"
            results[model_name] = {"accuracy": KNN_score, "report": KNN_report}

        elif model_name == "SVM":
            # SVM
            SVM = SVC(kernel='linear')
            SVM.fit(X_train, y_train)
            SVM_score = SVM.score(X_test, y_test)
            try:
                SVM_report = classification_report(y_test, SVM.predict(X_test))
            except:
                SVM_report = "None"
            results[model_name] = {"accuracy": SVM_score, "report": SVM_report}

    # Models with no classification report
    other_models = ["Linear Regression", "SGD", "Ridge", "Lasso", "Elastic Net", "Huber", "RANSAC", "Theil"]
    for model_name in other_models:
        if model_name == "Linear Regression":
            model = lm.LinearRegression()
            model.fit(X_train, y_train)
            model_score = model.score(X_test, y_test)
            results[model_name] = {"accuracy": model_score, "report": "None"}

        elif model_name == "SGD":
            model = lm.SGDRegressor()
            model.fit(X_train, y_train)
            model_score = model.score(X_test, y_test)
            results[model_name] = {"accuracy": model_score, "report": "None"}

        elif model_name == "Ridge":
            model = lm.Ridge()
            model.fit(X_train, y_train)
            model_score = model.score(X_test, y_test)
            results[model_name] = {"accuracy": model_score, "report": "None"}

        elif model_name == "Lasso":
            model = lm.Lasso()
            model.fit(X_train, y_train)
            model_score = model.score(X_test, y_test)
            results[model_name] = {"accuracy": model_score, "report": "None"}

        elif model_name == "Elastic Net":
            model = lm.ElasticNet()
            model.fit(X_train, y_train)
            model_score = model.score(X_test, y_test)
            results[model_name] = {"accuracy": model_score, "report": "None"}

        elif model_name == "Huber":
            model = lm.HuberRegressor()
            model.fit(X_train, y_train)
            model_score = model.score(X_test, y_test)
            results[model_name] = {"accuracy": model_score, "report": "None"}

        elif model_name == "RANSAC":
            model = lm.RANSACRegressor()
            model.fit(X_train, y_train)
            model_score = model.score(X_test, y_test)
            results[model_name] = {"accuracy": model_score, "report": "None"}

        elif model_name == "Theil":
            model = lm.TheilSenRegressor()
            model.fit(X_train, y_train)
            model_score = model.score(X_test, y_test)
            results[model_name] = {"accuracy": model_score, "report": "None"}

    print("Machine Learning Results (Dictionary):")
    print(results)
    print("=====================================")
    with open(results_file_name, "w") as json_file:
        json.dump(results, json_file, indent=4)  # Add indent for readability

    print(f"Machine Learning Results saved to '{results_file_name}'.")
    print("---------------------------------------------------------------------------------")

    return results


# The normal Machine Learning function
def Machine_Learning(X_train, X_test, y_train, y_test):
    print("Machine Learning")
    print("=====================================")

    log_reg = lm.LogisticRegression()
    log_reg.fit(X_train, y_train)
    log_reg_score = log_reg.score(X_test, y_test)
    print("Logistic Regression Testing Accuracy: ", log_reg_score)
    print("Logistic Regression Classification Report: ")
    print(classification_report(y_test, log_reg.predict(X_test)))
    print("=====================================")

    # Decision Tree
    # TODO: look at different types of criterion
    Decision_tree = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=None, min_samples_leaf=5)
    Decision_tree.fit(X_train, y_train)
    Decision_tree_score = Decision_tree.score(X_test, y_test)
    print("Decision Tree Testing Accuracy: ", Decision_tree_score)
    print("Decision Tree Classification Report: ")
    print(classification_report(y_test, Decision_tree.predict(X_test)))
    print("=====================================")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_score = knn.score(X_test, y_test)
    print("KNN Testing Accuracy: ", knn_score)
    print("=====================================")

    print("Naive Bayes")

    nb = GaussianNB()
    nb.fit(X_train, y_train, sample_weight=None)
    nb_score = nb.score(X_test, y_test)
    print("Naive Bayes Testing Accuracy: ", nb_score)
    print("Naive Bayes Classification Report: ")
    print(classification_report(y_test, nb.predict(X_test)))
    print("=====================================")

    lin_reg = lm.LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_reg_score = lin_reg.score(X_test, y_test)
    print("Linear Regression Testing Accuracy: ", lin_reg_score)
    print("=====================================")

    Sgd_reg = lm.SGDRegressor()
    Sgd_reg.fit(X_train, y_train)
    Sgd_reg_score = Sgd_reg.score(X_test, y_test)
    print("SGD Testing Accuracy: ", Sgd_reg_score)
    print("=====================================")

    Ridge_reg = lm.Ridge()
    Ridge_reg.fit(X_train, y_train)
    Ridge_reg_score = Ridge_reg.score(X_test, y_test)
    print("Ridge Testing Accuracy: ", Ridge_reg_score)
    print("=====================================")

    Lasso_reg = lm.Lasso()
    Lasso_reg.fit(X_train, y_train)
    Lasso_reg_score = Lasso_reg.score(X_test, y_test)
    print("Lasso Testing Accuracy: ", Lasso_reg_score)
    print("=====================================")

    Elastic_reg = lm.ElasticNet()
    Elastic_reg.fit(X_train, y_train)
    Elastic_reg_score = Elastic_reg.score(X_test, y_test)
    print("Elastic Testing Accuracy: ", Elastic_reg_score)
    print("=====================================")

    Huber_reg = lm.HuberRegressor()
    Huber_reg.fit(X_train, y_train)
    Huber_reg_score = Huber_reg.score(X_test, y_test)
    print("Huber Testing Accuracy: ", Huber_reg_score)
    print("=====================================")

    Ransac_reg = lm.RANSACRegressor()
    Ransac_reg.fit(X_train, y_train)
    Ransac_reg_score = Ransac_reg.score(X_test, y_test)
    print("Ransac Testing Accuracy: ", Ransac_reg_score)
    print("=====================================")

    SVM = SVC()
    SVM.fit(X_train, y_train)
    SVM_score = SVM.score(X_test, y_test)
    print("SVM Testing Accuracy: ", SVM_score)
    print("=====================================")

    # Theil regression is not working (Low memory)
    # Theil_reg = lm.TheilSenRegressor()
    # Theil_reg.fit(X_train, y_train)
    # Theil_reg_score = Theil_reg.score(X_test, y_test)
    # print("Theil Testing Accuracy: ", Theil_reg_score)

    models = [log_reg, Decision_tree, knn, lin_reg, Sgd_reg, Ridge_reg, Lasso_reg, Elastic_reg, Huber_reg, Ransac_reg,
              SVM, nb]
    scores = [log_reg_score,
              Decision_tree_score,
              knn_score,
              lin_reg_score,
              Sgd_reg_score,
              Ridge_reg_score,
              Lasso_reg_score,
              Elastic_reg_score,
              Huber_reg_score,
              Ransac_reg_score,
              SVM_score,
              nb_score]

    print("---------------------------------------------------------------------------------")

    return models, scores


def KNN_Model(X_train, X_test, y_train, y_test):
    print("KNN Model")
    print("=====================================")

    knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
    knn.fit(X_train, y_train)
    knn_score = knn.score(X_test, y_test)
    print("KNN Testing Accuracy: ", knn_score)
    print("KNN Classification Report: ")
    print(classification_report(y_test, knn.predict(X_test)))
    print("KNN Confusion Matrix: ")
    print(confusion_matrix(y_test, knn.predict(X_test)))

    plot_confusion_matrix(y_test, knn.predict(X_test), classes=knn.classes_, normalize=True, title='Normalized confusion matrix')
    accuracy = accuracy_score(y_test, knn.predict(X_test))
    precision = precision_score(y_test, knn.predict(X_test))
    recall = recall_score(y_test, knn.predict(X_test))
    f1 = f1_score(y_test, knn.predict(X_test))

    print("=====================================")

    print("---------------------------------------------------------------------------------")

    return knn, accuracy, precision, recall, f1


def Naive_Bayes(X_train, X_test, y_train, y_test):
    print("Naive Bayes")
    print("=====================================")

    nb = GaussianNB()
    nb.fit(X_train, y_train, sample_weight=None)
    nb_score = nb.score(X_test, y_test)
    print("Naive Bayes Testing Accuracy: ", nb_score)
    print("Naive Bayes Classification Report: ")
    print(classification_report(y_test, nb.predict(X_test)))
    print("Naive Bayes Confusion Matrix: ")
    print(confusion_matrix(y_test, nb.predict(X_test)))

    plot_confusion_matrix(y_test, nb.predict(X_test), classes=nb.classes_, normalize=True, title='Normalized confusion matrix')
    accuracy = accuracy_score(y_test, nb.predict(X_test))
    precision = precision_score(y_test, nb.predict(X_test))
    recall = recall_score(y_test, nb.predict(X_test))
    f1 = f1_score(y_test, nb.predict(X_test))

    print("=====================================")

    print("---------------------------------------------------------------------------------")

    return nb, accuracy, precision, recall, f1


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title('Normalized confusion matrix' if normalize else 'Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_model_accuracies(model_names, scores):
    plt.figure(figsize=(12, 8))
    plt.bar(model_names, scores, color='blue')
    plt.ylim(0, 1)
    plt.xlabel('Models')
    plt.ylabel('Accuracy Scores')
    plt.title('Comparison of Model Accuracies')
    plt.xticks(rotation=90)
    plt.show()

def plot_performance_metrics(precision, recall, accuracy, f1 = None):
    if f1 is not None:
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
    else:
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
        }

    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red'])
    plt.ylim(0, 1)
    plt.xlabel('Performans Metrikleri')
    plt.ylabel('Skorlar')
    plt.title('Model Performansı')
    plt.show()


if __name__ == '__main__':
    # Data
    # data_file = Data_path + "creditcard_train.csv"
    test_data_file = Data_path + "VerıKumesı_test.xlsx"
    # data = read_csv(data_file)
    test_data = read_csv(test_data_file)
    # Data_Summary(data)
    # Data_Summary(test_data)
    # print the number of 1 and 0 in the last column of the data
    print(test_data.iloc[:, -1].value_counts())

    accuracies = []
    model_names = []
    precisions = []
    recalls = []
    f1s = []

    # Visualize_Data(data)

    # Processing the data
    X_train, X_test, y_train, y_test = Preprocessing(test_data, train_data=True, test_size=0.2)
    # Test_Data_X, Test_Data_y = Preprocessing(data, train_data=False)

    # Machine Learning
    results_filename = Results_path + "machine_learning_results.json"
    knn, knn_accuracy, knn_precision, knn_recall, knn_f1 = KNN_Model(X_train, X_test, y_train, y_test)
    nb, nb_accuracy, nb_precision, nb_recall, nb_f1 = Naive_Bayes(X_train, X_test, y_train, y_test)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    # print(X_train.shape[1])
    Cnn_Model = MyModel(input_shape=(X_train.shape[1],))
    Cnn_Model = Cnn_Model.fit(X_train, y_train, class_weight=class_weights)
    #
    y_pred = Cnn_Model.predict(X_test).astype(int)

    X_train, X_test, y_train, y_test = Preprocessing(test_data, train_data=True, test_size=0.6)
    knn, knn_accuracy, knn_precision, knn_recall, knn_f1 = KNN_Model(X_train, X_test, y_train, y_test)
    nb, nb_accuracy, nb_precision, nb_recall, nb_f1 = Naive_Bayes(X_train, X_test, y_train, y_test)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    # print(X_train.shape[1])
    Cnn_Model = MyModel(input_shape=(X_train.shape[1],))
    Cnn_Model = Cnn_Model.fit(X_train, y_train, class_weight=class_weights)
    #
    y_pred = Cnn_Model.predict(X_test).astype(int)

    X_train, X_test, y_train, y_test = Preprocessing(test_data, train_data=True, test_size=0.7)
    knn, knn_accuracy, knn_precision, knn_recall, knn_f1 = KNN_Model(X_train, X_test, y_train, y_test)
    nb, nb_accuracy, nb_precision, nb_recall, nb_f1 = Naive_Bayes(X_train, X_test, y_train, y_test)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    # print(X_train.shape[1])
    Cnn_Model = MyModel(input_shape=(X_train.shape[1],))
    Cnn_Model = Cnn_Model.fit(X_train, y_train, class_weight=class_weights)
    y_pred = Cnn_Model.predict(X_test).astype(int)

    # models, scores = Machine_Learning(X_train, X_test, y_train, y_test)

    # results = Machine_Learning_Json_output(X_train, X_test, y_train, y_test, results_filename)
#
    # print(classification_report(y_test, y_pred))
#
    # plot_confusion_matrix(y_test, y_pred, classes=np.unique(y_train))
#
    # precision = precision_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    # accuracy = accuracy_score(y_test, y_pred)
    # model_names = [
    #     'Logistic Regression',
    #     'Decision Tree',
    #     'KNN',
    #     'Linear Regression',
    #     'SGD Regression',
    #     'Ridge Regression',
    #     'Lasso Regression',
    #     'Elastic Net Regression',
    #     'Huber Regression',
    #     'RANSAC Regression',
    #     'SVM',
    #     'Naive Bayes',
    #     'CNN'
    # ]
    # scores.append(accuracy)
    # plot_model_accuracies(model_names, scores)
    #
    # plot_performance_metrics(precision, recall, accuracy)
