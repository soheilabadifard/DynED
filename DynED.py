from matplotlib import pyplot as plt
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.data.data_stream import DataStream
from skmultiflow.drift_detection import ADWIN
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from itertools import combinations
from scipy.io import arff

import numpy as np
import pandas as pd
import time
import os
import warnings
import random
from tqdm import tqdm

random.seed(101)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

num_cpus = os.cpu_count()
print(f"Number of CPUs: {num_cpus}")


class DataWindow:
    """
    A class used to represent a Data Window.

    ...

    Attributes
    ----------
    window_features : list
        a list of features in the window
    window_label : list
        a list of labels corresponding to the features in the window
    window_size : int
        the size of the window
    class_based_window : dict
        a dictionary where each key is a class, and each value is a list of features belonging to that class

    Methods
    -------
    append(features, label):
        Appends features and their corresponding label to the window.
    get_data(n):
        Returns the last 'n' features and their corresponding labels from the window.
    balanced_data(n):
        Returns a balanced dataset with 'n' instances.
    get_window_len():
        Returns the current length of the window.
    """

    def __init__(self, value_of_classes, win_size=1000):
        self.window_features = []
        self.window_label = []
        self.window_size = win_size
        self.class_based_window = {cls: [] for cls in value_of_classes}

    def append(self, features, label):
        while len(self.window_label) > self.window_size:
            self.window_features.pop(0)
            self.window_label.pop(0)

        for _, value in self.class_based_window.items():
            while len(value) > self.window_size:
                value.pop(0)

        if len(features) != 1:
            for i, item in enumerate(features):
                self.window_features.append(item.reshape((1, features.shape[1])))
                self.window_label.append(label[i].reshape((1, label.shape[1])))
                self.class_based_window[label[i][0]].append(item.reshape((1, features.shape[1])))
        else:
            self.window_features.append(features)
            self.window_label.append(label.reshape(-1, 1))
            self.class_based_window[label[0]].append(features)

    def get_data(self, n):
        if n < len(self.window_features):
            return np.array(self.window_features[-n:]).reshape((n, self.window_features[0].shape[1])), \
                np.squeeze(np.array(self.window_label[-n:]))
        return np.array(self.window_features[-len(self.window_features):]).reshape((len(self.window_features),
                                                                                    self.window_features[0].shape[1])), \
            np.squeeze(np.array(self.window_label[-len(self.window_features):]))

    def balanced_data(self, n):
        balanced_data_x = []
        balanced_data_y = []
        key_len = len(self.class_based_window.keys())
        cnt = n // key_len
        for key, value in self.class_based_window.items():
            if len(value) > 0:
                aux_x = [value[-cnt:]]
                aux_y = [[key] * len(aux_x[0])]
                balanced_data_x += aux_x[0]
                balanced_data_y += aux_y[0]
        data_x_array = np.array(balanced_data_x).reshape((len(balanced_data_x), balanced_data_x[0].shape[1]))
        data_y_array = np.array(balanced_data_y).reshape((len(balanced_data_y), 1))
        x_shuffled, y_shuffled = shuffle(data_x_array, data_y_array, random_state=101)
        return x_shuffled, np.squeeze(y_shuffled)

    def get_window_len(self):
        return len(self.window_label)


class Classifier(HoeffdingTreeClassifier):
    """
    A subclass of HoeffdingTreeClassifier is used for classification tasks.

    ...

    Attributes
    ----------
    aux_accuracy : float
        auxiliary accuracy of the classifier
    y_prediction : array-like
        the prediction made by the classifier
    counter: int
        a counter for the number of predictions made
    accuracy: float
        the accuracy of the classifier
    true_predicted : int
        the number of true predictions made by the classifier
    num_of_class : int
        the number of classes in the classification task
    prediction_list : list
        a list of predictions made by the classifier
    prediction_list_threshold : list
        a list of thresholds for the predictions made by the classifier
    tm: float
        a timestamp indicating when the classifier was initialized

    Methods
    -------
    predict(dx, dy=None):
        Makes a prediction based on the input features and updates the accuracy.
    aux_predict(dx, dy):
        Makes an auxiliary prediction and updates the auxiliary accuracy.
    calc_accuracy(dy):
        Calculates and updates the accuracy of the classifier.
    get_accuracy():
        Returns the current accuracy of the classifier.
    get_aux_accuracy():
        Returns the current auxiliary accuracy of the classifier.
    get_total_prediction():
        Returns all predictions made by the classifier.
    get_count():
        Returns the total number of predictions made by the classifier.
    get_age():
        Returns the timestamp indicating when the classifier was initialized.
    get_aux_state():
        Returns a tuple containing the list of predictions and auxiliary accuracy.
    set_aux_state(state):
        Sets the list of predictions and auxiliary accuracy based on a given state.
    get_state():
        Returns a tuple containing counter, accuracy, true_predicted, and y_prediction.
    set_state(state):
        Sets counter, accuracy, true_predicted, and y_prediction based on a given state.
    get_weight():
        Returns weight which is calculated as true_predicted divided by counter.
        
    """

    def __init__(self, num_of_class, lst_thr):
        super().__init__(split_confidence=0.9, grace_period=50)
        self.aux_accuracy = 0
        self.y_prediction = None
        self.counter = 0
        self.accuracy = 0
        self.true_predicted = 0
        self.num_of_class = num_of_class
        self.prediction_list = []
        self.prediction_list_threshold = lst_thr
        self.tm = time.time()

    def predict(self, dx, dy=None):
        self.counter += 1 if len(dx) == 1 else len(dx)  # x.shape[0]
        self.y_prediction = super().predict(dx)
        self.calc_accuracy(dy)
        return self.y_prediction

    def aux_predict(self, dx, dy):
        y_prediction_aux = super().predict(dx)
        if len(dy) == 1 and y_prediction_aux.shape[0] == 1:  # y.shape[0]
            if y_prediction_aux == dy:
                self.prediction_list.append(1)
            else:
                self.prediction_list.append(0)
        else:
            cnt = (y_prediction_aux == dy).sum()
            self.prediction_list.append((y_prediction_aux & dy))
            self.aux_accuracy = (cnt / len(self.prediction_list[0])) * 100

        while len(self.prediction_list) > 1:
            self.prediction_list.pop(0)

    def calc_accuracy(self, dy):
        if len(dy) == 1 and self.y_prediction.shape[0] == 1:  # y.shape[0]
            if self.y_prediction == dy:
                self.true_predicted += 1
        else:
            self.true_predicted += (self.y_prediction == dy).sum()
        self.accuracy = (self.true_predicted / self.counter) * 100

    def get_accuracy(self):
        return self.accuracy

    def get_aux_accuracy(self):
        return self.aux_accuracy

    def __lt__(self, other):
        return self.accuracy > other.accuracy

    def __repr__(self, **kwargs) -> str:
        return str(self.accuracy)

    def get_total_prediction(self):
        return self.prediction_list

    def get_count(self):
        return self.counter

    def get_age(self):
        return self.tm

    def get_aux_state(self):
        return self.prediction_list, self.aux_accuracy

    def set_aux_state(self, state):
        self.prediction_list = state[0]
        self.aux_accuracy = state[1]

    def get_state(self):
        return self.counter, self.accuracy, self.true_predicted, self.y_prediction

    def set_state(self, state):
        self.counter = state[0]
        self.accuracy = state[1]
        self.true_predicted = state[2]
        self.y_prediction = state[3]

    def get_weight(self):
        return self.true_predicted / self.counter


# Majority Voting Functions

def majority_voting(classifiers, test_data_x, test_data_y):
    """
    Performs majority voting for a list of classifiers on a test dataset.

    Args:
    classifiers (list): A list of classifier objects.
    test_data (): A Bunch object representing the test dataset.

    Returns:
    A list of predicted labels for the test dataset.
    """
    predictions = [clf.predict(test_data_x, test_data_y) for clf in classifiers]
    candidate = -1
    votes = 0

    # Finding majority candidate
    for prediction in predictions:
        if votes == 0:
            candidate = prediction[0]
            votes = 1
        elif prediction[0] == candidate:
            votes += 1
        else:
            votes -= 1
    return candidate


# Data read

def read_data(pth):
    """
    Reads data from an arff file and returns it as a pandas DataFrame.

    Parameters
    ----------
    path: str
        The path to the .arff file.

    Returns
    -------
    data: DataFrame or int
        A pandas Frame containing the data from the .arff file. If the file is not of type .arff, 
        the function prints a message and returns -1.

    """
    data = None
    splt = pth.split('.')
    root = pth
    if splt[-1] == 'arff':
        data = arff.loadarff(root)
        data = pd.DataFrame(data[0])
        data = data.dropna()
        cat_cols = [col for col in data.columns if data[col].dtype == "O"]
        if cat_cols:
            data[cat_cols] = data[cat_cols].apply(lambda x: x.str.decode('utf8'))
            # find unique values of each categorical column
            uniq_vals = [data[col].unique() for col in cat_cols]
            # assign a number to each unique value of each categorical column
            for i in range(len(cat_cols)):
                data[cat_cols[i]] = data[cat_cols[i]].apply(lambda x: np.where(uniq_vals[i] == x)[0][0])

        # change the dtype of the last column of the data frame to int
        data.iloc[:, -1] = data.iloc[:, -1].astype(int)
    else:
        print("Please provide .arff type")
        return -1
    return data


# prediction Validation
def check_true(dy, y_hat):
    """
    Check if the predicted label matches the true label.

    Parameters
    ----------
    dy: int
        The true label.
    y_hat : int
        The predicted label.

    Returns
    -------
    int
        Returns 1 if the predicted label matches the true label; otherwise, returns 0.
    """
    return 1 if (dy == y_hat) else 0


# Window Average Function
def window_average(dx, n):
    """
    Calculates the moving average over a window of size 'n' for a given list.

    Parameters
    ----------
    dx: list
        The list for which the moving average is to be calculated.
    n: int
        The size of the window for the moving average.

    Returns
    -------
    w_avg : list
        A list of the moving averages. If the length of 'dx' is less than 'n', 
        it returns a list with a single element being the average of 'dx'.
    """
    low_index = 0
    high_index = low_index + n
    w_avg = []
    if len(dx) < high_index:
        return [sum(dx) / len(dx)]

    while high_index < len(dx):
        w_avg.append(sum(dx[low_index:high_index]) / n)
        low_index += n
        high_index += n
    return w_avg


def initialize_classifier(num_clss, prd_lst_thr, cnt):
    """
    Initializes a list of Classifier objects.

    Parameters
    ----------
    num_clss : int
        The number of classes for the classification task.
    prd_lst_thr : list
        A list of thresholds for the predictions made by the classifier.
    cnt: int
        The number of Classifier objects to be created.

    Returns
    -------
    cls_list : list
        A list of Classifier objects.
    """
    cls_list = []
    cls_list.extend(Classifier(num_clss, prd_lst_thr) for _ in range(cnt))
    return cls_list


def pre_train_classifier(cls_names, dx, dy, cls_lst, sample_train_size):
    """
    Pre-trains a list of Classifier objects.

    Parameters
    ----------
    cls_names : list
        The list of class names for the classification task.
    dx: array-like
        The input features for training.
    dy: array-like
        The true labels for training.
    cls_lst : list
        The list of Classifier objects to be pre-trained.
    sample_train_size : int
        The size of the training sample for each classifier.

    Returns
    -------
    cls: list
        The list of pre-trained Classifier objects.
    """
    cls = cls_lst
    for aux, classifier in enumerate(cls):
        classifier.partial_fit(dx[aux * sample_train_size:, :], dy[aux * sample_train_size:, 0],
                               classes=cls_names)
        _ = classifier.predict(dx[aux * sample_train_size:, :], dy[aux * sample_train_size:, 0])
    return cls


# Q statistics
def q_measure_updated(predict_queue, lng=None):
    """
    Calculates the Q-measure for a given queue of predictions.

    Parameters
    ----------
    predict_queue : list
        The list of predictions for which the Q-measure is to be calculated.
    lng: int, optional
        The length of the shortest prediction in the queue. If not provided, it is calculated.

    Returns
    -------
    q_matrix : ndarray
        A square matrix where each element (i, j) is the Q-measure between the i-th and j-th prediction in the queue.
    """
    min_length = lng if bool(lng) else len(min(predict_queue, key=len))
    q_matrix = np.zeros((len(predict_queue), len(predict_queue)), dtype=float)

    for i, j in combinations(range(len(predict_queue)), 2):
        both_correct = np.count_nonzero(predict_queue[i] & predict_queue[j])
        both_incorrect = min_length - np.count_nonzero(
            (predict_queue[i] | predict_queue[j]))
        fcorrect_sincorrect = np.count_nonzero(predict_queue[i] & ~predict_queue[j])
        fincorrect_scorrect = np.count_nonzero(~predict_queue[i] & predict_queue[j])

        q_matrix[i, j] = q_matrix[j, i] = 1 - (((both_correct * both_incorrect) -
                                                (fcorrect_sincorrect * fincorrect_scorrect)) /
                                               ((both_correct * both_incorrect) +
                                                (fcorrect_sincorrect * fincorrect_scorrect) + np.finfo(float).eps))
    return q_matrix


def kappa_metric(predict_queue, lng=None):
    """
    Calculates the Cohen's kappa metric for a given queue of predictions.

    Parameters
    ----------
    predict_queue : list
        The list of predictions for which the kappa metric is to be calculated.
    lng : int, optional
        The length of the shortest prediction in the queue. If not provided, it is calculated.

    Returns
    -------
    kappa_matrix : ndarray
        A square matrix where each element (i, j) is the kappa metric between the i-th and j-th prediction in the queue.
    """
    min_length = lng if bool(lng) else len(min(predict_queue, key=len))
    kappa_matrix = np.zeros((len(predict_queue), len(predict_queue)), dtype=float)

    for i, j in combinations(range(len(predict_queue)), 2):
        both_correct = np.count_nonzero(predict_queue[i][-min_length:] & predict_queue[j][-min_length:])
        both_incorrect = min_length - np.count_nonzero(
            (predict_queue[i][-min_length:] | predict_queue[j][-min_length:]))
        fcorrect_sincorrect = np.count_nonzero(predict_queue[i][-min_length:] & ~predict_queue[j][-min_length:])
        fincorrect_scorrect = np.count_nonzero(~predict_queue[i][-min_length:] & predict_queue[j][-min_length:])

        p_observed = (both_correct + both_incorrect) / min_length
        p_expected = ((both_correct + fcorrect_sincorrect) / min_length) * (
                both_correct + fincorrect_scorrect) / min_length + \
                     ((both_incorrect + fincorrect_scorrect) / min_length) * (
                             fcorrect_sincorrect + both_incorrect) / min_length
        if p_expected == 1:
            p_expected = 1 - np.finfo(float).eps
        kappa_matrix[i, j] = kappa_matrix[j, i] = (p_observed - p_expected) / (1 - p_expected)

    return kappa_matrix


def disagreement_measure(predict_queue, lng=None):
    """
    Calculates the disagreement measure for a given queue of predictions.

    Parameters
    ----------
    predict_queue : list
        The list of predictions for which the disagreement measure is to be calculated.
    lng: int, optional
        The length of the shortest prediction in the queue. If not provided, it is calculated.

    Returns
    -------
    disagreement: ndarray
        A square matrix where each element (i, j) is the disagreement measure between the i-th and j-th prediction in the queue.
    """
    min_length = lng if bool(lng) else len(min(predict_queue, key=len))
    disagreement = np.zeros((len(predict_queue), len(predict_queue)), dtype=float)

    for i, j in combinations(range(len(predict_queue)), 2):
        both_correct = np.count_nonzero(predict_queue[i] & predict_queue[j])
        both_incorrect = min_length - np.count_nonzero(
            (predict_queue[i] | predict_queue[j]))
        fcorrect_sincorrect = np.count_nonzero(predict_queue[i] & ~predict_queue[j])
        fincorrect_scorrect = np.count_nonzero(~predict_queue[i] & predict_queue[j])

        disagreement[i, j] = disagreement[j, i] = (fcorrect_sincorrect + fincorrect_scorrect) / (
                both_correct + both_incorrect + fcorrect_sincorrect + fincorrect_scorrect)
    return disagreement


def correlation_coefficient(predict_queue, lng=None):
    """
    Calculates the correlation coefficient for a given queue of predictions.

    Parameters
    ----------
    predict_queue : list
        The list of predictions for which the correlation coefficient is to be calculated.
    lng: int, optional
        The length of the shortest prediction in the queue. If not provided, it is calculated.

    Returns
    -------
    correlation: ndarray
        A square matrix where each element (i, j) is the correlation coefficient between the i-th and j-th prediction in the queue.
    """
    min_length = lng if bool(lng) else len(min(predict_queue, key=len))
    correlation = np.zeros((len(predict_queue), len(predict_queue)), dtype=float)

    for i, j in combinations(range(len(predict_queue)), 2):
        both_correct = np.count_nonzero(predict_queue[i] & predict_queue[j])
        both_incorrect = min_length - np.count_nonzero(
            (predict_queue[i] | predict_queue[j]))
        fcorrect_sincorrect = np.count_nonzero(predict_queue[i] & ~predict_queue[j])
        fincorrect_scorrect = np.count_nonzero(~predict_queue[i] & predict_queue[j])

        correlation[i, j] = correlation[j, i] = (both_correct * both_incorrect - fcorrect_sincorrect * fincorrect_scorrect) / \
                                                np.sqrt((both_correct + fcorrect_sincorrect) * (both_correct + fincorrect_scorrect) *
                                                        (both_incorrect + fcorrect_sincorrect) * (both_incorrect + fincorrect_scorrect))
    return correlation


def double_fault_measure(predict_queue, lng=None):
    """
    Calculates the double fault measure for a given queue of predictions.

    Parameters
    ----------
    predict_queue : list
        The list of predictions for which the double fault measure is to be calculated.
    lng: int, optional
        The length of the shortest prediction in the queue. If not provided, it is calculated.

    Returns
    -------
    double_fault : ndarray
        A square matrix where each element (i, j) is the double fault measure between the i-th and j-th prediction in the queue.
    """
    min_length = lng if bool(lng) else len(min(predict_queue, key=len))
    double_fault = np.zeros((len(predict_queue), len(predict_queue)), dtype=float)

    for i, j in combinations(range(len(predict_queue)), 2):
        both_correct = np.count_nonzero(predict_queue[i] & predict_queue[j])
        both_incorrect = min_length - np.count_nonzero(
            (predict_queue[i] | predict_queue[j]))
        fcorrect_sincorrect = np.count_nonzero(predict_queue[i] & ~predict_queue[j])
        fincorrect_scorrect = np.count_nonzero(~predict_queue[i] & predict_queue[j])

        double_fault[i, j] = double_fault[j, i] = both_incorrect / (
                both_correct + both_incorrect + fcorrect_sincorrect + fincorrect_scorrect)
    return double_fault


# MMR(exact MMR)
def mmr(len_classifiers, diversity_matrix, accuracy_scores, lmd, to_select):
    """
    Calculates the Maximal Marginal Relevance (MMR) scores for a given set of classifiers.

    Parameters
    ----------
    len_classifiers : int
        The number of classifiers.
    diversity_matrix : ndarray
        A square matrix where each element (i, j) is the diversity measure between the i-th and j-th classifier.
    accuracy_scores : list
        A list of accuracy scores for each classifier.
    lmd: float
        The lambda parameter is used in the MMR calculation. It determines the trade-off between accuracy and diversity.
    to_select : int
        The number of classifiers to select.

    Returns
    -------
    s: list
        A list of indices of the selected classifiers.
    score: float
        The total MMR score for the selected classifiers.
    """
    s = []
    score = 0
    while len(s) < to_select:
        if not s:
            s.append(accuracy_scores.index(max(accuracy_scores)))
            score += max(accuracy_scores)
        else:
            classifier_index = [i for i in range(len_classifiers) if i not in s]
            mmr_list = []
            for i in classifier_index:
                auxi = [diversity_matrix[i, j] for j in s]
                mmr_list.append([i, lmd * accuracy_scores[i] - (1 - lmd) * max(auxi)])
            s.append(max(mmr_list, key=lambda i: i[1])[0])
            score += max(mmr_list, key=lambda i: i[1])[1]
        if len(s) == len_classifiers:
            break
    return s, score


def prun2(list_of_classifiers, lmbd, dt_count, cls_num, diversity_type='q'):
    classifier_index = []
    mmr_score = 0
    accuracy_scores = [classifier.get_aux_accuracy() / 100 for classifier in list_of_classifiers]
    predict_queue = np.array([item.get_total_prediction() for item in list_of_classifiers]).reshape(len(list_of_classifiers), -1)

    if diversity_type == 'q':
        q_diversity_matrix = q_measure_updated(predict_queue, dt_count)
        classifier_index, mmr_score = mmr(len(list_of_classifiers), q_diversity_matrix, accuracy_scores, lmbd, cls_num + 1)

    elif diversity_type == 'kappa':
        kappa_diversity_matrix = kappa_metric(predict_queue, dt_count)
        classifier_index, mmr_score = mmr(len(list_of_classifiers), kappa_diversity_matrix, accuracy_scores, lmbd, cls_num + 1)

    elif diversity_type == 'disagreement':
        disagreement_diversity_matrix = disagreement_measure(predict_queue, dt_count)
        classifier_index, mmr_score = mmr(len(list_of_classifiers), disagreement_diversity_matrix, accuracy_scores, lmbd, cls_num + 1)

    elif diversity_type == 'correlation':
        correlation_diversity_matrix = correlation_coefficient(predict_queue, dt_count)
        classifier_index, mmr_score = mmr(len(list_of_classifiers), correlation_diversity_matrix, accuracy_scores, lmbd, cls_num + 1)

    elif diversity_type == 'double_fault':
        double_fault_diversity_matrix = double_fault_measure(predict_queue, dt_count)
        classifier_index, mmr_score = mmr(len(list_of_classifiers), double_fault_diversity_matrix, accuracy_scores, lmbd, cls_num + 1)

    new_list = [list_of_classifiers[i] for i in classifier_index]
    rest = [classifier for classifier in list_of_classifiers if classifier not in new_list]
    return new_list, rest, mmr_score


def add_classifiers(dt_window, cls_num, to_add, cls_thr, classes):
    new_list = []
    dx, dy = dt_window.balanced_data(to_add * 100)
    if len(dx) <= 100:
        cls = Classifier(cls_num, cls_thr)
        cls.partial_fit(dx, dy, classes)
        _ = cls.predict(dx, dy)
        new_list.append(cls)
    else:
        for i in range(to_add):
            cls = Classifier(cls_num, cls_thr)
            cls.partial_fit(dx[i:(i+1)*100, :], dy[i:(i+1)*100], classes)
            _ = cls.predict(dx, dy)
            new_list.append(cls)
    return new_list


def controller(dt_window, pool_thr, lmbd, cls_pool, smp_q, to_select, clustr, measure_type, slctd_pool=None):
    """
    this function controls the classifier's behavior and the pool mechanism
    :return:
    """
    score = 0
    dx, dy = dt_window.get_data(smp_q)
    main_pool = cls_pool if slctd_pool is None else cls_pool + slctd_pool
    main_pool.sort()
    while len(main_pool) > pool_thr:
        main_pool.pop(-1)
    slctd_pool = []
    rest = []
    first_selection = []
    cls_predictions = np.zeros((len(main_pool), smp_q))

    for i, classifier in enumerate(main_pool):
        classifier.aux_predict(dx, dy)
        cls_predictions[i] = classifier.get_total_prediction()[0]

    clustr.fit(cls_predictions)
    labels = clustr.labels_
    unique_labels = set(labels)

    clusters = {label: [] for label in unique_labels}
    for i, label in enumerate(labels):
        clusters[label].append(i)
    # Sort inside each cluster by the accuracy
    for cluster in clusters.values():
        cluster.sort(key=lambda j: main_pool[j].get_aux_accuracy(), reverse=True)

    # From each cluster, select the first 10 and put in a new list
    for cluster in clusters.values():
        first_selection += [main_pool[i] for i in cluster[:10]]
        rest += [main_pool[i] for i in cluster[10:]]

    if len(first_selection) > to_select:
        select_aux, rest_aux, score = prun2(first_selection, lmbd, len(dx), to_select, diversity_type=measure_type)
        rest += rest_aux
        slctd_pool += select_aux
    else:
        slctd_pool = first_selection

    return slctd_pool, rest, score


# Main
@ignore_warnings(category=ConvergenceWarning)
def main(fpth, diversity_type):
    # Read the Data
    df = read_data(fpth)
    stream = DataStream(df, allow_nan=True)

    # Define Hyper Parameters
    number_of_classes = len(stream.target_values)
    class_names = stream.target_values
    number_of_features = stream.n_features
    initial_components = 5
    initial_sample_train = 50
    sliding_window_size = 1000
    prediction_list_size = q_on_sampls = 50
    lambda_param = 0.6
    total_number_classifiers = 500
    cls_add_eachstep = 5
    number_of_selected = 9
    stream_acc = []
    stream_record = []
    mmr_score_selected = []
    stream_true = 0
    cnt = 0
    iaf = 0
    flg = False
    print(f'name: {fpth}, number_of_classes: {number_of_classes}, number_of_features: {number_of_features}')
    print('-------------------------------')

    # Create Classifier Objects
    classifier_list = initialize_classifier(number_of_classes, prediction_list_size, initial_components)
    data_x, data_y = stream.next_sample(initial_sample_train * 5)
    data_y = data_y.reshape(-1, 1)

    # Pre-train the Classifiers with the initial sample size difference
    classifier_list = pre_train_classifier(class_names, data_x, data_y, classifier_list, initial_sample_train)

    window = DataWindow(win_size=sliding_window_size, value_of_classes=class_names)
    window.append(data_x[:, :], data_y[:, :])
    # New Method

    drifter1 = ADWIN(delta=7e-1)
    cluster1 = KMeans(n_clusters=2, random_state=101)

    selected_classifier_list, classifier_list, mm_score = controller(cls_pool=classifier_list,
                                                                     dt_window=window,
                                                                     pool_thr=total_number_classifiers,
                                                                     lmbd=lambda_param,
                                                                     smp_q=q_on_sampls,
                                                                     to_select=number_of_selected,
                                                                     clustr=cluster1,
                                                                     measure_type=diversity_type)
    mmr_score_selected.append(mm_score)
    pbar = tqdm(total=len(df) - (initial_sample_train * 5))
    aux_cnt = 0
    while stream.has_more_samples():
        x, y = stream.next_sample()
        window.append(x, y)
        y_predict = majority_voting(selected_classifier_list, x, y)
        stream_true = stream_true + check_true(y, y_predict)
        stream_acc.append(stream_true / (cnt + 1))
        stream_record.append(check_true(y, y_predict))
        drifter1.add_element(check_true(y, y_predict))
        for k in selected_classifier_list:
            k.partial_fit(x, y, classes=stream.target_values)

        cnt += 1
        aux_cnt += 1

        if drifter1.detected_change():

            classifier_list += add_classifiers(dt_window=window, cls_num=number_of_classes,
                                               cls_thr=prediction_list_size,
                                               to_add=cls_add_eachstep,
                                               classes=stream.target_values)
            flg = True
        if aux_cnt == 100 and len(classifier_list) != 0 or flg:

            if len(stream_acc) > 100 and (stream_acc[-1] - stream_acc[-100]) != 0:
                iaf = (stream_acc[-1] - stream_acc[-100]) / 100
            if iaf >= 0:
                if lambda_param <= 0.9:
                    lambda_param += 0.1
            else:
                if lambda_param >= 0.1:
                    lambda_param -= 0.1

            selected_classifier_list, classifier_list, mm_score = controller(slctd_pool=selected_classifier_list,
                                                                             cls_pool=classifier_list,
                                                                             dt_window=window,
                                                                             pool_thr=total_number_classifiers,
                                                                             lmbd=lambda_param,
                                                                             smp_q=q_on_sampls,
                                                                             to_select=number_of_selected,
                                                                             clustr=cluster1,
                                                                             measure_type=diversity_type)
            aux_cnt = 0
            flg = False
            mmr_score_selected.append(mm_score)
        pbar.update()
    pbar.close()

    ddd_acc2 = window_average(stream_record, 1000)
    final_accuracy = f"Name: {fpth}, Overall Mean Accuracy: {np.mean(ddd_acc2)}"
    print(final_accuracy)
    a = len(df) // 30
    ddd_acc2 = window_average(stream_record, a)
    x = np.linspace(0, len(df), len(ddd_acc2), endpoint=True)
    plt.figure()
    plt.plot(x, ddd_acc2, 'r', label='DynED', marker="*")
    plt.xlabel('Percentage of data', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.show()
    plt.close()


if __name__ == "__main__":
    dataset = "Put the full address and name of the dataset here"
    measure = "double_fault"

    main(dataset, measure)

