from datetime import datetime
import gc
import json
import numpy
from numpy import unique
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.metrics import f1_score, matthews_corrcoef
from statistics import mean
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import os
import pandas
import utils
from experiments_classes import Log, Performance, DatasetReleases, ExperimentSetting, BagOfWordsExecTime
from time import time


save_vocabulary = True
save_frequency_vectors = True


def execute_experiment(n, setting):
    experiment_setting = setting[0]
    releases = setting[1]
    utils.print_space()
    print("Experiment " + str(n))
    print(experiment_setting)
    print(releases)

    if experiment_setting.validation == "release_based":
        X_train, y_train, X_test, y_test, bow_exec_time = get_release_based_data(experiment_setting.dataset, experiment_setting.approach, releases)
        X_train, y_train = balance(X_train, y_train, experiment_setting.balancing)
        performance = experiment_release_based(X_train, y_train, X_test, y_test, experiment_setting.classifier)
    else:  # cross-validation
        X, y, bow_exec_time = get_cross_validation_data(experiment_setting.dataset, experiment_setting.approach)
        X, y = balance(X, y, experiment_setting.balancing)
        performance = experiment_cross_validation(X, y, experiment_setting.classifier)

    print_performance(performance)
    log = Log.build(experiment_setting, releases, bow_exec_time, performance)
    return log


def print_performance(performance):
    print("Performance Summary:")
    print("Fit time: " + str(performance.fit_time) + " sec")
    print("Precision: " + str(performance.precision))
    print("Recall: " + str(performance.recall))
    print("Accuracy: " + str(performance.accuracy))
    print("Inspection rate: " + str(performance.inspection_rate))
    print("F1-score: " + str(performance.f1_score))
    print("MCC: " + str(performance.mcc))


def my_scorer(classifier, X, y):
    y_pred = classifier.predict(X)
    cm = confusion_matrix(y, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    inspection_rate = (tp + fp) / (tp + tn + fp + fn)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)
    return {"my_precision": precision, "my_recall": recall, "my_accuracy": accuracy,
            "my_inspection_rate": inspection_rate, "my_f1_score": f1, "my_mcc": mcc}


def balance(X, y, b):
    print_class_distribution(y)
    if b == "undersampling":
        print("Performing undersampling...")
        # define undersample strategy: the majority class will be undersampled to match the minority
        undersample = RandomUnderSampler(sampling_strategy='majority')
        X, y = undersample.fit_resample(X, y)
        print_class_distribution(y)
        return X, y
    elif b == "oversampling":
        print("Performing oversampling...")
        # define oversample strategy: Synthetic Minority Oversampling Technique
        # the minority class will be oversampled to match the majority
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)
        print_class_distribution(y)
        return X, y
    else:
        print("No data balancing technique applied.")
        return X, y


def experiment_cross_validation(X, y, c):
    if c == "logistic_regression":
        classifier = LogisticRegression(max_iter=10000)
    elif c == "naive_bayes":
        classifier = GaussianNB()
    else:  # default
        classifier = RandomForestClassifier()
    print("Starting experiment")
    print("3-fold cross validation...")
    score = cross_validate(classifier, X, y, cv=3, scoring=my_scorer)
    print("Done.")
    performance = Performance(fit_time=mean(score["fit_time"]), precision=mean(score["test_my_precision"]),
                              recall=mean(score["test_my_recall"]), accuracy=mean(score["test_my_accuracy"]),
                              inspection_rate=mean(score["test_my_inspection_rate"]),
                              f1_score=mean(score["test_my_f1_score"]), mcc=mean(score["test_my_mcc"]))
    return performance


def experiment_release_based(X_train, y_train, X_test, y_test, c):
    if c == "logistic_regression":
        classifier = LogisticRegression(max_iter=10000)
    elif c == "naive_bayes":
        classifier = GaussianNB()
    else:  # default
        classifier = RandomForestClassifier()
    print("Starting experiment")
    print("Training...")
    start = time()
    classifier.fit(X_train, y_train)
    stop = time()
    print("Testing...")
    score = my_scorer(classifier, X_test, y_test)
    print("Done.")
    performance = Performance(fit_time=stop-start, precision=score["my_precision"],
                              recall=score["my_recall"], accuracy=score["my_accuracy"],
                              inspection_rate=score["my_inspection_rate"],
                              f1_score=score["my_f1_score"], mcc=score["my_mcc"])
    return performance


def print_class_distribution(y):
    print("Dataset Summary: (0 is neutral, 1 is vulnerable)")
    classes = unique(y)
    total = len(y)
    for c in classes:
        n_examples = len(y[y == c])
        percent = n_examples / total * 100
        print('Class %d: %d/%d (%.1f%%)' % (c, n_examples, total, percent))


def get_release_based_data(dataset, approach, dataset_releases):
    all_df_dir = utils.get_path("my_" + approach + "_csv_" + dataset)
    all_df_file_names = os.listdir(all_df_dir)

    # retrieve training set
    print("Training set: ")
    print(dataset_releases.training_set_releases)
    train_df = pandas.read_csv(os.path.join(all_df_dir, dataset_releases.training_set_releases[0]+".csv"), index_col=0)
    if dataset_releases.num_training_set_releases > 1:
        for single_df_file in dataset_releases.training_set_releases[1:]:
            single_df = pandas.read_csv(os.path.join(all_df_dir, single_df_file+".csv"), index_col=0)
            train_df = train_df.append(single_df)

    # retrieve test set
    print("Test set: ")
    print(dataset_releases.test_set_release)
    test_df = pandas.read_csv(os.path.join(all_df_dir, dataset_releases.test_set_release+".csv"), index_col=0)

    # data preparation
    train_df.IsVulnerable.replace(('yes', 'no'), (1, 0), inplace=True)
    train_df.dropna(inplace=True)
    test_df.IsVulnerable.replace(('yes', 'no'), (1, 0), inplace=True)
    test_df.dropna(inplace=True)

    if approach == "metrics":
        # datasets ready to use
        X_train = train_df.iloc[:, 0:13].values
        y_train = train_df.iloc[:, 13].values
        X_test = test_df.iloc[:, 0:13].values
        y_test = test_df.iloc[:, 13].values
        bow_exec_time = BagOfWordsExecTime(-1, -1)
    else:  # BAG OF WORDS
        print("Working on training set...")
        y_train = train_df.iloc[:, 1].values
        train_text_tokens = train_df.iloc[:, 0].values
        train_text_tokens = clean(train_text_tokens)
        vocabulary, vocabulary_building_time = build_vocabulary(train_text_tokens)
        X_train, train_frequency_vectors_building_time = build_frequency_vectors(train_text_tokens, vocabulary)

        print("Working on test set...")
        y_test = test_df.iloc[:, 1].values
        test_text_tokens = test_df.iloc[:, 0].values
        test_text_tokens = clean(test_text_tokens)
        X_test, test_frequency_vectors_building_time = build_frequency_vectors(test_text_tokens, vocabulary)

        bow_exec_time = BagOfWordsExecTime(vocabulary_building_time, train_frequency_vectors_building_time+test_frequency_vectors_building_time)

    return X_train, y_train, X_test, y_test, bow_exec_time


def get_cross_validation_data(dataset, approach):
    # retrieve all dataframes
    all_df_dir = utils.get_path("my_" + approach + "_csv_" + dataset)
    all_df_file_names = os.listdir(all_df_dir)
    df = pandas.read_csv(os.path.join(all_df_dir, all_df_file_names[0]), index_col=0)
    for single_df_file in all_df_file_names:
        single_df = pandas.read_csv(os.path.join(all_df_dir, single_df_file), index_col=0)
        df = df.append(single_df)

    # data preparation
    df.IsVulnerable.replace(('yes', 'no'), (1, 0), inplace=True)
    df.dropna(inplace=True)

    if approach == "metrics":
        # dataset ready to use
        X = df.iloc[:, 0:13].values
        y = df.iloc[:, 13].values
        bow_exec_time = BagOfWordsExecTime(-1, -1)
    else:  # BAG OF WORDS
        y = df.iloc[:, 1].values
        text_tokens = df.iloc[:, 0].values
        text_tokens = clean(text_tokens)
        vocabulary, vocabulary_building_time = build_vocabulary(text_tokens)
        X, frequency_vectors_building_time = build_frequency_vectors(text_tokens, vocabulary)
        bow_exec_time = BagOfWordsExecTime(vocabulary_building_time, frequency_vectors_building_time)

    return X, y, bow_exec_time


def build_vocabulary(text_tokens):
    print("Building vocabulary...")
    start = time()
    # vocabulary also stores frequencies
    vocabulary = {}
    for row in text_tokens:
        words = row.split()
        for w in words:
            if w not in vocabulary.keys():
                vocabulary[w] = 1
            else:
                vocabulary[w] += 1
    stop = time()
    vocabulary_building_time = stop - start
    print("Vocabulary building time: " + str(vocabulary_building_time))
    print("Vocabulary contains " + str(len(vocabulary.keys())) + " words")
    # uncomment to use only the 200 most frequent words in the vocabulary
    # most_freq = heapq.nlargest(200, vocabulary, key=vocabulary.get)
    return vocabulary, vocabulary_building_time


def build_frequency_vectors(text_tokens, vocabulary):
    print("Building frequency vectors...")
    start = time()
    all_frequency_vectors = []
    for row in text_tokens:
        splitted = row.split()
        single_row_frequency_vector = []
        for word in vocabulary:
            c = splitted.count(word)
            single_row_frequency_vector.append(c)
        all_frequency_vectors.append(single_row_frequency_vector)
    stop = time()
    frequency_vectors_building_time = stop - start
    print("Frequency vectors building time: " + str(frequency_vectors_building_time))
    all_frequency_vectors = numpy.asarray(all_frequency_vectors)
    return all_frequency_vectors, frequency_vectors_building_time


def clean(text_tokens):
    print("Cleaning text tokens...")
    for i in range(len(text_tokens)):
        text_tokens[i] = clean_tokens_row(text_tokens[i])
    return text_tokens


def clean_tokens_row(tokens_row):
    # only retain tokens (starting with t_)
    cleaned = " "
    splitted = tokens_row.lower().split()
    for t in splitted:
        if t.startswith("t_"):
            cleaned = cleaned + t + " "
    return cleaned

