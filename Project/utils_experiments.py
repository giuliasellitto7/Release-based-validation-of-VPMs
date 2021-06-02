from experiments_classes import Dataset, Approach, Validation, Balancing, Classifier, DatasetReleases, ExperimentSetting
import utils
import os
import json


def choose(enumeration):
    utils.print_space()
    print("Please choose " + enumeration.__name__)
    items = list(enumeration)
    for x in items:
        print(str(x.value) + ": " + x.name)
    selection = input("Selection:")
    selection = selection.strip()
    if selection.isnumeric():
        if int(selection) <= len(items):
            chosen = items[int(selection)-1].name
            print("Selected: " + chosen)
            return chosen
    print("Invalid selection!")
    return choose(enumeration)


def set_experiment(i):
    utils.print_space()
    print("Set experiment " + str(i))
    dataset = choose(Dataset)
    approach = choose(Approach)
    validation = choose(Validation)
    if validation == "release_based":
        releases = choose_releases(dataset, approach)
    else:
        releases = DatasetReleases.cross_validation()
    balancing = choose(Balancing)
    classifier = choose(Classifier)
    setting = ExperimentSetting(dataset, approach, validation, balancing, classifier)
    utils.print_space()
    return setting, releases


def choose_releases(dataset, approach):
    utils.print_space()
    n = input("How many releases do you want to use for training?").strip()
    if not n.isnumeric():
        print("Invalid input! Please insert a number")
        return choose_releases(dataset, approach)
    num_training_set_releases = int(n)
    training_set_releases = []

    # select test release
    all_df_dir = utils.get_path("my_" + approach + "_csv_" + dataset)
    all_df_file_names = os.listdir(all_df_dir)
    num_files = len(all_df_file_names)
    start_list_from = num_training_set_releases
    print("Please choose test release")
    for i in range(num_files):
        if i >= start_list_from:
            print(str(i) + ": " + all_df_file_names[i][:-4])
    loop = True
    while loop:
        selection = input("Selection: ")
        if selection.isnumeric():
            test_set_release_index = int(selection)
            test_set_release = all_df_file_names[int(selection)][:-4]
            print("Test release: "+test_set_release)
            loop = False
        else:
            print("Invalid selection!")

    # retrieve training releases
    for i in range(num_training_set_releases):
        training_set_releases.append(all_df_file_names[test_set_release_index-i-1][:-4])
    print("Training releases: "+str(training_set_releases))

    return DatasetReleases(num_training_set_releases, training_set_releases, test_set_release)


def want_further_experiment():
    utils.print_space()
    print("Do you want to set another experiment?")
    print("1: Yes, set another experiment")
    print("2: No, start running the experiments")
    selection = input("Selection:")
    selection = selection.strip()
    if selection == "1":
        return True
    elif selection == "2":
        return False
    else:
        print("Invalid selection!")
        return want_further_experiment()


def generate_all_experiments_settings():
    all_experiments_list = []
    datasets = ["phpmyadmin"]
    approaches = ["metrics", "text"]
    balancing = ["none", "undersampling", "oversampling"]
    classifiers = ["random_forest"]
    # cross-validation
    for d in datasets:
        for a in approaches:
            for b in balancing:
                for c in classifiers:
                    setting = ExperimentSetting(d, a, "cross_validation", b, c)
                    releases = DatasetReleases.cross_validation()
                    all_experiments_list.append((setting, releases))
    # release-based
    for d in datasets:
        for a in approaches:
            for b in balancing:
                for c in classifiers:
                    setting = ExperimentSetting(d, a, "release_based", b, c)
                    all_releases = generate_all_releases(d, a)
                    for releases in all_releases:
                        all_experiments_list.append((setting, releases))
    return all_experiments_list


def generate_all_releases(dataset, approach):
    all_releases = []
    num_training_set_releases = 3
    # get list of all data files
    all_df_dir = utils.get_path("my_" + approach + "_csv_" + dataset)
    all_df_file_names = os.listdir(all_df_dir)
    num_files = len(all_df_file_names)
    start_list_from = num_training_set_releases
    # generate all possible release for testing
    for test_set_release_index in range(start_list_from, num_files):
        test_set_release = all_df_file_names[test_set_release_index][:-4]
        # retrieve training releases
        training_set_releases = []
        for i in range(num_training_set_releases):
            training_set_releases.append(all_df_file_names[test_set_release_index - i - 1][:-4])
        # setting generated
        all_releases.append(DatasetReleases(num_training_set_releases, training_set_releases, test_set_release))
    return all_releases


