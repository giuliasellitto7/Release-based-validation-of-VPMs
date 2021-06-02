import pandas
import sys


def get_path(key):
    path_config = pandas.read_csv("path_config.csv", index_col=0)
    return path_config.loc[key]["path"]


def welcome():
    print("Welcome! This program is part of the thesis project:")
    print("\"The Impact of Release-based Validation on Software Vulnerability Prediction Models\"")
    print("developed by Giulia Sellitto in April and May 2021")


def bye():
    print("Bye!")
    sys.exit()


def print_space():
    print("--------------------------------")
