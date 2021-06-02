import utils
from utils_experiments import set_experiment, want_further_experiment
from experiments_classes import Log
from experiments_implementation import execute_experiment
import pandas
import os


utils.welcome()

output_file = utils.get_path("my_experiments_csv_file")
if not os.path.exists(output_file):
    # prepare csv file for experiments output
    header = Log.header()
    pandas.DataFrame(header).to_csv(output_file, index=False)

n = 1
experiments_to_run = [set_experiment(n)]

while want_further_experiment():
    n = n + 1
    experiments_to_run.append(set_experiment(n))

for i in range(0, n):
    log = execute_experiment(i+1, experiments_to_run[i])
    pandas.DataFrame(log).to_csv(output_file, mode='a', index=False, header=False)

utils.print_space()
print("All done!")
print(str(n) + " experiments saved to file: " + output_file)
utils.bye()

