import utils
from utils_experiments import generate_all_experiments_settings
from experiments_classes import Log
from experiments_implementation import execute_experiment
import pandas


utils.welcome()

output_file = utils.get_path("my_replication_csv_file")
# prepare csv file for experiments output
pandas.DataFrame(Log.header()).to_csv(output_file, index=False)

experiments_to_run = generate_all_experiments_settings()

for i in range(0, len(experiments_to_run)):
    log = execute_experiment(i+1, experiments_to_run[i])
    pandas.DataFrame(log).to_csv(output_file, mode='a', index=False, header=False)

utils.print_space()
print("All done!")
print(str(len(experiments_to_run)) + " experiments saved to file: " + output_file)
utils.bye()

