import os
import dill as pickle


def get_filepath(test_problem_name):
    return "../thermodynamicestimators/test_cases/generated_data/{}.pkl".format(test_problem_name)


def load_when_available(test_problem_name):
    path = get_filepath(test_problem_name)
    if os.path.isfile(path):
        with open(path, 'rb') as input:
            try:
                return pickle.load(input)
            except:
                pass

    return None


def save_dataset(dataset, test_problem_name):
    filename = get_filepath(test_problem_name)

    if dataset is not None:
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)
