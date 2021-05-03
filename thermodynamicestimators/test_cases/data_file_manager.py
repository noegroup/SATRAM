import dill as pickle
import os

def get_filepath(test_problem_name, estimator_type):
    return "../thermodynamicestimators/test_cases/generated_data/{} {}.pkl".format(test_problem_name, estimator_type)


def load_when_available(test_problem_name, estimator_type):
    path = get_filepath(test_problem_name, estimator_type)
    if os.path.isfile(path):
        with open(path, 'rb') as input:
            try:
                return pickle.load(input)
            except:
                pass

    return None


def save_dataset(dataset, test_problem_name, estimator_type):
    filename = get_filepath(test_problem_name, estimator_type)

    if dataset is not None:
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)
