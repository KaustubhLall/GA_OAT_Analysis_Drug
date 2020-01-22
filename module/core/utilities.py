from core.ai import *

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

random_forest_config_parameters = {
    'n_estimators': 100,  # default value
    'criterion'   : 'entropy',
    # 'n_jobs'      : -1,  # multi-processor speedup
    }


# todo make a readme detailing all the specifics used in each algorithm. For
#  example for GA we do not split the dataset, etc. Document everything.

class BruteForce:
    pass
re

def auc(real, pred):
    """
    Find the accuracy of given predictions.
    :param real: real values.
    :param pred: predicted labels.
    :return: auc score.
    """
    real = list(real)
    pred = list(pred)
    assert len(real) == len(pred)

    # print('real: {}\npred: {}'.format(real, pred))
    corr, total = 0, 0

    for r, p in zip(real, pred):
        # corr += abs(p - r) ** 2
        corr += p == r
        total += 1

    return corr / total


# noinspection PyBroadException
def prompt_num_epochs():
    """
    Prompts user to enter a valid number for the number of epochs to run any
    code.
    :return: number of epochs, less than 100,000.
    """
    while True:
        ans = input('Enter the number of epochs to run the selection for: ')

        try:
            ans = int(ans)
            assert 0 < ans < 1e6

        except:
            print('Please supply a valid integer input less than 100,000.')
            continue

        return ans
