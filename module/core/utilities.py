import os

import neat


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

random_forest_config_parameters = {
    'n_estimators': 100,  # default value
    'criterion'   : 'entropy',
    # 'n_jobs'      : -1,  # multi-processor speedup
    }

decision_tree_config_parameters = {
    'criterion'   : 'entropy',
    # 'n_jobs'      : -1,  # multi-processor speedup
    }


# todo make a readme detailing all the specifics used in each algorithm. For
#  example for GA we do not split the dataset, etc. Document everything.


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


# noinspection PyBroadException
def prompt_num_features():
    """
    Prompts user to enter a valid number for the number of epochs to run any
    code.
    :return: number of epochs, less than 100,000.
    """
    while True:
        ans = input('Enter the number of output features to run the selection '
                    'for: ')

        try:
            ans = int(ans)
            assert 0 < ans < 9

        except:
            print('Please supply a valid integer input less than 9.')
            continue

        return ans


class BaseGA:

    def __init__(self, config='./configs/default.config', checkpoint_prefix=''):
        self.conf_filepath = config
        self.checkpoint_name_prefix = './output' + os.sep + checkpoint_prefix\
                                      + \
                                      '.checkpoint'

    def create_session(self, n_epochs):
        """
        Creates a new GA to run from a given configuration file.
        :param n_epochs: number of epochs this population is expected to run.
        :return: config file and population object
        """
        conf_filepath = self.conf_filepath

        # make a config file
        conf = self.create_config(conf_filepath)

        # make a new population
        pop = neat.Population(conf)

        # make statistical reporters
        stats = neat.StatisticsReporter()
        pop.add_reporter(neat.StdOutReporter(True))
        pop.add_reporter(stats)

        # make a checkpointer to save progress every for 10 equally spaced
        # checkpoints
        pop.add_reporter(
            neat.Checkpointer(n_epochs // 10,
                              filename_prefix=self.checkpoint_name_prefix))

        return conf, pop, stats

    def create_config(cself, conf_file):
        return neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            conf_file
            )