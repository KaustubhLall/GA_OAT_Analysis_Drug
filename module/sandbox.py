import os

import neat
import sklearn.metrics.roc_auc_score as auc
from sklearn.ensemble import RandomForestClassifier

from module import visualize
from .dataloader import *

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

random_forest_config_parameters = {
    'n_estimators': 100,  # default value
    'criterion'   : 'entropy',
    'n_jobs'      : -1,  # multi-processor speedup

    }

train_data, train_labels, test_data, test_labels, header = \
    split_metabolite_oat1_big()


# todo
def get_train_data():
    return None, None


# todo
def get_test_data():
    return None, None


# todo replace this error function
def sd(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return msd_list(a, b)

    return (a - b) ** 2


def msd_list(a, b):
    assert len(a) == len(b), 'Target output length doesnt match predictions'
    return len(a) / sum([sd(*x) for x in zip(a, b)])


def create_config(conf_file):
    # have everything set to default settings for now, can technically change
    # the config file.
    return neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        conf_file
        )


def find_error(net):
    """
    This function takes in a net, runs it on the training input and compares the
    accuracy with the training output.
    To run it on the training input, we take the output of the net, and run
    it through a random forest classifiers, and then check the accuracy of
    the output of the random forest using the features engineered by the GA.

    :param net: The neural net that will be engineering the features for a
    specific genome.
    :return: error of the genome.
    """
    train_data, train_labels = get_train_data()

    engineered_features = list(map(net.activate, train_data))

    clf = RandomForestClassifier(**random_forest_config_parameters)

    # train the model
    clf.fit(engineered_features, train_labels)

    # test the model
    test_data, test_labels = get_test_data()
    test_features_engineered = list(map(net.activate, test_data))

    predictions = clf.predict(test_features_engineered)

    # error is the auc one-versus-rest. To be deemed correct, the label must
    # exactly match
    error = auc(predictions, test_labels, multi_class='ovr')

    return error


def fitness(genomes, conf):
    train_input, train_output = get_train_data()

    for gid, genome in genomes:
        genome.fitness = len(train_input)
        net = neat.nn.FeedForwardNetwork.create(genome, conf)

        error = find_error(net)
        # todo ensure output of auc is under 1.
        genome.fitness = 1 - error


def run(epochs):
    conf_filepath = './configs/default.config'

    # make a config file
    conf = create_config(conf_filepath)

    # make a new population
    pop = neat.Population(conf)

    # make statistical reporters
    stats = neat.StatisticsReporter()
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(stats)

    # make a checkpointer to save progress every 10 epochs
    pop.add_reporter(neat.Checkpointer(10))

    # find the winner
    winner = pop.run(fitness, epochs)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, conf)

    corr, total = 0., 0.

    for xi, xo in zip(get_test_data()):
        output = winner_net.activate(xi)
        print(
            "input {!r}, expected output {!r}, got {!r}".format(xi, xo, output),
            end='')
        if output[0] > 0.6:
            output = 1
        else:
            output = 0

        if abs(xo - output) < 0.05:
            corr += 1
            print(' [correct]')
        else:
            print(' [incorrect]')
        total += 1

    print("Test acc:", corr / total)
    node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    visualize.draw_net(conf, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


run(3000)

# p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
# p.run(fitness, 10)
