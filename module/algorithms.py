from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, KFold
from tqdm import tqdm

from ai import *
from dataloader import *
from module import visualize

# noinspection PyBroadException


random_forest_config_parameters = {
    'n_estimators': 100,  # default value
    'criterion'   : 'entropy',
    'n_jobs'      : -1,  # multi-processor speedup

    }


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


class FeatureEngineering:
    """
    This class will implement high level API to call the genetic algorithm 
    NEAT and run it to engineer a set of features.
    """
    output_features = 5
    acc_function = None

    @staticmethod
    def set_output_features(k):
        """
        Tells the class how many output features to manufacture for the RFW
        classifier to use.
        :param k: desired number of features.
        :return: None.
        """
        FeatureEngineering.output_features = k

    @staticmethod
    def metabolite_small_dataset():
        """
        Runs the small dataset of OAT1-OAT3.

        The GA will receive all the features together.
        Then, the random forest on top will use those features and get the
        accuracy on the dataset using leave one out.
        :return: None.
        """
        num_epochs = prompt_num_epochs()

        dl = DataLoaderMetabolite()
        train_data, train_labels, header = dl.load_oat1_3_small()

        algo = GA('./configs/metabolite.config')
        conf, pop, stats = algo.create_session(num_epochs)

        FeatureEngineering.acc_function = find_error_metabolite_small

        winner = pop.run(FeatureEngineering.fitness, num_epochs)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        visualize.draw_net(conf, winner, True,
                           node_names=FeatureEngineering.create_node_names(
                               header))
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

    @staticmethod
    def metabolite_large_dataset():
        """
        Runs the large dataset of OAT1-OAT3.

        The GA will receive all the features together.
        Then, the random forest on top will use those features and get the
        accuracy on the dataset using 10-fold-cross validation.
        :return: None.
        """
        num_epochs = prompt_num_epochs()

        dl = DataLoaderMetabolite()
        train_data, train_labels, header = dl.load_oat1_3_big()

        algo = GA('./configs/metabolite.config')
        conf, pop, stats = algo.create_session(num_epochs)

        FeatureEngineering.acc_function = find_error_metabolite_large

        winner = pop.run(FeatureEngineering.fitness, num_epochs)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        visualize.draw_net(conf, winner, True,
                           node_names=FeatureEngineering.create_node_names(
                               header))
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

    @staticmethod
    def metabolite_combined_dataset():
        """
         Runs the combined dataset of OAT1-OAT3-OATP.

        The GA will receive all the features together.
        Then, the random forest on top will use those features and get the
        accuracy on the dataset using 10-fold-cross validation.
         :return: None.
         """
        num_epochs = prompt_num_epochs()

        dl = DataLoaderMetabolite()
        train_data, train_labels, header = dl.load_oat1_3_p_combined()

        algo = GA('./configs/metabolite.config')
        conf, pop, stats = algo.create_session(num_epochs)

        FeatureEngineering.acc_function = find_error_metabolite_large

        winner = pop.run(FeatureEngineering.fitness, num_epochs)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        visualize.draw_net(conf, winner, True,
                           node_names=FeatureEngineering.create_node_names(
                               header))
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

    @staticmethod
    def fitness(genomes, conf):
        for gid, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, conf)
            acc = FeatureEngineering.acc_function(net)
            assert 0 < acc < 1, 'Got unexpected accuracy of %s' % acc
            genome.fitness = acc

    @staticmethod
    def create_node_names(node_labels):
        node_list = list(range(-len(node_labels), 0))
        output_labels = ['EF_' + str(x) for x in
                         range(FeatureEngineering.output_features)]
        output_list = list(range(len(output_labels)))

        return dict(zip(node_list + output_list, node_labels + output_labels))


def find_error_metabolite_small(net):
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
    correct_count = 0  # count of how many are correct in LOO
    total_count = 0
    dl = DataLoaderMetabolite()
    train_data, train_labels, header = dl.load_oat1_3_small()

    engineered_features = np.array(list(map(net.activate, train_data)))

    # once we have the activations for the new engineered features, we will
    # test them using leave one out for validation. We will generate as many
    # required folds, and take the simple average of the accuracies

    leave_one_out = LeaveOneOut()

    err = evaluate_model_on_folds(engineered_features, leave_one_out,
                                  train_data,
                                  train_labels)

    return err


def find_error_metabolite_large(net):
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

    dl = DataLoaderMetabolite()
    train_data, train_labels, header = dl.load_oat1_3_big()

    engineered_features = np.array(list(map(net.activate, train_data)))

    # once we have the activations for the new engineered features, we will
    # test them using leave one out for validation. We will generate as many
    # required folds, and take the simple average of the accuracies

    x_validation = KFold(n_splits=10)

    err = evaluate_model_on_folds(engineered_features, x_validation, train_data,
                                  train_labels)

    return err


def find_error_metabolite_combined(net):
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

    dl = DataLoaderMetabolite()
    train_data, train_labels, header = dl.load_oat1_3_p_combined()

    engineered_features = np.array(list(map(net.activate, train_data)))

    # once we have the activations for the new engineered features, we will
    # test them using leave one out for validation. We will generate as many
    # required folds, and take the simple average of the accuracies

    x_validation = KFold(n_splits=10)

    err = evaluate_model_on_folds(engineered_features, x_validation, train_data,
                                  train_labels)

    return err


def evaluate_model_on_folds(engineered_features, evaluator, train_data,
                            train_labels):
    correct_count = 0  # count of how many are correct in LOO
    total_count = 0
    for train_index, test_index in tqdm(evaluator.split(train_data)):
        clf = RandomForestClassifier(**random_forest_config_parameters)

        sub_train_data = engineered_features[train_index]
        sub_train_labels = train_labels[train_index]

        sub_test_data = engineered_features[test_index]
        sub_test_labels = train_labels[test_index]

        clf.fit(sub_train_data, sub_train_labels)

        output = clf.predict(sub_test_data)

        correct_count += (output == sub_test_labels)
        total_count += 1
        # correct_count += roc_auc_score(sub_test_labels, output,
        #                                multi_class='ovr')
    err = correct_count / total_count
    return err
