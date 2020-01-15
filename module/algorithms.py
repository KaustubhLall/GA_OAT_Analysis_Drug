import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, KFold

import visualize
from ai import *
from dataloader import *

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

random_forest_config_parameters = {
    'n_estimators': 100,  # default value
    'criterion'   : 'entropy',
    # 'n_jobs'      : -1,  # multi-processor speedup

    }


class FeatureSelectionGA:
    """
    This class will implement high level API to call the genetic algorithm
    NEAT and run it as a classifier.
    """
    output_features = 6
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

        The GA will receive all the features together and then use leave one
        out to find its accuracy over given number of epochs.
        :return: None.
        """
        num_epochs = prompt_num_epochs()

        dl = DataLoaderMetabolite()
        train_data, train_labels, header = dl.load_oat1_3_small()

        algo = GA('./configs/metabolite_BI.config',
                  checkpoint_prefix='GA_metab_sm_')

        FeatureSelectionGA.acc_function = feature_sel_err_metab_small

        FeatureSelectionGA.run_session(algo, header, num_epochs)

    @staticmethod
    def metabolite_large_dataset():
        """
        Runs the small dataset of OAT1-OAT3.

        The GA will receive all the features together and then use leave one
        out to find its accuracy over given number of epochs.
        :return: None.
        """
        num_epochs = prompt_num_epochs()

        dl = DataLoaderMetabolite()
        train_data, train_labels, header = dl.load_oat1_3_large()

        algo = GA('./configs/metabolite_BI.config',
                  checkpoint_prefix='GA_metab_lg_')

        FeatureSelectionGA.acc_function = feature_sel_err_metab_large

        FeatureSelectionGA.run_session(algo, header, num_epochs)

    @staticmethod
    def metabolite_combined_dataset():
        """
        Runs the small dataset of OAT1-OAT3.

        The GA will receive all the features together and then use leave one
        out to find its accuracy over given number of epochs.
        :return: None.
        """
        num_epochs = prompt_num_epochs()

        dl = DataLoaderMetabolite()
        train_data, train_labels, header = dl.load_oat1_3_p_combined()

        algo = GA('./configs/metabolite_MULTI.config',
                  checkpoint_prefix='GA_metab_cb_')

        FeatureSelectionGA.acc_function = feature_sel_err_metab_comb

        FeatureSelectionGA.run_session(algo, header, num_epochs)

    @staticmethod
    def run_session(algorithm, header, num_epochs):
        """
        Runs a session for any dataset to do feature engineering.
        :param algorithm: GA object to use to create the session.
        :param header: names of input features.
        :param num_epochs: how many epochs to run for.
        :return: None, will open popups using visualization code.
        """
        conf, pop, stats = algorithm.create_session(num_epochs)
        winner = pop.run(FeatureSelectionGA.fitness, num_epochs)
        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))
        visualize.draw_net(conf, winner, True,
                           node_names=create_node_names(
                               header))
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

    @staticmethod
    def run_genome(conf, genome):
        """
        Runs a single genome. This is a wrapper for async calls from fitness
        function.
        :param conf: configuration object.
        :param genome: genome to run.
        :return: None, adjusts genome fitness in place.
        """
        net = neat.nn.FeedForwardNetwork.create(genome, conf)
        acc = FeatureSelectionGA.acc_function(net)
        assert 0 <= acc <= 1, 'Got unexpected accuracy of %s' % acc
        genome.fitness = acc

    @staticmethod
    def fitness(genomes, conf):
        """
        Function calls the appropriate error function to find the fitness of
        a given genome. Fitness in our case is defined by the AUC of the genome.
        :param genomes: list of genomes
        :param conf: configuration object for NEAT
        :return: none
        """
        # spawn half as many processes as there are genomes
        # executor = concurrent.futures.ProcessPoolExecutor(5)
        # futures = [executor.submit(FeatureEngineering.run_genome, genome) for
        #            genome in genomes]
        # concurrent.futures.wait(futures)

        # old code
        for gid, genome in genomes:
            FeatureSelectionGA.run_genome(conf, genome)


class FeatureEngineering:
    """
    This class will implement high level API to call the genetic algorithm 
    NEAT and run it to engineer a set of features.
    """
    output_features = 6
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

        FeatureEngineering.acc_function = feature_eng_err_metab_small

        algo = GA('./configs/metabolite_FE.config',
                  checkpoint_prefix='FE_metab_sm_')

        FeatureEngineering.run_session(algo, header, num_epochs)

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
        train_data, train_labels, header = dl.load_oat1_3_large()

        algo = GA('./configs/metabolite_FE.config',
                  checkpoint_prefix='FE_metab_lg_')

        FeatureEngineering.acc_function = feature_eng_err_metab_large

        FeatureEngineering.run_session(algo, header, num_epochs)

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

        FeatureEngineering.acc_function = feature_eng_err_metab_comb

        algo = GA('./configs/metabolite_FE.config',
                  checkpoint_prefix='FE_metab_cb_')
        FeatureEngineering.run_session(algo, header, num_epochs)

    @staticmethod
    def run_session(algorithm, header, num_epochs):
        """
        Runs a session for any dataset to do feature engineering.
        :param algorithm: GA object to use to create the session.
        :param header: names of input features.
        :param num_epochs: how many epochs to run for.
        :return: None, will open popups using visualization code.
        """
        conf, pop, stats = algorithm.create_session(num_epochs)
        winner = pop.run(FeatureEngineering.fitness, num_epochs)
        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))
        visualize.draw_net(conf, winner, True,
                           node_names=create_node_names(
                               header))
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

    @staticmethod
    def run_genome(conf, genome):
        """
        Runs a single genome. This is a wrapper for async calls from fitness
        function.
        :param conf: configuration object.
        :param genome: genome to run.
        :return: None, adjusts genome fitness in place.
        """
        net = neat.nn.FeedForwardNetwork.create(genome, conf)
        acc = FeatureEngineering.acc_function(net)
        assert 0 <= acc <= 1, 'Got unexpected accuracy of %s' % acc
        genome.fitness = acc

    @staticmethod
    def fitness(genomes, conf):
        """
        Function calls the appropriate error function to find the fitness of
        a given genome. Fitness in our case is defined by the AUC of the genome.
        :param genomes: list of genomes
        :param conf: configuration object for NEAT
        :return: none
        """
        # spawn half as many processes as there are genomes
        # executor = concurrent.futures.ProcessPoolExecutor(5)
        # futures = [executor.submit(FeatureEngineering.run_genome, genome) for
        #            genome in genomes]
        # concurrent.futures.wait(futures)

        # old code
        for gid, genome in genomes:
            FeatureEngineering.run_genome(conf, genome)


def create_node_names(node_labels):
    """
    Takes in a list of labels and creates a dict that is used by the
    visualization module to create names for visualization.
    :param node_labels: strings for the features.
    :return:
    """
    node_list = list(range(-len(node_labels), 0))
    output_labels = ['EF_' + str(x) for x in
                     range(FeatureEngineering.output_features)]
    output_list = list(range(len(output_labels)))

    return dict(zip((list(node_list) + list(output_list)),
                    (list(node_labels) + list(output_labels))))


def fitness(genomes, conf):
    """
    Function calls the appropriate error function to find the fitness of
    a given genome. Fitness in our case is defined by the AUC of the genome.
    :param genomes: list of genomes
    :param conf: configuration object for NEAT
    :return: none
    """
    # spawn half as many processes as there are genomes
    # executor = concurrent.futures.ProcessPoolExecutor(5)
    # futures = [executor.submit(FeatureEngineering.run_genome, genome) for
    #            genome in genomes]
    # concurrent.futures.wait(futures)

    # old code
    for gid, genome in genomes:
        FeatureEngineering.run_genome(conf, genome)


def feature_eng_err_metab_small(net):
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
    train_data, train_labels, header = dl.load_oat1_3_small()

    engineered_features = np.array(list(map(net.activate, train_data)))

    # once we have the activations for the new engineered features, we will
    # test them using leave one out for validation. We will generate as many
    # required folds, and take the simple average of the accuracies

    leave_one_out = LeaveOneOut()

    acc = evaluate_model_using_engineered_features(engineered_features,
                                                   leave_one_out,
                                                   train_data,
                                                   train_labels)

    return acc


def feature_eng_err_metab_large(net):
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
    train_data, train_labels, header = dl.load_oat1_3_large()

    engineered_features = np.array(list(map(net.activate, train_data)))

    # once we have the activations for the new engineered features, we will
    # test them using leave one out for validation. We will generate as many
    # required folds, and take the simple average of the accuracies

    x_validation = KFold(n_splits=10)

    acc = evaluate_model_using_engineered_features(engineered_features,
                                                   x_validation, train_data,
                                                   train_labels)

    return acc


def feature_eng_err_metab_comb(net):
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

    acc = evaluate_model_using_engineered_features(engineered_features,
                                                   x_validation, train_data,
                                                   train_labels)

    return acc


def feature_sel_err_metab_small(net):
    """
    This function takes in a net, runs it on the training input and compares the
    accuracy with the training output.

    :param net: The neural net that will be selecting the features for a
    specific genome.
    :return: error of the genome.
    """
    dl = DataLoaderMetabolite()
    train_data, train_labels, header = dl.load_oat1_3_small()

    # activate the net for the training data

    predictions = np.array(list(map(net.activate, train_data)))

    return score_multi_pred_output(predictions, train_labels)


def feature_sel_err_metab_large(net):
    """
    This function takes in a net, runs it on the training input and compares the
    accuracy with the training output.

    :param net: The neural net that will be selecting the features for a
    specific genome.
    :return: error of the genome.
    """
    dl = DataLoaderMetabolite()
    train_data, train_labels, header = dl.load_oat1_3_large()

    # activate the net for the training data

    predictions = np.array(list(map(net.activate, train_data)))

    return score_multi_pred_output(predictions, train_labels)


def feature_sel_err_metab_comb(net):
    """
    This function takes in a net, runs it on the training input and compares the
    accuracy with the training output.

    :param net: The neural net that will be selecting the features for a
    specific genome.
    :return: error of the genome.
    """
    dl = DataLoaderMetabolite()
    train_data, train_labels, header = dl.load_oat1_3_p_combined()

    # activate the net for the training data

    predictions = np.array(list(map(net.activate, train_data)))

    return score_multi_pred_output(predictions, train_labels)


def evaluate_model_using_engineered_features(engineered_features, evaluator,
                                             train_data,
                                             train_labels):
    """
    Finds the accuracy of a random forest model on a list of folds of the
    training data.

    :param engineered_features: input features (engineered) for the classifier.
    :param evaluator: leave one out / cross validation folds.
    :param train_data: training data (X)
    :param train_labels: training labels (Y)
    :return: accuracy of the model
    """
    correct_count = 0  # count of how many are correct in LOO
    total_count = 0

    for train_index, test_index in evaluator.split(train_data):
        clf = RandomForestClassifier(**random_forest_config_parameters)

        sub_train_data = engineered_features[train_index]
        sub_train_labels = train_labels[train_index]

        sub_test_data = engineered_features[test_index]
        sub_test_labels = train_labels[test_index]

        clf.fit(sub_train_data, sub_train_labels)

        output = clf.predict(sub_test_data)

        correct_count += auc(sub_test_labels, output)
        total_count += 1

    acc = correct_count / total_count
    return acc


def score_multi_pred_output(predictions, train_labels):
    """
    Predictions is a nxk array where the output is true for the kth class if
    it is the max in the column.

    Todo : should probably replace this with softmax accuracy.

    Ex.

    [[0.5, 0.6], [0.5, 0.1]] --> [class 1, class 0] since the respective
    argument indices are highest in the corresponding sublist.

    :param predictions: list of prob distribution over the output classes.
    :param train_labels: actual expected labels.
    :return: corresponding score out of 1.0
    """

    return auc(list(map(np.argmax, predictions)), train_labels)


def auc(real, pred):
    """
    Find the accuracy of given predictions.
    :param real: real values.
    :param pred: predicted labels.
    :return: auc score.
    """
    assert len(real) == len(pred)

    corr, total = 0, 0
    for r, p in zip(real, pred):
        if r == p:
            corr = + 1
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
