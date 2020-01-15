from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut

from ai import *
from dataloader import *

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
    err_function = None

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
        :return: None.
        """
        num_epochs = prompt_num_epochs()

        '''
        The GA will recieve all the features together.
        Then, the random forest on top will use those features and get the 
        accuracy on the dataset using leave one out.        
        '''
        algo = GA('./configs/metabolite_small.config')
        conf, pop = algo.create_session(num_epochs)

        FeatureEngineering.err_function = find_error_metabolite_small

        winner = pop.run(FeatureEngineering.fitness, num_epochs)

    @staticmethod
    def metabolite_large_dataset():
        pass

    @staticmethod
    def metabolite_combined_dataset():
        pass

    @staticmethod
    def fitness(genomes, conf):
        for gid, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, conf)
            error = FeatureEngineering.err_function(net)
            # todo ensure output of auc is under 1.
            genome.fitness = 1 - error


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

    loo = LeaveOneOut()

    for train_index, test_index in loo.split(train_data):
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
