import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, KFold

from core.utilities import random_forest_config_parameters, auc
from core.dataloader import DataLoaderMetabolite


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
    score = auc(train_labels, list(map(np.argmax, predictions)))
    return score


def create_node_names(node_labels, num_outputs):
    """
    Takes in a list of labels and creates a dict that is used by the
    visualization module to create names for visualization.
    :param num_outputs: number of output nodes to draw.
    :param node_labels: strings for the features.
    :return:
    """
    node_list = list(range(-len(node_labels), 0))
    output_labels = ['EF_' + str(x) for x in range(num_outputs)]
    output_list = list(range(len(output_labels)))

    return dict(zip((list(node_list) + list(output_list)),
                    (list(node_labels) + list(output_labels))))