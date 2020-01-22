import csv
import math
from itertools import combinations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from core.dataloader import DataLoaderMetabolite
from core.utilities import prompt_num_features, random_forest_config_parameters, \
    decision_tree_config_parameters, auc, os


def eval_classifier(data, labels, evaluator, classifier):
    correct_count = 0
    total_count = 0

    for train_index, test_index in evaluator.split(data):
        sub_train_data = data[train_index]
        sub_train_labels = labels[train_index]

        sub_test_data = data[test_index]
        sub_test_labels = labels[test_index]

        classifier.fit(sub_train_data, sub_train_labels)

        output = classifier.predict(sub_test_data)

        correct_count += auc(sub_test_labels, output)
        total_count += 1

    acc = correct_count / total_count
    return acc


def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n - r)


class BruteForceMetabolite:
    chunk_size = 48

    @staticmethod
    def metabolite_small_dataset():
        """
        Runs the brute-force combination of all features. Will not remove
        highly correlated ones, and logs all relevant information.

        :return: None, makes changes to disk.
        """
        dl = DataLoaderMetabolite()
        data, labels, header = dl.load_oat1_3_small()
        evaluator = LeaveOneOut()
        k = prompt_num_features()

        outfile = open('output' + os.sep + 'BF_metab_raw_%s' % k, 'w')

        BruteForceMetabolite.run_brute_force(data, evaluator, header, labels, outfile, k)

    @staticmethod
    def metabolite_large_dataset():
        """
        Runs the brute-force combination of all features. Will not remove
        highly correlated ones, and logs all relevant information.

        :return: None, makes changes to disk.
        """
        """
        Runs the brute-force combination of all features. Will not remove
        highly correlated ones, and logs all relevant information.

        :return: None, makes changes to disk.
        """
        dl = DataLoaderMetabolite()
        data, labels, header = dl.load_oat1_3_large()
        evaluator = KFold(n_splits=10)
        k = prompt_num_features()

        outfile = open('output' + os.sep + 'BF_metab_raw_%s' % k, 'w')

        BruteForceMetabolite.run_brute_force(data, evaluator, header, labels, outfile, k)

    @staticmethod
    def metabolite_combined_dataset():
        """
        Runs the brute-force combination of all features. Will not remove
        highly correlated ones, and logs all relevant information.

        :return: None, makes changes to disk.
        """
        dl = DataLoaderMetabolite()
        data, labels, header = dl.load_oat1_3_p_combined()
        evaluator = KFold(n_splits=10)
        k = prompt_num_features()

        outfile = open('output' + os.sep + 'BF_metab_cm_raw_%s' % k, 'w')

        BruteForceMetabolite.run_brute_force(data, evaluator, header, labels, outfile)

    @staticmethod
    def run_brute_force(data, evaluator, header, labels, outfile, k):
        total_features = list(range(len(header)))
        total_iterations = nCr(len(total_features), k)
        # we use leave one out to evaluate fitness
        classifiers = [
            RandomForestClassifier(**random_forest_config_parameters),
            DecisionTreeClassifier(**decision_tree_config_parameters)
            ]

        output_csv_header = ['Feature %d' % x for x in range(k)] + [
            'Random Forest', 'Decision Tree']

        # create an output file to write to
        csv_writer = csv.writer(outfile, delimiter=',')
        csv_writer.writerow(output_csv_header)

        # generate all sequences
        counter = 0
        output_chunk = []
        for feature_indices in tqdm(combinations(total_features, k),
                                    total=total_iterations):

            feature_names = [header[i] for i in feature_indices]

            decision_tree_score = eval_classifier(data, labels,
                                                  evaluator, classifiers[0])
            random_forest_score = eval_classifier(data, labels,
                                                  evaluator, classifiers[1])

            counter += 1

            output_row = feature_names + [random_forest_score,
                                          decision_tree_score]

            output_chunk.append(output_row)

            if counter >= BruteForceMetabolite.chunk_size:
                map(csv_writer.writerow, output_chunk)

                counter = 0
                output_chunk.clear()
