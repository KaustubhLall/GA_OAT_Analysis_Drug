import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_metabolite_oat1_small(split=0.8):
    """
    Splits and loads in the small oat1-oat3 metabolite dataset.
    :param split: size of training set split.
    :return: training data, training labels, test data, test labels, names of features
    """
    dat = DataLoaderMetabolite()

    assert 0 < split < 1

    X, Y, header = dat.load_oat1_3_small()

    train_data, train_labels, test_data, test_labels = \
        train_test_split(X, Y, train_size=split, shuffle=True)

    return train_data, train_labels, test_data, test_labels, header


def split_metabolite_oat1_big(split=0.8):
    """
    Splits and loads in the large oat1-oat3 metabolite dataset.
    :param split: size of training set split.
    :return: training data, training labels, test data, test labels, names of features
    """
    dat = DataLoaderMetabolite()

    assert 0 < split < 1

    X, Y, header = dat.load_oat1_3_large()

    train_data, train_labels, test_data, test_labels = \
        train_test_split(X, Y, train_size=split, shuffle=True)

    return train_data, train_labels, test_data, test_labels, header


def split_metabolite_oat1_all(split=0.8):
    """
    Splits and loads in the combined oat1-oat3-oatp metabolite dataset.
    :param split: size of training set split.
    :return: training data, training labels, test data, test labels, names of features
    """
    dat = DataLoaderMetabolite()

    assert 0 < split < 1

    X, Y, header = dat.load_oat1_3_p_combined()

    train_data, train_labels, test_data, test_labels = \
        train_test_split(X, Y, train_size=split, shuffle=True)

    return train_data, train_labels, test_data, test_labels, header


class DataLoaderMetabolite:
    """
    Had methods for loading in the three datasets.

    """
    def __init__(self, scale=False):
        """
        Sets up a data loader object with the specified parameters.
        :param scale: if true, the scipy.StandardScaler will be used to scale the data.
        :type scale: bool
        """
        self.scale = scale

    def load_oat1_3_small(self):
        """
        Loads the for small fold metabolites.
        :return: X, Y, Header (names of features)
        :rtype:
        """
        to_drop = [0, 2, 3, 4, ]
        label_index = 1  # this is from source
        source_df = pd.read_csv('./datasets/metabolites/OAT1OAT3Small.csv')
        return self.process_df_for_dataset_loader(label_index, source_df,
                                                  to_drop)

    def load_oat1_3_large(self):
        """
        Loads the for large fold metabolites.
        :return: X, Y, Header (names of features)
        :rtype:
        """
        to_drop = [0, 2, 3, 4, ]
        label_index = 1  # this is from source

        source_df = pd.read_csv('./datasets/metabolites/OAT1OAT3Big.csv')
        return self.process_df_for_dataset_loader(label_index, source_df,
                                                  to_drop)

    def load_oat1_3_p_combined(self):
        """
        Loads the for all metabolites, including liver.
        :return: X, Y, Header (names of features)
        :rtype:
        """
        to_drop = [0, 1, 3, 4, 5, ]
        label_index = 2  # this is from source

        source_df = pd.read_csv('./datasets/metabolites/OAT1OAT3OATP.csv')
        return self.process_df_for_dataset_loader(label_index, source_df,
                                                  to_drop)

    def process_df_for_dataset_loader(self, label_index, source_df, to_drop):
        """
        Code to process, scale and clean a csv file loaded in as a dataset.
        :param label_index: index where the label is stored.
        :param source_df: dataframe to get for parsing.
        :param to_drop: indices of the columns to drop from source.
        :return: X, Y and feature names (header) of the csv after dropping
        """
        source_df['SLC'] = source_df['SLC'].astype('category').cat.codes
        df = source_df.drop(source_df.columns[to_drop + [label_index]], axis=1)
        X = np.array([np.array(df.iloc[x, :]) for x in range(df.shape[0])])
        Y = np.array(source_df.iloc[:, label_index])
        header = np.array(df.columns)

        if self.scale:
            feature_scaler = StandardScaler()
            X = feature_scaler.transform(X)

        return X, Y, header

# dat = DataLoaderMetabolite()
#
# x, y, h = dat.load_oat1_3_large()
# print(dat.load_oat1_3_large())
