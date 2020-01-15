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


class FeatureEngineering:
    """
    This class will implement high level API to call the genetic algorithm 
    NEAT and run it to engineer a set of features.
    """
    output_features = 5

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


        pass

    @staticmethod
    def metabolite_large_dataset():
        pass

    @staticmethod
    def metabolite_combined_dataset():
        pass
