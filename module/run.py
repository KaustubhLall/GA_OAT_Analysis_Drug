from algorithms import *

while True:
    """Ask for dataset"""
    dataset_selection = input('''
Type in the number of the dataset to load and run: 
    1. Metabolite OAT 1 OAT 3 Small
    2. Metabolite OAT 1 OAT 3 Large
    3. Metabolite OAT 1 OAT 3 OAT P's Combined (Kidney-Liver)
    
    e/q to exit.
    
Response: 
            '''
                              )

    if dataset_selection.lower() in ['e', 'q']:
        exit(1)

    if dataset_selection not in ['1', '2', '3']:
        print('Select a valid option on last prompt.')
        continue

    # if input('You selected option {}. Proceed? (y/n)'.format(
    #         dataset_selection)).lower() not in ['y', 'yes']:
    #     continue

    """ Ask for algorithm"""
    run_selection = input('''
    Type in the number of the dataset to load and run: 
        1. Run brute force using Random forest and Decision tree.
        2. Run genetic algorithm.
        3. Run feature engineering by genetic algorithm.

        e/q to exit.

    Response: ''')

    if run_selection.lower() in ['e', 'q']:
        exit(1)

    if run_selection not in ['1', '2', '3']:
        print('Select a valid option on last prompt.')
        continue

    # if input('You selected option {}. Proceed? (y/n)'.format(
    #         run_selection)).lower() not in ['y', 'yes']:
    #     continue

    """ Run small dataset"""
    if dataset_selection == '1':
        # run the small metabolite dataset. Error used here is going to be
        # hold-one- out

        # ask if we want to run GA, GA to feature engineer, or simple brute
        # force

        """Brute force"""
        if run_selection == '1':
            pass

        """Genetic algorithm"""
        if run_selection == '2':
            pass

        """Feature engineering using GA"""
        if run_selection == '3':
            print('Now running the small metabolite dataset for feature '
                  'engineering.')
            FeatureEngineering.metabolite_small_dataset()
            pass

    """ Run big dataset"""
    if dataset_selection == '2':
        # run the small metabolite dataset. Error used here is going to be
        # 10-fold cross validation
        pass

    """ Run combined dataset"""
    if dataset_selection == '3':
        # run the small metabolite dataset. Error used here is going to be
        # 10 fold cross validation
        pass

    action_exit = input(
        'Finished running. Select another option to continue, or e/q to exit.')

    if action_exit.lower() in ['e', 'q']:
        break
