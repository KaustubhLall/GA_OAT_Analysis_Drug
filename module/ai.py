import neat


class GA:

    def __init__(self, config='./configs/default.config', checkpoint_prefix=''):
        self.conf_filepath = config
        self.checkpoint_name_prefix = checkpoint_prefix + '.checkpoint'

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
