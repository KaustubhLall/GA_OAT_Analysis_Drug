import neat
import visualize
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

inputs = [
]

for e in inputs:
    assert len(e) == 32, e


test_inputs = [
    [446.207, 432.381, 4.31897, -5.17769, 105.423, 474.81, 0.222031971, 59, 0, 8, 3, -2, 4, 656, 0, -2, 33, 0, 0, 8, 7,
     0, 0, 8, 59, 0, 0, 0, 0, 1, 1, 0],
    [424.246, 457.587, 2.50836, -2.44788, 97.8024, 485.291, 0.201533513, 66, 8, 9, 4, -1, 2, 656, 0, -1, 30, 2, 8, 13,
     17, 0.020449396, 0.081797584, 11, 66, 0, 0, 0, 0, 1, 3, 0],
    [435.227, 427.718, 5.02817, -5.79391, 93.4272, 490.087, 0.190633908, 61, 1, 8, 2, -2, 3, 608, 0, -2, 32, 0, 0, 11,
     9, 0, 0, 11, 61, 0, 0, 0, 0, 1, 0, 0],
    [440.16, 404.725, 5.2218, -6.63067, 95.6163, 441.5, 0.216571461, 53, 0, 8, 2, -2, 5, 660, 0, -2, 33, 0, 0, 4, 3, 0,
     0, 7, 53, 0, 0, 0, 0, 1, 0, 0],
    [302.011, 280.171, 3.22154, -3.71515, 49.9466, 299.874, 0.166558621, 31, 0, 6, 1, -1, 1, 370, 0, -1, 19, 0, 0, 7, 3,
     0, 0, 6, 31, 0, 0, 0, 0, 1, 0, 0],
    [179.095, 187.264, 1.96378, -2.31502, 30.6239, 218.055, 0.140441173, 26, 0, 3, 1, 0, 1, 162, 0, 0, 13, 0, 0, 4, 3,
     0, 0, 4, 26, 0, 0, 0, 0, 0, 0, 1],
    [152.016, 138.206, -0.235168, -2.60213, 45.1179, 141.892, 0.317973529, 14, 0, 4, 2, 1, 2, 190, 1, 0, 10, 0, 2, 0, 0,
     0, 0.044328304, 0, 14, 0, 0, 0, 0, 0, 0, 0],
    [385.248, 421.067, 1.52703, -1.91196, 55.8225, 441.722, 0.126374733, 59, 0, 7, 0, 1, 4, 529, 1, 0, 28, 1, 12, 4, 15,
     0.017913924, 0.214967083, 6, 59, 0, 0, 0, 0, 0, 0, 0],
    [454.283, 514.207, 5.27969, -4.98191, 51.6173, 549.866, 0.093872507, 71, 1, 6, 0, 1, 2, 606, 1, 0, 33, 0, 0, 15, 14,
     0, 0, 14, 71, 0, 0, 0, 0, 0, 0, 0],
    [426.217, 434.885, 4.94765, -6.71307, 76.1362, 470.385, 0.161859328, 58, 0, 6, 1, -1, 5, 624, 0, -1, 32, 0, 5, 4, 8,
     0, 0.065671783, 6, 58, 0, 0, 0, 0, 0, 0, 0],
    [356.088, 353.613, 4.01376, -5.27717, 44.4057, 363.416, 0.122189722, 42, 0, 6, 1, -2, 3, 616, 0, -2, 25, 0, 3, 5, 3,
     0, 0.067558894, 4, 42, 0, 0, 0, 0, 1, 0, 0],
    [331.063, 305.577, 1.38647, -3.6079, 80.726, 313.125, 0.257807585, 36, 0, 8, 2, 0, 3, 611, 0, 0, 23, 0, 2, 2, 1, 0,
     0.024775165, 3, 36, 0, 0, 0, 0, 0, 1, 0],
    [396.045, 396.192, 0.461149, -3.3527, 91.7535, 379.303, 0.241900275, 42, 2, 11, 2, -1, 3, 680, 0, -1, 26, 1, 5, 6,
     6, 0.010898767, 0.054493834, 8, 42, 0, 0, 0, 0, 1, 0, 1],
    [378.902, 252.54, 0.772575, -4.55218, 102.975, 281.059, 0.366382148, 28, 1, 10, 4, 0, 2, 571, 0, 0, 20, 0, 1, 1, 2,
     0, 0.009711095, 2, 28, 0, 0, 0, 1, 0, 0, 0],
    [645.142, 639.673, -0.31936, -6.57789, 182.693, 610.805, 0.299102005, 45, 3, 19, 4, -1, 5, 1250, 0, -1, 44, 1, 9, 8,
     10, 0.005473663, 0.049262971, 12, 45, 0, 0, 0, 0, 1, 0, 0],
    [225.086, 206.629, -1.66662, -1.78613, 93.7013, 226.394, 0.413885969, 16, 0, 6, 4, 1, 2, 308, 1, 0, 16, 0, 2, 3, 3,
     0, 0.021344421, 4, 16, 0, 0, 0, 1, 0, 1, 0],
    [246.126, 254.642, 2.58608, -2.25393, 42.181, 288.138, 0.14639166, 36, 2, 5, 1, -1, 2, 494, 0, -1, 18, 0, 5, 4, 7,
     0, 0.118536782, 4, 36, 0, 0, 0, 0, 1, 0, 0],
    [296.964, 210.356, -0.374064, -3.22592, 104.021, 239.275, 0.434734093, 17, 0, 10, 4, 0, 2, 316, 0, 0, 17, 0, 1, 0,
     1, 0, 0.009613443, 1, 17, 0, 0, 0, 1, 0, 0, 0],
    [229.052, 217.447, -1.04958, -2.00012, 71.7411, 225.845, 0.317656357, 15, 2, 6, 3, 0, 2, 331, 0, 0, 15, 0, 7, 1, 4,
     0, 0.097573079, 2, 15, 0, 0, 0, 1, 0, 1, 0],
    [287.078, 227.531, -2.96911, -1.764, 105.329, 274.392, 0.383863232, 19, 1, 8, 4, -2, 2, 354, 0, -2, 19, 0, 0, 4, 4,
     0, 0, 5, 19, 0, 0, 0, 1, 0, 0, 0],
    [180.042, 168.038, 1.1816, -1.65193, 49.4842, 190.315, 0.260012085, 21, 0, 6, 1, -1, 1, 212, 0, -1, 13, 0, 0, 3, 1,
     0, 0, 3, 21, 0, 0, 0, 0, 1, 0, 1],
    [273.063, 211.508, -3.38471, -1.01282, 106.494, 254.85, 0.417869335, 30, 0, 8, 4, -2, 2, 327, 0, -2, 18, 0, 0, 3, 3,
     0, 0, 5, 30, 0, 0, 0, 1, 0, 0, 0],
    [224.08, 254.25, -0.734363, -1.79637, 65.8791, 239.993, 0.274504256, 28, 2, 6, 2, 0, 2, 388, 0, 0, 16, 0, 8, 2, 4,
     0, 0.121434567, 2, 28, 0, 0, 0, 0, 0, 1, 0],
    [421.038, 317.663, 2.15131, -4.66218, 102.915, 353.567, 0.291076373, 41, 1, 10, 4, 0, 3, 740, 0, 0, 27, 0, 1, 2, 3,
     0, 0.009716757, 4, 41, 0, 0, 0, 1, 0, 0, 0],
    [294.949, 212.025, -0.290636, -2.30232, 103.926, 231.327, 0.449260138, 23, 0, 10, 3, -1, 2, 532, 0, -1, 17, 0, 1, 0,
     0, 0, 0.009622231, 1, 23, 0, 0, 0, 1, 0, 0, 0],
    [241.11, 234.92, 3.59308, -3.91814, 36.1633, 270.794, 0.133545426, 33, 0, 3, 2, -1, 2, 292, 0, -1, 18, 0, 0, 3, 2,
     0, 0, 3, 33, 0, 0, 0, 0, 1, 0, 0]
]
test_outputs = [0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                ]


def get_data():
    return inputs, outputs


def sd(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return msd_list(a, b)

    return (a - b) ** 2


def msd_list(a, b):
    assert len(a) == len(b), 'Target output length doesnt match predictions'
    return len(a) / sum([sd(*x) for x in zip(a, b)])


def create_config(conf_file):
    # have everything set to default settings for now, can technically change the config file.
    return neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        conf_file
    )


def fitness(genomes, conf):
    train_input, train_output = get_data()

    for gid, genome in genomes:
        genome.fitness = len(train_input)
        net = neat.nn.FeedForwardNetwork.create(genome, conf)

        for xi, xo in zip(train_input, train_output):
            pred = net.activate(xi)
            genome.fitness -= sd(pred[0], xo)
            # print('fitness of', gid, 'is', sd(pred[0], xo))


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

    for xi, xo in zip(test_inputs, test_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output), end='')
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
