from copy import deepcopy
from itertools import cycle
from pprint import pprint as pprint
import sys
import argparse
import matplotlib.pyplot as plt
import random
import math


########################################################################
#                              Task 1                                  #
########################################################################


def distance_euclidean(p1, p2):
    '''
    p1: tuple: 1st point
    p2: tuple: 2nd point

    Returns the Euclidean distance b/w the two points.
    '''

    distance = None

    # TODO [task1]:
    # Your function must work for all sized tuples.

    dims = len(p1)
    square_sum = 0
    for i in range(dims):
        square_sum += (p1[i] - p2[i]) ** 2
    distance = math.sqrt(square_sum)

    ########################################
    return distance


def initialization_forgy(data, k):
    '''
    data: list of tuples: the list of data points
    k: int: the number of cluster centroids to return

    Returns a list of tuples, representing the cluster centroids
    '''

    centroids = []

    # TODO [task1]:
    # Initialize the cluster centroids by sampling k unique datapoints from data

    centroids = random.sample(data, k)
    
    ########################################
    assert len(centroids) == k
    return centroids


def kmeans_iteration_one(data, centroids, distance):
    '''
    data: list of tuples: the list of data points
    centroids: list of tuples: the current cluster centroids
    distance: callable: function implementing the distance metric to use

    Returns a list of tuples, representing the new cluster centroids after one iteration of k-means clustering algorithm.
    '''

    new_centroids = []
    cluster = {}

    # TODO [task1]:
    # You must find the new cluster centroids.
    # Perform just 1 iteration (assignment+updation) of k-means algorithm.

    # assigned cluster to each data point (cluster no. = index of centroid)
    for datum in data:
        dist = [distance(centroid, datum) for centroid in centroids]
        min_dist = argmin(dist)
        cluster[datum] = min_dist
    
    # re-calculating centroids
    for k in range(len(centroids)):
        points = [key for key, val in cluster.iteritems() if val == k]
        if (len(points) == 0):
            new_centroid = centroids[k]
        else:
            s = list(map(sum, zip(*points)))
            l = len(points) * 1.
            new_centroid = tuple(x / l for x in s)
        new_centroids.append(new_centroid)
    
    ########################################
    assert len(new_centroids) == len(centroids)
    return new_centroids


def hasconverged(old_centroids, new_centroids, epsilon = 1e-1):
    '''
    old_centroids: list of tuples: The cluster centroids found by the previous iteration
    new_centroids: list of tuples: The cluster centroids found by the current iteration

    Returns true iff no cluster centroid moved more than epsilon distance.
    '''

    converged = False

    # TODO [task1]:
    # Use Euclidean distance to measure centroid displacements.

    disp = list(map(distance_euclidean, old_centroids, new_centroids))
    for d in disp:
        if (d > epsilon):
            converged = False
            break
        else:
            converged = True

    ########################################
    return converged


def iteration_many(data, centroids, distance, maxiter, algorithm, epsilon = 1e-1):
    '''
    maxiter: int: Number of iterations to perform

    Uses the iteration_one function.
    Performs maxiter iterations of the clustering algorithm, and saves the cluster centroids of all iterations.
    Stops if convergence is reached earlier.

    Returns:
    all_centroids: list of (list of tuples): Each element of all_centroids is a list of the cluster centroids found by that iteration.
    '''

    all_centroids = []
    all_centroids.append(centroids)

    # TODO [task1]:
    # Perform iterations by calling the iteration_one function multiple times. Make sure to pass the algorithm argument to iteration_one (already defined).
    # Stop only if convergence is reached, or if max iterations have been exhausted.
    # Save the results of each iteration in all_centroids.
    # Tip: use deepcopy() if you run into weirdness.

    while (True):
        new_centroids = iteration_one(data, centroids, distance, algorithm)
        all_centroids.append(new_centroids)
        if (hasconverged(centroids, new_centroids, epsilon = epsilon) or maxiter == 0):
            break
        maxiter -= 1
        centroids = new_centroids

    ########################################
    return all_centroids


def performance_SSE(data, centroids, distance):
    '''
    data: list of tuples: the list of data points
    centroids: list of tuples: representing the cluster centroids

    Returns: The Sum Squared Error of the clustering represented by centroids, on the data.
    '''

    sse = None

    # TODO [task1]:
    # Calculate the Sum Squared Error of the clustering represented by centroids, on the data.
    # Make sure to use the distance metric provided.

    sse = 0
    for datum in data:
        dist = [distance(centroid, datum) for centroid in centroids]
        min_dist = argmin(dist)
        sse += dist[min_dist] ** 2

    ########################################
    return sse


########################################################################
#                              Task 3                                  #
########################################################################


def initialization_kmeansplusplus(data, distance, k):
    '''
    data: list of tuples: the list of data points
    distance: callable: a function implementing the distance metric to use
    k: int: the number of cluster centroids to return

    Returns a list of tuples, representing the cluster centroids
    '''

    centroids = []

    # TODO [task3]:
    # Use the kmeans++ algorithm to initialize k cluster centroids.
    # Make sure you use the distance function given as parameter.

    # NOTE: Provide extensive comments with your code.

    if (k == len(data)):
        centroids = data
    else:
        for i in range(k):
            init_center = random.choice(data)
            probabilities = []
            for datum in data:
                probabilities.append(distance(datum, init_center) ** 2)
            prob_sum = sum(probabilities) * 1.0
            probabilities = [x / prob_sum for x in probabilities]
            # choose a centroid according to probability
            choice = random.random()
            j, total = 0, probabilities[0]
            while (choice >= total):
                j += 1
                total += probabilities[j]
            centroid = data[j]
            centroids.append(centroid)

    ########################################
    assert len(centroids) == k
    return centroids


########################################################################
#                              Task 4                                  #
########################################################################


def distance_manhattan(p1, p2):
    '''
    p1: tuple: 1st point
    p2: tuple: 2nd point

    Returns the Manhattan distance b/w the two points.
    '''

    # default k-means uses the Euclidean distance.
    # Changing the distant metric leads to variants which can be more/less robust to outliers,
    # and have different cluster densities. Doing this however, can sometimes lead to divergence!

    distance = None

    # TODO [task4]:
    # Your function must work for all sized tuples.

    dims = len(p1)
    abs_sum = 0
    for i in range(dims):
        abs_sum += abs(p1[i] - p2[i])
    distance = abs_sum

    ########################################
    return distance

# calculates median along a particular dimension
def median(nums):
    n = len(nums)
    nums = sorted(list(nums))
    median = 0
    if (n % 2 == 1):
        median = nums[n // 2]
    else:
        median = sum(nums[n // 2 - 1 : n // 2 + 1]) / 2.
    return median


def kmedians_iteration_one(data, centroids, distance):
    '''
    data: list of tuples: the list of data points
    centroids: list of tuples: the current cluster centroids
    distance: callable: function implementing the distance metric to use

    Returns a list of tuples, representing the new cluster centroids after one iteration of k-medians clustering algorithm.
    '''

    new_centroids = []
    cluster = {}

    # TODO [task4]:
    # You must find the new cluster centroids.
    # Perform just 1 iteration (assignment+updation) of k-medians algorithm.

    # assigned cluster to each data point
    for datum in data:
        dist = [distance(centroid, datum) for centroid in centroids]
        min_dist = argmin(dist)
        cluster[datum] = min_dist
    
    # re-calculating centroids
    for k in range(len(centroids)):
        points = [key for key, val in cluster.iteritems() if val == k]
        # calculating median of points in a cluster
        if (len(points) == 0):
            new_centroid = centroids[k]
        else:
            new_centroid = tuple(map(median, zip(*points)))
        new_centroids.append(new_centroid)
        
    ########################################
    assert len(new_centroids) == len(centroids)
    return new_centroids


def performance_L1(data, centroids, distance):
    '''
    data: list of tuples: the list of data points
    centroids: list of tuples: representing the cluster centroids

    Returns: The L1-norm error of the clustering represented by centroids, on the data.
    '''

    l1_error = None

    # TODO [task4]:
    # Calculate the L1-norm error of the clustering represented by centroids, on the data.
    # Make sure to use the distance metric provided.
    
    l1_error = 0
    for datum in data:
        dist = [distance(centroid, datum) for centroid in centroids]
        min_dist = argmin(dist)
        l1_error += abs(dist[min_dist])

    ########################################
    return l1_error


########################################################################
#                       DO NOT EDIT THE FOLLWOING                      #
########################################################################


def argmin(values):
    return min(enumerate(values), key=lambda x: x[1])[0]


def avg(values):
    return float(sum(values))/len(values)


def readfile(filename):
    '''
    File format: Each line contains a comma separated list of real numbers, representing a single point.
    Returns a list of N points, where each point is a d-tuple.
    '''
    data = []
    with open(filename, 'r') as f:
        data = f.readlines()
    data = [tuple(map(float, line.split(','))) for line in data]
    return data


def writefile(filename, centroids):
    '''
    centroids: list of tuples
    Writes the centroids, one per line, into the file.
    '''
    if filename is None:
        return
    with open(filename, 'w') as f:
        for m in centroids:
            f.write(','.join(map(str, m)) + '\n')
    print 'Written centroids to file ' + filename


def iteration_one(data, centroids, distance, algorithm):
    '''
    algorithm: algorithm to use {kmeans, kmedians}

    Uses the kmeans_iteration_one or kmedians_iteration_one function as required.

    Returns a list of tuples, representing the new cluster centroids after one iteration of clustering algorithm.
    '''

    if algorithm == 'kmeans':
        return kmeans_iteration_one(data, centroids, distance)
    elif algorithm == 'kmedians':
        return kmedians_iteration_one(data, centroids, distance)
    else:
        print 'Unavailable algorithm.\n'
        sys.exit(1)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input', type=str, help='Dataset filename')
    parser.add_argument('-a', '--algorithm', dest='algorithm', type=str, help='Algorithm to use - {kmeans, kmedians}. Default: kmeans', default='kmeans')
    parser.add_argument('-i', '--init', '--initialization', dest='init', type=str, default='forgy', help='The initialization algorithm to be used - {forgy, kmeans++}. Default: forgy')
    parser.add_argument('-o', '--output', dest='output', type=str, help='Output filename. If not provided, centroids are not saved.')
    parser.add_argument('-m', '--iter', '--maxiter', dest='maxiter', type=int, default=1000, help='Maximum number of iterations of the algorithm to perform (may stop earlier if convergence is achieved). Default: 1000')
    parser.add_argument('-e', '--eps', '--epsilon', dest='epsilon', type=float, default=1e-3, help='Minimum distance the cluster centroids move b/w two consecutive iterations for the algorithm to continue. Default: 1e-3')
    parser.add_argument('-k', '--k', dest='k', type=int, default=8, help='The number of clusters to use. Default: 8')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=0, help='The RNG seed. Default: 0')
    parser.add_argument('-n', '--numexperiments', dest='numexperiments', type=int, default=1, help='The number of experiments to run. Default: 1')
    parser.add_argument('--outliers',dest='outliers',default=False,action='store_true',help='Flag for visualizing data without outliers. If provided, outliers are not plotted.')
    parser.add_argument('--verbose',dest='verbose',default=False,action='store_true',help='Turn on verbose.')
    _a = parser.parse_args()

    args = {}
    for a in vars(_a):
        args[a] = getattr(_a, a)

    if _a.algorithm.lower() in ['kmeans', 'means', 'k-means']:
        args['algorithm'] = 'kmeans'
        args['dist'] = distance_euclidean
    elif _a.algorithm.lower() in ['kmedians', 'medians', 'k-medians']:
        args['algorithm'] = 'kmedians'
        args['dist'] = distance_manhattan
    else:
        print 'Unavailable algorithm.\n'
        parser.print_help()
        sys.exit(1)

    if _a.init.lower() in ['k++', 'kplusplus', 'kmeans++', 'kmeans', 'kmeansplusplus']:
        args['init'] = initialization_kmeansplusplus
    elif _a.init.lower() in ['forgy', 'frogy']:
        args['init'] = initialization_forgy
    else:
        print 'Unavailable initialization function.\n'
        parser.print_help()
        sys.exit(1)

    print '-'*40 + '\n'
    print 'Arguments:'
    pprint(args)
    print '-'*40 + '\n'
    return args


def visualize_data(data, all_centroids, args):
    print 'Visualizing...'
    centroids = all_centroids[-1]
    k = args['k']
    distance = args['dist']
    clusters = [[] for _ in range(k)]
    for point in data:
        dlist = [distance(point, centroid) for centroid in centroids]
        clusters[argmin(dlist)].append(point)

    # plot each point of each cluster
    colors = cycle('rgbwkcmy')

    for c, points in zip(colors, clusters):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.scatter(x,y, c=c)

    if not args['outliers']:
        # plot each cluster centroid
        colors = cycle('krrkgkgr')
        colors = cycle('rgbkkcmy')

        for c, clusterindex in zip(colors, range(k)):
            x = [iteration[clusterindex][0] for iteration in all_centroids]
            y = [iteration[clusterindex][1] for iteration in all_centroids]
            plt.plot(x,y, '-x', c=c, linewidth='1', mew=15, ms=2)
    plt.show()


def visualize_performance(data, all_centroids, distance):
    if distance == distance_euclidean:
        errors = [performance_SSE(data, centroids, distance) for centroids in all_centroids]
        ylabel = 'Sum Squared Error'
    else:
        errors = [performance_L1(data, centroids, distance) for centroids in all_centroids]
        ylabel = 'L1-norm Error'
    plt.plot(range(len(all_centroids)), errors)
    plt.title('Performance plot')
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.show()


if __name__ == '__main__':

    args = parse()
    # Read data
    data = readfile(args['input'])
    print 'Number of points in input data: {}\n'.format(len(data))
    verbose = args['verbose']

    totalerror = 0
    totaliter = 0

    for experiment in range(args['numexperiments']):
        print 'Experiment: {}'.format(experiment+1)
        random.seed(args['seed'] + experiment)
        print 'Seed: {}'.format(args['seed'] + experiment)

        # Initialize centroids
        centroids = []
        if args['init'] == initialization_forgy:
            centroids = args['init'](data, args['k'])  # Forgy doesn't need distance metric
        else:
            centroids = args['init'](data, args['dist'], args['k'])

        if verbose:
            print 'centroids initialized to:'
            print centroids
            print ''

        # Run clustering algorithm
        all_centroids = iteration_many(data, centroids, args['dist'], args['maxiter'], args['algorithm'], args['epsilon'])

        if args['dist'] == distance_euclidean:
            error = performance_SSE(data, all_centroids[-1], args['dist'])
            error_str = 'Sum Squared Error'
        else:
            error = performance_L1(data, all_centroids[-1], args['dist'])
            error_str = 'L1-norm Error'
        totalerror += error
        totaliter += len(all_centroids)-1
        print '{}: {}'.format(error_str, error)
        print 'Number of iterations till termination: {}'.format(len(all_centroids)-1)
        print 'Convergence achieved: {}'.format(hasconverged(all_centroids[-1], all_centroids[-2]))

        if verbose:
            print '\nFinal centroids:'
            print all_centroids[-1]
            print ''

    print '\n\nAverage error: {}'.format(float(totalerror)/args['numexperiments'])
    print 'Average number of iterations: {}'.format(float(totaliter)/args['numexperiments'])

    if args['numexperiments'] == 1:
        # save the result
        if 'output' in args and args['output'] is not None:
            writefile(args['output'], all_centroids[-1])

        # If the data is 2-d and small, visualize it.
        if len(data) < 5000 and len(data[0]) == 2:
            if args['outliers']:
                visualize_data(data[2:], all_centroids, args)
            else:
                visualize_data(data, all_centroids, args)

        visualize_performance(data, all_centroids, args['dist'])
