"""

Created in May 2022
@author: Max Watson
IncremenetalGMM
"""

import numpy as np
import math
import traceback
import matplotlib.pyplot as plot
import random
import matplotlib as mpl


class IncrementalGMM(object):
    """
        Reference IGMM Paper*
    """

    def __init__(self, x_val, y_val, z_val = [],vector_length = 2, starting_data = [], **kwargs):
        #TODO add better comments and clearer variable names
        # accumulators , and standard base initialization

        # Start by initializing IGMM data structures and variables
        self.feature_dimension = vector_length # Feature set dimension

        self.mus = {0:self.mu_init(x_val, y_val, z_val)}  # Initialize mus dictionary for IGMM
        self.sigmaInverse = self.sigma_init(starting_data)  # Initialize sigmaInverse to create precision matrix dict
        self.alphas = {0:(self.sigmaInverse * np.identity(self.feature_dimension))}  # Initialize precision mat dict

        self.ages = {0:1} # Represents the number of points in each component
        self.accumulators = {0:1.}  # dict for summed soft probabilities
        self.covariance_determinants = {0:(np.linalg.det(self.alphas[0]) ** -1)} # dict to track cov mat dets
        self.component_probabilities = {0: 1.}

        self.corresponding_data = {0: np.array([self.mus[0]])} # Initialize corresponding_data of cluster 0 to first mu
        self.unclassified_points = [] # list to keep track of unclassified points

        self.K = 0  # Set the current number of components to 1
        self.KMax = 10  # Set the maximum number of components to 10
        self.mahalanobisMax = 6  # Set the threshold for the mahalanobis distance to 6 (85% of chi^2, 4 DoF)
        # TODO allow for initialization of data to fit
        self.data = np.array([[]])  # Begin with no data to fit
        self.min_data = 500  # minimum number of points needed to collect before fitting
        self.fig_num = 2  # Start fig_num at 2 to offset figure for generated data

        self.state = 'COLLECTING' # READY_TO_FIT, 'FITTED', 'EXPIRED', cluster-wise states regarding the merge/split

        # These are currently unused
        # self.printing = False  # Variable to track if printing in functions, not currently used
        # self.converged = False  # when model converges or fitted, it is True
        # self.model_param = None  # for persistence

    def sigma_init(self, starting_data):
        # Function to determine starting sigma based on whether starting data was provided
        if len(starting_data) == 0:
            return 1
        else:
            return np.std(starting_data) ** 2 ** -1

    # Initialize mu to a ndarray of length 2 or 3, depending on whether a z value was provided
    def mu_init(self, x_val, y_val, z_val):
        if z_val:
            return np.array([x_val, y_val, z_val])
        else:
            return np.array([x_val, y_val])

    def run(self, parameters):
        # Function to call functions based on current state
        if self.state == "COLLECTING":
            self.add_data(parameters)
        elif self.state == "READY_TO_FIT":
            self.fit(self.data_stream)
        elif self.state == "FITTED":
            pass  # self.predict(parameters)
        else:
            pass
    # TODO add a function to predict using input data

    # Function to append data to self.data, and reevaluate whether the IGMM is ready to fit
    def add_data(self, data_stream):
        self.data = np.append(self.data, data_stream, 0)
        if len(self.data) > self.min_data:
            self.state = "READY_TO_FIT"

    # Function to add a point to a component
    # TODO generalize this for more than 2 dimensions
    def add(self, x_val, y_val):
        """
        add new data points

        @return:
        """
        point = np.array([x_val,y_val])
        updated = False
        for j in range(self.K):
            dist = self.mahalanobis(point, self.mus[j], self.alphas[j])
            if dist < self.mahalanobisMax:
                self.update(j, point)
                updated = True
        if not updated and self.remaining_k() > 0:
            self.create(point)
        else:
            self.add_outlier(point)
        pass

    # Function to record a point that was not classified as belonging to any components
    def add_outlier(self, point):
        if len(self.unclassified_points) == 0:
            self.unclassified_points = np.array([point])
        else:
            self.unclassified_points = np.append(self.unclassified_points, np.array([point]), 0)

    # Function to return the remaining number of components that can be created before reaching max # of components
    def remaining_k(self):
        return self.KMax - self.K

    # Function to return the current state of the IGMM
    def get_state(self):
        """

        @return:
        """
        return self.state

    # Function to fit the IGMM, if it is ready
    def fit(self, data_stream):
        self.add_data(data_stream)
        if not self.state == "READY_TO_FIT":
            return
        for i in range(len(self.data)):
            self.add(self.data[i][0], self.data[i][1])
        self.state = "FITTED"

    # Function to find and merge overlapping functions
    def merge_pass(self):
        done_merging = False
        while not done_merging:
            print(self.K)
            done_merging = True
            if self.K < 3:
                return
            for i in range(self.K):
                if i >= self.K:
                    break
                for j in range(i):
                    if j >= self.K or i >= self.K:
                        break
                    dist1 = self.mahalanobis(self.mus[i], self.mus[j], self.alphas[j])
                    dist2 = self.mahalanobis(self.mus[j], self.mus[i], self.alphas[i])
                    dist = (dist1 + dist2) / 2
                    if dist < self.mahalanobisMax:  # and dist1 < 0.1 * maxDist:
                        points = self.corresponding_data[j].copy() # look at all points in j
                        for x in self.corresponding_data[i]:
                            if not self.arr_in_list(x, self.corresponding_data[j]):
                                points = np.append(points, np.array([x]), 0)
                        done_merging = False
                        random.shuffle(points)
                        self.merge_clusters(points, i, j)
                        self.K -= 1

    # Function to merge 2 overlapping functions
    def merge_clusters(self, points, j1, j2):
        self.remove_cluster(j1)
        self.K -= 1
        self.remove_cluster(j2)
        self.K -= 1
        self.corresponding_data[self.K] = np.array([points[0]])
        self.create(points[0])

        for i in range(1, len(points)):
            dist = self.mahalanobis(points[i], self.mus[self.K - 1], self.alphas[self.K - 1])
            if dist < self.mahalanobisMax:
                self.corresponding_data[self.K - 1] = np.append(self.corresponding_data[self.K - 1], np.array([points[i]]), 0)
                # ZDict[i].append(getZScores(X[mergeX[i]], mus[K - 1], np.linalg.inv(alphas[K - 1]), dist))
                self.update(self.K - 1, points[i])

            else:
                self.add_outlier(points[i])

    # Unused function
    def split_clusters(self, j):
        pass

    # Does not change K
    # Function to remove a cluster from the model
    def remove_cluster(self, j):
        num_components = len(self.mus)
        iterables = [self.accumulators, self.ages, self.alphas, self.component_probabilities, self.corresponding_data,\
                     self.covariance_determinants, self.mus]
        for i in range(j, num_components - 1):
            for iterable in iterables:
                iterable[i] = iterable.pop(i + 1)
        num_components -= 1
        for i1 in range(num_components):
            summed = 0
            for i2 in range(num_components):
                summed += self.accumulators[i2]
            self.component_probabilities[i1] = self.accumulators[i1] / summed

    # Function to find the spatial distance between two vectors
    def spatial_distance(self, x1, x2):
        total = 0
        for i in range(len(x1)):
            total += (x1[i] - x2[i]) ** 2
        return total ** (1/2)

    # This function is currently unused. Checks if a cluster is sparse and requires a split
    def is_cluster_sparse(self, j):
        eigenvalues = np.linalg.eigvalsh(self.alphas[j])
        total_values = 0
        for value in eigenvalues:
            if value != 0:
                value = value ** -1
            total_values += value
        avg_value = total_values / (len(eigenvalues))
        num_less = 0
        for i in self.corresponding_data[j]:
            if self.spatial_distance(i, self.mus[j]) < avg_value:
                num_less += 1
        if num_less > (0.5 * len(self.corresponding_data[j])):
            self.split_clusters(j)

    # Function to find outliers
    # TODO fix this function so that outlier classification is more relaxed
    def get_outliers(self):
        notOutliers = self.corresponding_data[0]
        for i in range(1, self.K):
            notOutliers += self.corresponding_data[i]
        for i in self.corresponding_data[-1]:
            if True in [np.array_equal(i,x) for x in notOutliers]:
                self.corresponding_data[-1].remove(i)

    # Function to plot the clusters (Assumes 2 dimensional data)
    def plot(self):
        colors = ['r', 'b', 'y', 'k', 'm', 'c']
        for i in range(self.K):
            plot.figure(self.fig_num)
            plot.scatter(self.corresponding_data[i][:, 0], self.corresponding_data[i][:, 1], c = colors[i % 6])
            plot.scatter((self.mus[i][0]), (self.mus[i][1]), c='g')
        plot.show()
        pass

    # Unused function
    def model_params(self):
        pass

    # Function to calculate mahalanobis distance between a point x and a component
    def mahalanobis(self, x, mu, alpha):
        # Function to calculate mahalanobis distance
        # alpha is the precision matrix ( inverse of co-variance matrix)
        try:
            return abs(np.matmul((x - mu).transpose(), np.matmul(alpha, x - mu)))
        except ValueError:
            print("alpha = {}, x = {}, mu = {}".format(alpha, x, mu))

    # Function to check if an array is in a given list of arrays
    def arr_in_list(self, arr, arr_list):
        for arr_2 in arr_list:
            if np.array_equal(arr, arr_2):
                return True
        return False

    # Function to create a component with mu = x
    # Increments K by 1
    def create(self, x):
        K = self.K
        self.accumulators[K] = 1.
        self.ages[K] = 1
        self.alphas[K] = self.sigmaInverse * np.identity(self.feature_dimension)
        self.component_probabilities[K] = 1 / sum(self.accumulators.values())
        self.covariance_determinants[K] = np.linalg.det(self.alphas[K]) ** -1
        self.mus[K] = np.copy(x)
        self.corresponding_data[K] = np.array([(self.mus[K])])
        self.K += 1

    # update a component j using the posterior probabilities of data point x
    # Algorithm 2 update
    def update(self, j, x):
        self.corresponding_data[j] = np.append(self.corresponding_data[j], np.array([x]), 0)
        dist = (self.mahalanobis(x, self.mus[j], self.alphas[j]))

        pxj = 1 / ((2 * math.pi) ** (self.feature_dimension / 2) * math.sqrt(self.covariance_determinants[j]))
        pxj *= math.exp(-0.5 * dist)  #

        #if self.printing:
        #    print(pxj)

        pjx = pxj * self.component_probabilities[j]  #
        totProbs = 0
        for i in range(self.K):
            pxi = 1 / ((2 * math.pi) ** (self.feature_dimension / 2) * math.sqrt(self.covariance_determinants[i]))
            pxi *= math.exp(-0.5 * dist)  #
            totProbs += pxi * self.component_probabilities[i]  #

        if totProbs == 0:
            print(self.component_probabilities)
            print()
        try:
            pjx /= totProbs  #

        except ZeroDivisionError as err:
            print(str(err) + traceback.format_exc() + "pjx = {}, totProbs = {}".format(pjx, totProbs))

        #if self.printing:
        #    print(pjx)

        self.ages[j] += 1
        self.accumulators[j] += pjx

        ej = x - self.mus[j]
        weight = pjx / self.accumulators[j]

        deltaMu = weight * ej
        self.mus[j] += deltaMu

        oldAlpha = self.alphas[j]

        inputLen = self.feature_dimension
        e = np.ndarray([inputLen, 1])
        for i in range(inputLen):
            e[i, 0] = ej[i]
        ej = e

        newAlpha = np.matmul(oldAlpha, np.matmul(ej, np.matmul(np.transpose(ej), oldAlpha)))
        newAlpha *= (weight * (1 - 3 * weight + weight ** 2)) / ((weight - 1) ** 2 * (weight ** 2 - 2 * weight - 1))
        self.alphas[j] = newAlpha + oldAlpha / (1 - weight)

        self.component_probabilities[j] = self.accumulators[j] / sum(self.accumulators.values())

        self.covariance_determinants[j] = (1 - weight) ** self.feature_dimension * self.covariance_determinants[j] * (
                1 + (weight * (1 + weight * (weight - 3))) / (1 - weight) * np.matmul(ej.transpose(),
                                                                                      np.matmul(oldAlpha, ej)))
        pass
        #return mus

# Function to generate sample data for the GMM
def generate_sample_data():
    # generate data from 5 normal gaussian distributions
    D = 4;

    sigmas = np.ndarray([D + 1, 2, 2])
    mus = np.array([[1, 1], [10, 10], [10, 1], [1, 10], [5, 5]])
    starting_mus = mus
    for i in range(D + 1):
        sigmas[i] = [[1, 0], [0, 1]]

    num_vectors = 1500;
    len_vectors = 2

    data = np.ndarray((num_vectors, len_vectors))
    starting_correspondences = []

    count = 0
    for i in range(num_vectors):
        index = random.randint(0, 4)
        starting_correspondences.append(index)
        data[i] = np.random.multivariate_normal(mus[index], sigmas[index])

    splot = plot.subplot(1, 1, 1)

    xs = [[], [], [], [], []]
    ys = [[], [], [], [], []]
    colors = ['r', 'b', 'k', 'y', 'm']

    for i in range(len(data)):
        index = starting_correspondences[i]
        xs[index].append(data[i, 0])
        ys[index].append(data[i, 1])
    for i in range(5):
        plot.scatter(xs[i], ys[i], c =colors[i])
        v, w = np.linalg.eigh(sigmas[i])
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180.0 * angle / np.pi  # convert to degree
        xs_to_plot = xs[i]
        ys_to_plot = ys[i]
        plot.scatter(xs_to_plot, ys_to_plot, c=colors[i])
        plot.scatter((mus[i][0]), (mus[i][1]), c='g')

        ell = mpl.patches.Ellipse(mus[i], v[0], v[1], 180.0 + angle, color=colors[i % 5])
        ell.set_clip_box(splot.bbox)

        ell.set_alpha(0.5)
        splot.add_artist(ell)
    return [data, starting_mus, sigmas, starting_correspondences, count/num_vectors]

# Function called to run the GMM
def main():
    #TODO merge should require state
    #TODO restructure so run is called instead of fit
    data = generate_sample_data()
    data_stream = data[0]
    iGMM = IncrementalGMM(data_stream[0][0], data_stream[0][1])
    iGMM.fit(data_stream[1:])
    #iGMM.merge_pass()
    iGMM.plot()
    val = 0
    for i in range(iGMM.K):
        print(len(iGMM.corresponding_data[i]))
        val += len(iGMM.corresponding_data[i])
    print(val)
    print(len(iGMM.unclassified_points))
main()