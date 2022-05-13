import numpy as np
import math
import traceback
import matplotlib.pyplot as plot
import random


class IncrementalGMM(object):
    """

    """

    # def __init__(self, x_vector, y_vector, categorical_data=False, binary_data=False, ordered_pairs=True, weighted=False, validate_input=True):
    def __init__(self, x_val, y_val, **kwargs):
        # accumulators , and standard base initialization
        self.vector_length = 2
        self.mus = {0:np.array([x_val,y_val])}
        self.sigmaInverse = 1. # Need better initialization for sigma here
        self.sigmas = {0:np.array([[(x_val+y_val)/(2+x_val),0],[0,(x_val+y_val)/(2+y_val)]])}  # sigma should be representative of only a part of the data stream
        self.covariance_determinants = {0:np.linalg.det(self.sigmas[0])}
        self.component_probabilities = {0:1.}
        self.ages = {0:1}
        self.accumulators = {0:1.}
        self.alphas = {0:1.}
        self.corresponding_data = {0:[np.copy(self.mus[0])]}
        self.K = 1
        self.KMax = 10
        self.fig_num = 1
        self.mahalanobisMax = 6
        self.printing = False
        self.converged = False # when model converges or fitted, it is True
        self.model_param = None # for persistence
        self.state = 'COLLECTING' # READY_TO_FIT, 'FITTED', 'EXPIRED'
        self.unclassified_points = []
        pass

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
                self.update(j,point)
                updated = True
        if not updated and self.remainingK() > 0:
            self.create(point)
        else:
            self.add_outlier(point)
        pass

    def add_outlier(self, point):
        self.unclassified_points.append(point)

    def remainingK(self):
        return self.KMax - self.K

    def is_fitted(self):
        """

        @return:
        """
        #Need to implement convergence checks here
        #return self.converge

    def fit(self, data_stream):
        for i in range(len(data_stream)):
            self.add(data_stream[i][0], data_stream[i][1])
        self.plot()

    def merge_clusters(self):
        pass

    def split_clusters(self, j):
        pass

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


    def spatial_distance(self, x1, x2):
        total = 0
        for i in range(len(x1)):
            total += (x1[i] - x2[i]) ** 2
        return total ** (1/2)

    def is_cluster_sparse(self, j):
        # TODO determine whether sanity check on age is needed
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

    def get_outliers(self):
        notOutliers = self.corresponding_data[0]
        for i in range(1, self.K):
            notOutliers += self.corresponding_data[i]
        for i in self.corresponding_data[-1]:
            if True in [np.array_equal(i,x) for x in notOutliers]:
                self.corresponding_data[-1].remove(i)
        #TODONE import this code from jupyter notebook

    def plot(self):
        colors = ['r', 'b', 'y', 'k', 'm', 'c']
        for i in range(self.K):
            plot.figure(self.fig_num)
            A = self.corresponding_data[i]

            Xs_to_plot = A[:][0]
            Ys_to_plot = A[:][1]
            plot.scatter(Xs_to_plot, Ys_to_plot, c=colors[i % 6])
            plot.scatter((self.mus[i][0]), (self.mus[i][1]), c='g')
        # #getOutliers(self.corresponding_data, K)
        # A = self.corresponding_data[-1]
        # Xs_to_plot = A[:][0]
        # Ys_to_plot = A[:][1]
        # #plot.scatter(Xs_to_plot, Ys_to_plot, c='c')
        plot.show()
        pass

    def model_params(self):
        pass


    # function to calculate mahalanobis distance
    def mahalanobis(self, x, mu, alpha):
        # alpha is the precision matrix ( inverse of co-variance matrix)
        try:
            return abs(np.matmul((x - mu).transpose(), np.matmul(alpha, x - mu)))
        except ValueError:
            print("alpha = {}, x = {}, mu = {}".format(alpha, x, mu))

    def create(self, x):
        K = self.K
        self.accumulators[K] = 1.
        self.ages[K] = 1
        self.alphas[K] = self.sigmaInverse * np.identity(self.vector_length)
        self.component_probabilities[K] = 1 / sum(self.accumulators.values())
        self.covariance_determinants[K] = np.linalg.det(self.alphas[K]) ** -1
        self.mus[K] = np.copy(x)
        self.corresponding_data[K] = {K:[np.copy((self.mus[K]))]}
        self.K += 1

    # update a component j using the posterior probabilities of data point x
    # Algorithm 2 update
    def update(self, j, x):
        self.corresponding_data[j].append(np.copy(x))
        dist = (self.mahalanobis(x, self.mus[j], self.alphas[j]))

        pxj = 1 / ((2 * math.pi) ** (self.vector_length / 2) * math.sqrt(self.covariance_determinants[j]))
        pxj *= math.exp(-0.5 * dist)  #

        if self.printing:
            print(pxj)

        pjx = pxj * self.component_probabilities[j]  #
        totProbs = 0
        for i in range(self.K):
            pxi = 1 / ((2 * math.pi) ** (self.vector_length / 2) * math.sqrt(self.covariance_determinants[i]))
            pxi *= math.exp(-0.5 * dist)  #
            totProbs += pxi * self.component_probabilities[i]  #

        if totProbs == 0:
            print(self.component_probabilities)
            print()
        try:
            pjx /= totProbs  #

        except ZeroDivisionError as err:
            print(str(err) + traceback.format_exc() + "pjx = {}, totProbs = {}".format(pjx, totProbs))

        if self.printing:
            print(pjx)

        self.ages[j] += 1
        self.accumulators[j] += pjx

        ej = x - self.mus[j]
        weight = pjx / self.accumulators[j]

        deltaMu = weight * ej
        self.mus[j] += deltaMu

        oldAlpha = self.alphas[j]

        inputLen = self.vector_length
        e = np.ndarray([inputLen, 1])
        for i in range(inputLen):
            e[i, 0] = ej[i]
        ej = e

        newAlpha = np.matmul(oldAlpha, np.matmul(ej, np.matmul(np.transpose(ej), oldAlpha)))
        newAlpha *= (weight * (1 - 3 * weight + weight ** 2)) / ((weight - 1) ** 2 * (weight ** 2 - 2 * weight - 1))
        self.alphas[j] = newAlpha + oldAlpha / (1 - weight)

        self.component_probabilities[j] = self.accumulators[j] / sum(self.accumulators.values())

        self.covariance_determinants[j] = (1 - weight) ** self.vector_length * self.covariance_determinants[j] * (
                1 + (weight * (1 + weight * (weight - 3))) / (1 - weight) * np.matmul(ej.transpose(),
                                                                                      np.matmul(oldAlpha, ej)))
        pass
        #return mus
def generate_sample_data():
    # generate data from 5 normal gaussian distributions
    D = 4;

    sigmas = np.ndarray([D + 1, 2, 2])
    mus = np.array([[1, 1], [10, 10], [10, 1], [1, 10], [5, 5]])
    starting_mus = mus
    for i in range(D + 1):
        sigmas[i] = [[1, 0], [0, 1]]

    numvectors = 1500;
    lenvectors = 2
    X = np.ndarray((numvectors, lenvectors))
    startingCorrespondences = []

    maxZ = 0
    count = 0
    for i in range(numvectors):
        distant = False
        index = random.randint(0, 4)
        startingCorrespondences.append(index)
        X[i] = np.random.multivariate_normal(mus[index], sigmas[index])
        for j in range(lenvectors):
            ZScore = (X[i][j] - mus[index][j]) / sigmas[index][j, j]
            if abs(ZScore) > 3:
                distant = True
            if abs(ZScore) > abs(maxZ):
                maxZ = ZScore
        if distant:
            count += 1
    pass
    return [X, starting_mus, sigmas, startingCorrespondences, maxZ, count/numvectors]


def main():
    data = generate_sample_data()
    X = data[0]
    IGMM = IncrementalGMM(X[0][0], X[0][1])
    IGMM.fit(X[1,:])
    IGMM.plot()

main()