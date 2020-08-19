import numpy as np
from scipy.linalg import block_diag
from .utils import (
    count_not_none,
    ImpossibleDistribution,
    simulate_correlated_bernoulli,
)

        
class PopulationDistribution:
    """Generates random populations given the infection probability of each
    person and the correlation structure.

    Either `probabilities` must be provided or both `prevalence` and
    `population` must be provided, but not all three.

    `correlation`, `correlation_matrix` and `cluster_correlations` are
    optional, but only one can be provided.

    Parameters
    ----------
    probabilities : 1D array-like, optional
        The infection probability of each person in the population. This may
        be different for each person.
    prevalence : float, optional
        The infection probability of each person in the population. This is
        the same for all people.
    population : int, optional
        The size of the population.
    correlation : non-negative number, optional
        If this provided, all people are deemed to be part of a single cluster
        and have the same pairwise correlation with regard to infection status.
    correlation_matrix : 2D NumPy array-like, optional
        A positive definite matrix containing non-negative correlations of the
        infection status of each pair of people in the population. If
        `probabilities` is given, the rows/columns of `correlation_matrix`
        must be in the same order as `probabilities`. For example, the second
        row/column must correspond to the person whose probability of
        infection is given by the second element of `probabilities`.
    cluster_correlations : 2D array-like, optional
        If a 2D array-like object is provided, it must have two columns where
        the first column contains the non-negative pairwise correlation of the
        infection statuses of the people in each cluster and the second column
        contains the number of people in each cluster. If the sum of all the
        cluster sizes is less than the population, then the infection statuses
        of the remaining people are assumed to have no correlation with each
        other. If any cluster has a size of one, then the correlation of that
        cluster is ignored. If `probabilities` is given, the individuals must
        be in the same order as in `cluster_correlations`. For instance, if
        `cluster_correlations` is [(0.3, 5), (0.2, 4)], the infection
        probabilities of the individuals in the cluster with correlation of
        0.3 must be be the first five elements of `probabilities`, and the
        infection probabilities of the individuals in the cluster with
        correlation of 0.2 must be the next four elements of `probabilities`.
        The correlation of the infection probabilities of individuals in
        different clusters is assumed to be 0.
    num_samples : int, optional
        The number of populations to generate.
    random_state : int, optional
        If an int is given, `random_state` is the seed that will be used by
        the random number generator.

    Attributes
    ----------
    _probabilities : 1D NumPy ndarray
        The infection probability of each person in the population.
    _cluster_correlations : 2D NumPy ndarray or None
        If the infection statuses among people are uncorrelated or if
        `correlation_matrix` is given, then this is None. Otherwise, this is a
        2D NumPy ndarray where the first column contains the pairwise
        correlation of the infection statuses of the people in each cluster
        and the second column contains the number of people in each cluster.
        The individuals in the clusters are in the same order as in
        `_probabilities`. For instance, if `_cluster_correlations` is
        array([[0.3, 5], [0.2, 4]]), the infection probabilities of the
        individuals in the cluster with correlation of 0.3 is given by the
        first five elements of `_probabilities`, and the infection
        probabilities of the individuals in the cluster with correlation of
        0.2 is given by the next four elements of `_probabilities`.
    _correlation_matrix : 2D NumPy ndarray
        A `len(_probabilities)` by `len(_probabilities)` correlation matrix
        containing the pairwise correlation of the infection statuses of each
        individual in the population. The rows and columns are in the same
        order as `_probabilities`.
    _samples : 2D NumPy ndarray
        A `num_samples` by `len(_probabilities)` array of the randomly
        generated infection statuses of each person.

    Raises
    ------
    TypeError
        * If it is not the case that only `probabilities` has been given or
          only `prevalence` and `probability` have been given together.
        * If more than one of `correlation`, `correlation_matrix` and
          `cluster_correlations` has been given.
    ImpossiblePopulationDistribution
        If population samples with the specified probabilities and correlation
        structure cannot be generated.

    Notes
    -----
    If a correlation structure is provided, the time it will take to generate
    random samples will blow out as the population starts to exceed a
    thousand.

    """    
    def __init__(self,
                 probabilities = None,
                 prevalence = None,
                 population = None,
                 correlation = None,
                 correlation_matrix = None,
                 cluster_correlations = None,
                 num_samples = 1,
                 random_state = None):
        prevalence_population_count = count_not_none(prevalence, population)
        if probabilities is not None:
            if prevalence_population_count > 0:
                raise TypeError('If `probabilities` is given, then `prevalence` and `population` must not be given.')
            self._probabilities = np.array(probabilities)
        elif prevalence_population_count == 2:
            self._probabilities = np.full(shape = population, fill_value = prevalence)
        else:
            raise TypeError('Either `prevalence` and `population` must be given together or `probabilities` must be given.')

        if count_not_none(correlation, correlation_matrix, cluster_correlations) > 1:
            raise TypeError('Only one of `correlation`, `correlation_matrix` or `cluster_correlations` can be given.')
        identity_matrix = np.eye(self.population)
        if cluster_correlations is not None:
            cluster_correlations = np.array(cluster_correlations) if len(cluster_correlations) else None
        elif correlation is not None:
            if correlation != 0: # correlation < 0 should raise ValueError in simulate_correlated_bernoulli()
                cluster_correlations = np.array([[correlation, self.population]])
        elif correlation_matrix is not None:
            correlation_matrix = np.array(correlation_matrix)
            if np.allclose(correlation_matrix, identity_matrix):
                correlation_matrix = None

        self._cluster_correlations = cluster_correlations
        self._correlation_matrix = correlation_matrix
        if self._cluster_correlations is None and self._correlation_matrix is None: # uncorrelated case
            self._correlation_matrix = identity_matrix
            self._samples = np.random.RandomState(random_state).binomial(n = 1,
                                                                         p = self._probabilities,
                                                                         size = (num_samples, len(self._probabilities)))
        else:
            if self._cluster_correlations is not None:
                self._correlation_matrix = self._make_correlation_matrix(cluster_correlations = self._cluster_correlations,
                                                                         population = self.population)
            try:
                self._samples = simulate_correlated_bernoulli(p = self._probabilities,
                                                              correlation = correlation or self._correlation_matrix,
                                                              num_samples = num_samples,
                                                              random_state = random_state)
            except ImpossibleDistribution:
                raise ImpossiblePopulationDistribution('Not possible to generate population samples with the specified '
                                                       'infection probabilities and correlation structure.') from None


    @staticmethod
    def _make_correlation_matrix(cluster_correlations, population):
        """Returns a `population` by `population` correlation matrix
        corresponding to `cluster_correlations`.

        Parameters
        ----------
        cluster_correlations : 2D NumPy ndarray
            A 2D NumPy ndarray where the first column contains the pairwise
            correlation of the infection statuses of the people in each
            cluster and the second column contains the number of people in
            each cluster.
        population : int
            The size of the population.

        Returns
        -------
        2D NumPy ndarray
            A `population` by `population` correlation matrix. Clusters of
            size 1 will have a correlation of 1 regardless of the correlation
            given in `cluster_correlations`.

        """
        missing = population - cluster_correlations[:, 1].sum()
        if missing > 0:
            cluster_correlations = np.append(arr = cluster_correlations,
                                             values = [[0, missing]],
                                             axis = 0)
        correlation_matrices = []
        for cluster_correlation, cluster_size in cluster_correlations:
            cluster_size = int(cluster_size)
            correlation_matrix = np.full(shape = (cluster_size, cluster_size),
                                         fill_value = cluster_correlation)
            np.fill_diagonal(correlation_matrix, 1)
            correlation_matrices.append(correlation_matrix)
        return block_diag(*correlation_matrices)


    @property
    def probabilities(self):
        """Returns a 1D NumPy ndarray of the infection probabilities of each
        person in the population.
        """
        return self._probabilities


    @property
    def cluster_correlations(self):
        """Returns a 2D NumPy ndarray where the first column contains the
        pairwise correlation of the infection statuses of the people in each
        cluster and the second column contains the number of people in each
        cluster.

        If there are no clusters or no cluster identification has not been
        performed, then None will be returned.
        """
        return self._cluster_correlations


    @property
    def correlation_matrix(self):
        """Returns a 2D NumPy ndarray containing the pairwise correlation of
        the infection statuses of each individual in the population.
        
        The rows and columns are in the same order as `probabilities`.
        """
        return self._correlation_matrix


    @property
    def population(self):
        """Returns the size of the population."""
        return len(self._probabilities)


    @property
    def samples(self):
        """Returns an n by k NumPy ndarray of the randomly generated infection
        statuses of each person, where n is the number of samples and k is the
        population.
        """
        return self._samples


    @property
    def num_samples(self):
        """Returns the number of population samples created."""
        return len(self._samples)

        
    def __iter__(self):
        for population in self._samples:
            yield population



class ImpossiblePopulationDistribution(ValueError):
    """Raised when population samples with the specified probabilities and
    correlation structure cannot be generated.
    """