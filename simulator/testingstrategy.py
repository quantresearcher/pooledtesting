import numpy as np
from abc import abstractmethod, ABCMeta


class Test:
    """A diagnostic test for detecting the presence of a disease.

    Parameters
    ----------
    sensitivity : float
        The true positive rate of the test.
    specificity : float
        The true negative rate of the test.
    random_state : int, optional
        If an int is given, `random_state` is the seed that will be used by
        the random number generator.

    Attributes
    ----------
    _sensitivity : see Parameters
    _specificity : see Parameters
    _random_number_generator : random.Random
        A random number generator.

    """    
    def __init__(self,
                 sensitivity,
                 specificity,
                 random_state = None):
        self._sensitivity = sensitivity
        self._specificity = specificity
        self._random_number_generator = np.random.RandomState(random_state)


    @property
    def sensitivity(self):
        """Returns the true positive rate of the test."""
        return self._sensitivity


    @property
    def specificity(self):
        """Returns the true negative rate of the test."""
        return self._specificity


    def conduct(self, pools):
        """Returns a NumPy ndarray of bools for whether each pool represented
        in `pools` tests positive or not.

        Parameters
        ----------
        pools : NumPy ndarray
            Contains the infection status of each pool - a truthy value for
            infected and a falsy value for non-infected. Can be any shape.

        Returns
        -------
        NumPy ndarray
            An array of bools for whether the corresponding pool in `pools`
            tests positive or not. This is the same shape as `pools`.

        """
        positive_test_probabilities = np.where(pools, self._sensitivity, 1 - self._specificity)
        return self._random_number_generator.rand(*positive_test_probabilities.shape) < positive_test_probabilities


    def __str__(self):
        return f'Test(sensitivity = {self._sensitivity}, specificity = {self._specificity})'


class TestingStrategy(metaclass = ABCMeta):
    """A strategy for testing a population.

    Parameters
    ----------
    test : Test
        The diagnostic test to use.

    Attributes
    ----------
    _test : see Parameters
    _total_tests_conducted : int
        The total number of tests conducted.

    """
    def __init__(self,
                 test):
        self._test = test
        self._total_tests_conducted = 0


    @property
    def test(self):
        """Returns the diagnostic Test used."""
        return self._test
    

    @property
    def total_tests_conducted(self):
        """Returns the total number of tests conducted."""
        return self._total_tests_conducted
    

    @abstractmethod
    def run(self, population_distribution):
        """Conducts testing on all the population samples of
        `population_distribution`.

        Parameters
        ----------
        population_distribution : PopulationDistribution
            Contains the population samples to test on, and the infection
            probabilities and correlation structures of the people within
            the population.

        """


class TwoStageTestingMixin:
    """Mixin class for all strategies that employ some variant of two-stage
    hierarchical testing.
    """
    def _run(self, population_samples, pool_size):
        """Returns the total number of tests conducted when two-stage testing
        is performed on `population_samples` using pools of size `pool_size`
        in the first stage.

        Parameters
        ----------
        population_samples : 2D NumPy ndarray
            Each column contains the infection status of a person over
            different samples. The infection status must be 1s and 0s, or
            bools.
        pool_size : int
            The number of people to include in each pool in the first stage.

        Returns
        -------
        int
            The total number of tests conducted.

        """
        if pool_size == 1:
            return population_samples.size
        population = population_samples.shape[1]
        remaining_pool_size = population % pool_size
        col_index = population - remaining_pool_size
        pools = population_samples[:, :col_index].reshape(-1, pool_size).any(axis = 1)
        num_positive_tests = self._test.conduct(pools).sum()
        total_tests_conducted = pools.size + num_positive_tests * pool_size
        if remaining_pool_size > 0:
            pools = population_samples[:, col_index:].any(axis = 1)
            total_tests_conducted += pools.size
            if remaining_pool_size > 1:
                num_positive_tests = self._test.conduct(pools).sum() 
                total_tests_conducted += num_positive_tests * remaining_pool_size
        return total_tests_conducted


class ThreeStageTestingMixin:
    """Mixin class for all strategies that employ some variant of three-stage
    hierarchical testing.
    """
    def _run(self,
             population_samples,
             stage2_pool_sizes):
        """Returns the total number of tests conducted when three-stage
        testing is performed on `population_samples` using pools of size
        `sum(stage2_pool_sizes)` in the first stage and pools of size
        `stage2_pool_sizes` in the second stage.

        Parameters
        ----------
        population_samples : 2D NumPy ndarray
            Each column contains the infection status of a person over
            different samples. The infection status must be 1s and 0s, or
            bools. The population must be greater than or equal to the pool
            size in the first stage of testing.
        stage2_pool_sizes : array-like of ints
            The number of people to include in each pool in the second stage.

        Returns
        -------
        int
            The total number of tests conducted.

        """
        stage2_pool_sizes = np.array(stage2_pool_sizes)
        stage1_pool_size = stage2_pool_sizes.sum()
        population = population_samples.shape[1]
        remaining_stage1_pool_size = population % stage1_pool_size
        col_index = population - remaining_stage1_pool_size
        samples = population_samples[:, :col_index].reshape(-1, stage1_pool_size)
        stage1_pools = samples.any(axis = 1)
        stage1_positive_tests_mask = self._test.conduct(stage1_pools)
        indices = np.insert(np.cumsum(stage2_pool_sizes), 0, [0])
        indices = indices[indices < samples.shape[1]]
        stage2_pools = np.add.reduceat(samples[stage1_positive_tests_mask],
                                       indices = indices,
                                       axis = 1)
        stage2_positive_tests_mask = self._test.conduct(stage2_pools)
        stage2_pool_sizes_greater_than_one = stage2_pool_sizes.copy()
        stage2_pool_sizes_greater_than_one[stage2_pool_sizes_greater_than_one <= 1] = 0
        total_tests_conducted = stage1_pools.size + stage2_pools.size + (stage2_positive_tests_mask * stage2_pool_sizes_greater_than_one).sum()
        if remaining_stage1_pool_size > 0:
            samples = population_samples[:, col_index:]
            stage1_pools = samples.any(axis = 1)
            stage1_positive_tests_mask = self._test.conduct(stage1_pools)
            total_tests_conducted += stage1_pools.size
            if remaining_stage1_pool_size > 1:
                indices = np.insert(np.cumsum(stage2_pool_sizes), 0, [0])
                indices = indices[indices < samples.shape[1]]
                stage2_pools = np.add.reduceat(samples[stage1_positive_tests_mask],
                                               indices = indices,
                                               axis = 1)
                stage2_positive_tests_mask = self._test.conduct(stage2_pools)
                stage2_pool_sizes_greater_than_one = stage2_pool_sizes[:stage2_positive_tests_mask.shape[1]].copy()
                stage2_pool_sizes_greater_than_one[-1] -= stage2_pool_sizes_greater_than_one.sum() - remaining_stage1_pool_size
                stage2_pool_sizes_greater_than_one[stage2_pool_sizes_greater_than_one <= 1] = 0
                total_tests_conducted += stage2_pools.size + (stage2_positive_tests_mask * stage2_pool_sizes_greater_than_one).sum()
        return total_tests_conducted


class TwoDimensionalArrayTestingMixin:
    """Mixin class for all strategies that employ some variant of
    two-dimensional array testing.
    """
    def _run(self,
             population_samples,
             num_rows,
             num_columns,
             test_master_pools):
        """Returns the total number of tests conducted on `population_samples`
        when a two-dimensional grid of `num_rows` rows and `num_columns`
        columns is used to form pools.

        Parameters
        ----------
        population_samples : 2D NumPy ndarray
            Each column contains the infection status of a person over
            different samples. The infection status must be 1s and 0s, or
            bools. The population must be greater than or equal to the product
            of `num_rows` and `num_columns`.
        num_rows : int
            The number of rows in the two-dimensional grid.
        num_columns : int
            The number of columns in the two-dimensional grid.
        test_master_pools : bool
            Whether to test all the people in each grid as a pool first. If
            the pool tests negative, then no testing of rows or columns will be
            required.

        Returns
        -------
        int
            The total number of tests conducted.

        Notes
        -----
        There are three possible outcomes when the pools from a grid are
        tested:

            * if exactly one row or one column pool tests positive, then
              individual retesting is not required
            * if at least two rows and at least two columns test positive,
              then individual retesting needs to be performed on the
              samples at the intersections
            * if a row (column) tests positive without any column (row)
              testing positive, everyone within the row (column) needs to be
              retested individually

        In this implementation, those people that are not part of any grid due
        to non-divisibility are tested individually.

        """
        total_tests_conducted = 0
        population = population_samples.shape[1]
        grid_size = num_rows * num_columns
        num_grids, num_individuals_remaining = divmod(population, grid_size)
        grids = population_samples[:, :population - num_individuals_remaining].reshape(-1, num_rows, num_columns)
        if test_master_pools:
            master_pools = grids.any(axis = (1, 2))
            total_tests_conducted += len(master_pools)
            grids = grids[self._test.conduct(master_pools)]
        row_pools = grids.any(axis = 2)
        column_pools = grids.any(axis = 1)
        num_positive_rows = self._test.conduct(row_pools).sum(axis = 1)
        num_positive_columns = self._test.conduct(column_pools).sum(axis = 1)
        min_positive_dimension = np.minimum(num_positive_rows, num_positive_columns)
        total_pools_tested = (num_rows + num_columns) * len(grids)
        min_positive_dimension = np.minimum(num_positive_rows, num_positive_columns)
        total_individuals_tested = np.where(min_positive_dimension > 1, num_positive_rows * num_positive_columns, 0).sum() + \
                                   np.where(min_positive_dimension == 0, num_positive_rows * num_columns + num_positive_columns * num_rows, 0).sum()
        total_tests_conducted += total_pools_tested + total_individuals_tested + num_individuals_remaining * len(population_samples)
        return total_tests_conducted


class TwoStageTesting(TestingStrategy, TwoStageTestingMixin):
    """A strategy that involves testing pools of people in the first stage and
    retesting people from those pools that test positive individually in the
    second stage.

    This strategy does not take into account the infection probability of each
    person and the correlation structure.
    
    Parameters
    ----------
    test : Test
        The diagnostic test to use. The sensitivity and specificity are
        assumed to be constant across all pool sizes.
    pool_size : int
        The number of people to include in each pool in the first stage.
        
    Attributes
    ----------
    _pool_size : see Parameters
        
    """
    def __init__(self,
                 test,
                 pool_size):
        super().__init__(test)
        self._pool_size = pool_size


    def run(self, population_distribution):
        """Conducts testing on all the population samples of
        `population_distribution`.

        Parameters
        ----------
        population_distribution : PopulationDistribution
            Contains the population samples to test on.

        Raises
        ------
        ValueError
            If `population_distribution.population` is less than the pool
            size.

        """
        if population_distribution.population < self._pool_size:
            raise ValueError('The population must be greater than or equal to the pool size.')
        self._total_tests_conducted = self._run(population_samples = population_distribution.samples,
                                                pool_size = self._pool_size)


    def __repr__(self):
        return f'TwoStageTesting(test = {self._test}, pool_size = {self._pool_size})'
        

class ThreeStageTesting(TestingStrategy, ThreeStageTestingMixin):
    """A strategy that involves testing pools of people in the first stage,
    testing smaller pools of people in the second stage, and testing people
    individually in the third stage.

    A person is only tested in a stage if the pool they were a part of in the
    previous stage tested positive.

    This strategy does not take into account the infection probability of each
    person and the correlation structure.
    
    Parameters
    ----------
    test : Test
        The diagnostic test to use. The sensitivity and specificity are
        assumed to be constant across all pool sizes.
    pool_sizes : array-like of ints
        The number of people to include in each pool in the second stage. The
        pool size to use in the first stage will be the sum of all these.
        
    Attributes
    ----------
    _stage1_pool_size : int
        The number of people to include in each pool in the first stage.
    _stage2_pool_sizes : list of ints
        The number of people to include in each pool in the second stage.
        
    """
    def __init__(self,
                 test,
                 pool_sizes):
        super().__init__(test)
        self._stage1_pool_size = sum(pool_sizes)
        self._stage2_pool_sizes = list(pool_sizes)
        

    def run(self, population_distribution):
        """Conducts testing on all the population samples of
        `population_distribution`.

        Parameters
        ----------
        population_distribution : PopulationDistribution
            Contains the population samples to test on.

        Raises
        ------
        ValueError
            If `population_distribution.population` is less than the pool size
            used in the first stage.

        """
        if population_distribution.population < self._stage1_pool_size:
            raise ValueError('The population must be greater than or equal to the pool size in the first stage.')
        self._total_tests_conducted = self._run(population_samples = population_distribution.samples,
                                                stage2_pool_sizes = self._stage2_pool_sizes)


    def __repr__(self):
        return f'ThreeStageTesting(test = {self._test}, pool_sizes = {self._stage2_pool_sizes})'


class TwoDimensionalArrayTesting(TestingStrategy, TwoDimensionalArrayTestingMixin):
    """A strategy that involves arranging individuals in a two-dimensional
    grid, testing pools formed from a single row or column, and then
    separately retesting those individuals that lie at intersections of
    positive rows and columns.

    If the population is not a multiple of the grid size, then the people that
    aren't part of any grid are tested individually.

    Parameters
    ----------
    test : Test
        The diagnostic test to use. The sensitivity and specificity are
        assumed to be constant across all pool sizes.
    num_rows : int
        The number of rows in the two-dimensional grid.
    num_columns : int
        The number of columns in the two-dimensional grid.
    test_master_pools : bool
        Whether to test all the people in each grid as a pool first. If
        the pool tests negative, then no testing of rows or columns will be
        required.

    Attributes
    ----------
    _num_rows : see Parameters
    _num_columns : see Parameters
    _test_master_pools : see Parameters

    """
    def __init__(self,
                 test,
                 num_rows,
                 num_columns,
                 test_master_pools = False):
        super().__init__(test)
        self._num_rows = num_rows
        self._num_columns = num_columns
        self._test_master_pools = test_master_pools


    def run(self, population_distribution):
        """Conducts testing on all the population samples of
        `population_distribution`.

        Parameters
        ----------
        population_distribution : PopulationDistribution
            Contains the population samples to test on.

        Raises
        ------
        ValueError
            If `population_distribution.population` is less than the grid
            size.

        """
        if population_distribution.population < self._num_rows * self._num_columns:
            raise ValueError('The population must be greater than or equal to the grid size.')
        self._total_tests_conducted = self._run(population_samples = population_distribution.samples,
                                                num_rows = self._num_rows,
                                                num_columns = self._num_columns,
                                                test_master_pools = self._test_master_pools)


    def __repr__(self):
        return f'TwoDimensionalArrayTesting(test = {self._test}, num_rows = {self._num_rows}, ' \
               f'num_columns = {self._num_columns}, test_master_pools = {self._test_master_pools})'


class HighRiskLowRiskTwoStageTesting(TestingStrategy, TwoStageTestingMixin):
    """A strategy that involves first dividing the population into low-risk
    and high-risk groups, and then performing regular two-stage testing
    separately on these two groups.

    Parameters
    ----------
    test : Test
        The diagnostic test to use. The sensitivity and specificity are
        assumed to be constant across all pool sizes.
    pool_sizes : array-like of 2 ints
        The first int should be the pool size for people with infection
        probabilities below `threshold` and the second int should be the pool
        size for people with infection probabilities greater than or equal to
        `threshold`.
    threshold : float
        The probability of infection used to separate low-risk and high-risk
        individuals. Those individuals with probabilities below this threshold
        will be deemed low-risk and pooled together, and those with
        probabilities at or above the threshold will be deemed high-risk and
        pooled together.
        
    Attributes
    ----------
    _pool_sizes : see Parameters
    _threshold : see Parameters
        
    """
    def __init__(self,
                 test,
                 pool_sizes,
                 threshold):
        super().__init__(test)
        self._pool_sizes = pool_sizes
        self._threshold = threshold


    def run(self, population_distribution):
        """Conducts testing on all the population samples of
        `population_distribution`.

        Parameters
        ----------
        population_distribution : PopulationDistribution
            Contains the population samples to test on.

        """
        low_risk_people_mask = population_distribution.probabilities < self._threshold
        low_risk_people = population_distribution.samples[:, low_risk_people_mask]
        high_risk_people = population_distribution.samples[:, ~low_risk_people_mask]
        for subpopulation_samples, pool_size in zip([low_risk_people, high_risk_people], self._pool_sizes):
            self._total_tests_conducted += self._run(population_samples = subpopulation_samples,
                                                     pool_size = pool_size)


    def __repr__(self):
        return f'HighRiskLowRiskTwoStageTesting(test = {self._test}, ' \
               f'pool_sizes = {self._pool_sizes}, threshold = {self._threshold})'