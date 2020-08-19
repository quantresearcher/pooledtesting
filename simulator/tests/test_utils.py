import numpy as np
from numpy.testing import assert_allclose
from unittest import TestCase
from ..utils import (
    count_not_none,
    freeze_iterable_arguments,
    ImpossibleDistribution,
    simulate_correlated_bernoulli,
)


NUM_SAMPLES = 10 ** 5
RANDOM_STATE = 42


class BernoulliSimulation(TestCase):

    def test_simulate_correlated_bernoulli_when_correlation_is_a_negative_number(self):
        with self.assertRaises(ValueError):
            simulate_correlated_bernoulli(p = [0.1, 0.2],
                                          correlation = -0.1,
                                          num_samples = NUM_SAMPLES,
                                          random_state = RANDOM_STATE)

    def test_simulate_correlated_bernoulli_when_correlation_is_a_matrix_containing_at_least_one_negative_number(self):
        with self.assertRaises(ValueError):
            simulate_correlated_bernoulli(p = [0.1, 0.2],
                                          correlation = [[1, -0.4], [-0.4, 1]],
                                          num_samples = NUM_SAMPLES,
                                          random_state = RANDOM_STATE)

    def test_simulate_correlated_bernoulli_when_p_and_correlation_have_different_lengths(self):
        p = np.array([0.05, 0.1, 0.15, 0.2])
        correlation_matrix = np.array([[1, 0.3, 0.2, 0.1],
                                       [0.3, 1, 0.3, 0.2],
                                       [0.2, 0.3, 1, 0.3],
                                       [0.1, 0.2, 0.3, 1]])
        with self.assertRaises(ValueError):
            simulate_correlated_bernoulli(p = p[:-1],
                                          correlation = correlation_matrix,
                                          num_samples = NUM_SAMPLES,
                                          random_state = RANDOM_STATE)
        with self.assertRaises(ValueError):
            simulate_correlated_bernoulli(p = p,
                                          correlation = correlation_matrix[:-1],
                                          num_samples = NUM_SAMPLES,
                                          random_state = RANDOM_STATE)
        with self.assertRaises(ValueError):
            simulate_correlated_bernoulli(p = p,
                                          correlation = correlation_matrix[:, :-1],
                                          num_samples = NUM_SAMPLES,
                                          random_state = RANDOM_STATE)

    def test_simulate_correlated_bernoulli_when_correlation_is_a_non_symmetric_matrix(self):
        non_symmetric_matrix = np.array([[1, 0.3, 0.2, 0.12],
                                         [0.3, 1, 0.3, 0.2],
                                         [0.2, 0.3, 1, 0.3],
                                         [0.1, 0.2, 0.3, 1]])
        with self.assertRaises(ValueError):
            simulate_correlated_bernoulli(p = np.array([0.05, 0.1, 0.15, 0.2]),
                                          correlation = non_symmetric_matrix,
                                          num_samples = NUM_SAMPLES,
                                          random_state = RANDOM_STATE)

    def test_simulate_correlated_bernoulli_when_correlation_is_a_matrix_that_is_not_positive_semidefinite(self):
        non_positive_semidefinite_matrix = np.array([[1, 0.9, 0.8, 0.1],
                                                     [0.9, 1, 0.3, 0.2],
                                                     [0.8, 0.3, 1, 0.3],
                                                     [0.1, 0.2, 0.3, 1]])
        with self.assertRaises(ValueError):
            simulate_correlated_bernoulli(p = np.array([0.05, 0.1, 0.15, 0.2]),
                                          correlation = non_positive_semidefinite_matrix,
                                          num_samples = NUM_SAMPLES,
                                          random_state = RANDOM_STATE)

    def test_simulate_correlated_bernoulli_when_correlation_is_zero(self):
        samples = simulate_correlated_bernoulli(p = [0.1, 0.2],
                                                correlation = 0,
                                                num_samples = NUM_SAMPLES,
                                                random_state = RANDOM_STATE)
        self.assertEqual(len(samples), NUM_SAMPLES)
        sample_means = samples.mean(axis = 0)
        self.assertAlmostEqual(sample_means[0], 0.1, places = 2)
        self.assertAlmostEqual(sample_means[1], 0.2, places = 2)
        sample_correlation_matrix = np.corrcoef(samples, rowvar = False)
        self.assertAlmostEqual(sample_correlation_matrix[0, 1], 0, places = 2)

    def test_simulate_correlated_bernoulli_when_correlation_is_a_positive_number(self):
        samples = simulate_correlated_bernoulli(p = np.array([0.1, 0.2]),
                                                correlation = 0.3,
                                                num_samples = NUM_SAMPLES,
                                                random_state = RANDOM_STATE)
        self.assertEqual(len(samples), NUM_SAMPLES)
        sample_means = samples.mean(axis = 0)
        self.assertAlmostEqual(sample_means[0], 0.1, places = 2)
        self.assertAlmostEqual(sample_means[1], 0.2, places = 2)
        sample_correlation_matrix = np.corrcoef(samples, rowvar = False)
        self.assertAlmostEqual(sample_correlation_matrix[0, 1], 0.3, places = 2)

    def test_simulate_correlated_bernoulli_when_a_correlation_matrix_is_provided_as_a_numpy_ndarray_and_p_and_correlation_matrix_are_not_homogenous(self):
        p = np.array([0.05, 0.1, 0.15, 0.2])
        correlation_matrix = np.array([[1, 0.3, 0.2, 0.1],
                                       [0.3, 1, 0.3, 0.2],
                                       [0.2, 0.3, 1, 0.3],
                                       [0.1, 0.2, 0.3, 1]])
        samples = simulate_correlated_bernoulli(p = p,
                                                correlation = correlation_matrix,
                                                num_samples = NUM_SAMPLES,
                                                random_state = RANDOM_STATE)
        self.assertEqual(len(samples), NUM_SAMPLES)
        sample_means = samples.mean(axis = 0)
        assert_allclose(p, sample_means, atol = 1e-2)
        sample_correlation_matrix = np.corrcoef(samples, rowvar = False)
        assert_allclose(correlation_matrix, sample_correlation_matrix, atol = 1e-2)

    def test_simulate_correlated_bernoulli_when_a_correlation_matrix_is_provided_as_a_numpy_ndarray_and_p_and_correlation_matrix_are_not_homogenous(self):
        p = np.array([0.1, 0.1, 0.1, 0.1])
        correlation_matrix = np.array([[1, 0.3, 0.3, 0.3],
                                       [0.3, 1, 0.3, 0.3],
                                       [0.3, 0.3, 1, 0.3],
                                       [0.3, 0.3, 0.3, 1]])
        samples = simulate_correlated_bernoulli(p = p,
                                                correlation = correlation_matrix,
                                                num_samples = NUM_SAMPLES,
                                                random_state = RANDOM_STATE)
        self.assertEqual(len(samples), NUM_SAMPLES)
        sample_means = samples.mean(axis = 0)
        assert_allclose(p, sample_means, atol = 1e-2)
        sample_correlation_matrix = np.corrcoef(samples, rowvar = False)
        assert_allclose(correlation_matrix, sample_correlation_matrix, atol = 1e-2)

    def test_simulate_correlated_bernoulli_when_a_correlation_matrix_is_provided_as_a_nested_list(self):
        p = [0.05, 0.1, 0.15, 0.2]
        correlation_matrix = [[1, 0.3, 0.2, 0.1],
                              [0.3, 1, 0.3, 0.2],
                              [0.2, 0.3, 1, 0.3],
                              [0.1, 0.2, 0.3, 1]]
        samples = simulate_correlated_bernoulli(p = p,
                                                correlation = correlation_matrix,
                                                num_samples = NUM_SAMPLES,
                                                random_state = RANDOM_STATE)
        self.assertEqual(len(samples), NUM_SAMPLES)
        sample_means = samples.mean(axis = 0)
        assert_allclose(p, sample_means, atol = 1e-2)
        sample_correlation_matrix = np.corrcoef(samples, rowvar = False)
        assert_allclose(correlation_matrix, sample_correlation_matrix, atol = 1e-2)

    def test_simulate_correlated_bernoulli_when_the_inputs_specify_an_impossible_multivariate_distribution(self):
        p = np.array([0.05, 0.05, 0.001, 0.05, 0.001, 0.001])
        correlation_matrix = np.array([[   1, 0.25, 0.25,    0,    0,    0],
                                       [0.25,    1, 0.25,    0,    0,    0],
                                       [0.25, 0.25,    1,    0,    0,    0],
                                       [   0,    0,    0,    1, 0.25, 0.25],
                                       [   0,    0,    0, 0.25,    1, 0.25],
                                       [   0,    0,    0, 0.25, 0.25,    1]])
        with self.assertRaises(ImpossibleDistribution):
            simulate_correlated_bernoulli(p = p,
                                          correlation = correlation_matrix,
                                          num_samples = NUM_SAMPLES,
                                          random_state = RANDOM_STATE)


class MiscellaneousTests(TestCase):

    def test_count_not_none(self):
        self.assertEqual(count_not_none(None), 0)
        self.assertEqual(count_not_none('test'), 1)
        self.assertEqual(count_not_none(None, 'test'), 1)
        self.assertEqual(count_not_none('test', None, 42), 2)

    def test_freeze_iterable_arguments(self):
        @freeze_iterable_arguments
        def func(iterable1, non_iterable, iterable2, iterable3):
            return iterable1, non_iterable, iterable2, iterable3

        iterable1, non_iterable, iterable2, iterable3 = func([2, 3], 4, set([5, 6, 7]), np.array([8, 9]))
        self.assertEqual(iterable1, tuple([2, 3]))
        self.assertEqual(non_iterable, 4)
        self.assertIsInstance(iterable2, tuple)
        self.assertEqual(len(iterable2), 3)
        self.assertIn(5, iterable2)
        self.assertIn(6, iterable2)
        self.assertIn(7, iterable2)
        self.assertEqual(iterable3, tuple([8, 9]))