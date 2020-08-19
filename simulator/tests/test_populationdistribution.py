import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from unittest import TestCase
from ..populationdistribution import (
    ImpossiblePopulationDistribution,
    PopulationDistribution
)


RANDOM_STATE = 42
NUM_SAMPLES = 10 ** 6


class PopulationDistributionTests(TestCase):

    def test_when_none_of_probabilities_or_prevalence_or_population_are_given(self):
        with self.assertRaises(TypeError):
            PopulationDistribution(num_samples = NUM_SAMPLES,
                                   random_state = RANDOM_STATE)

    def test_when_probabilities_is_given_and_at_least_one_of_prevalence_and_population_is_given(self):
        with self.assertRaises(TypeError):
            PopulationDistribution(probabilities = [0.1, 0.2],
                                   prevalence = 0.6,
                                   num_samples = NUM_SAMPLES,
                                   random_state = RANDOM_STATE)
        with self.assertRaises(TypeError):
            PopulationDistribution(probabilities = [0.1, 0.2],
                                   population = 2,
                                   num_samples = NUM_SAMPLES,
                                   random_state = RANDOM_STATE)
        with self.assertRaises(TypeError):
            PopulationDistribution(probabilities = [0.1, 0.2],
                                   prevalence = 0.6,
                                   population = 2,
                                   num_samples = NUM_SAMPLES,
                                   random_state = RANDOM_STATE)
    
    def test_when_probabilities_is_not_given_and_only_one_of_prevalence_and_population_is_given(self):
        with self.assertRaises(TypeError):
            PopulationDistribution(prevalence = 0.6,
                                   num_samples = NUM_SAMPLES,
                                   random_state = RANDOM_STATE)
        with self.assertRaises(TypeError):
            PopulationDistribution(population = 2,
                                   num_samples = NUM_SAMPLES,
                                   random_state = RANDOM_STATE)
        
    def test_when_more_than_one_of_correlation_input_type_is_given(self):
        with self.assertRaises(TypeError):
            PopulationDistribution(probabilities = [0.1, 0.2],
                                   correlation = 0.3,
                                   correlation_matrix = np.eye(2),
                                   cluster_correlations = [[0.5, 2]],
                                   num_samples = NUM_SAMPLES,
                                   random_state = RANDOM_STATE)
        with self.assertRaises(TypeError):
            PopulationDistribution(probabilities = [0.1, 0.2],
                                   correlation = 0.3,
                                   correlation_matrix = np.eye(2),
                                   num_samples = NUM_SAMPLES,
                                   random_state = RANDOM_STATE)
        with self.assertRaises(TypeError):
            PopulationDistribution(probabilities = [0.1, 0.2],
                                   correlation = 0.3,
                                   cluster_correlations = [[0.5, 2]],
                                   num_samples = NUM_SAMPLES,
                                   random_state = RANDOM_STATE)
        with self.assertRaises(TypeError):
            PopulationDistribution(probabilities = [0.1, 0.2],
                                   correlation_matrix = np.eye(2),
                                   cluster_correlations = [[0.5, 2]],
                                   num_samples = NUM_SAMPLES,
                                   random_state = RANDOM_STATE)

    def test_when_correlation_is_negative(self):
        """This Exception isn't explicitly handled in the
        PopulationDistribution() constructor.
        """
        with self.assertRaises(ValueError):
            PopulationDistribution(probabilities = [0.1, 0.2],
                                   correlation = -0.1,
                                   num_samples = NUM_SAMPLES,
                                   random_state = RANDOM_STATE)
        correlation_matrix = [[1, -0.5],
                              [-0.5, 1]]
        with self.assertRaises(ValueError):
            PopulationDistribution(probabilities = [0.1, 0.2],
                                   correlation_matrix = correlation_matrix,
                                   num_samples = NUM_SAMPLES,
                                   random_state = RANDOM_STATE)
        with self.assertRaises(ValueError):
            PopulationDistribution(probabilities = [0.1, 0.2],
                                   cluster_correlations = [(-0.5, 2)],
                                   num_samples = NUM_SAMPLES,
                                   random_state = RANDOM_STATE)

    def test_when_probabilities_is_given_and_prevalence_and_population_are_not_and_correlations_are_not_given(self):
        probabilities = [0.1, 0.3]
        population_distribution = PopulationDistribution(probabilities = probabilities,
                                                         num_samples = NUM_SAMPLES,
                                                         random_state = RANDOM_STATE)
        assert_array_equal(population_distribution.probabilities, probabilities)
        self.assertIsNone(population_distribution.cluster_correlations)
        assert_array_equal(population_distribution.correlation_matrix, np.eye(2))
        self.assertEqual(population_distribution.population, 2)
        self.assertEqual(population_distribution.num_samples, NUM_SAMPLES)
        samples = population_distribution.samples
        for mean, prob in zip(samples.mean(axis = 0), probabilities):
            self.assertAlmostEqual(mean, prob, places = 2)
        sample_correlation_matrix = np.corrcoef(samples, rowvar = False)
        self.assertAlmostEqual(sample_correlation_matrix[0, 1], 0, places = 2)
        samples_generator = iter(population_distribution)
        assert_array_equal(samples[0], next(samples_generator))
        assert_array_equal(samples[1], next(samples_generator))
        for sample in samples_generator:
            pass
        assert_array_equal(samples[-1], sample)

    def test_when_probabilities_is_given_but_prevalence_and_population_are_and_correlation_is_not_given(self):
        population_distribution = PopulationDistribution(prevalence = 0.6,
                                                         population = 3,
                                                         num_samples = NUM_SAMPLES,
                                                         random_state = RANDOM_STATE)
        assert_array_equal(population_distribution.probabilities, [0.6] * 3)
        self.assertIsNone(population_distribution.cluster_correlations)
        self.assertEqual(population_distribution.population, 3)
        self.assertEqual(population_distribution.num_samples, NUM_SAMPLES)
        samples = population_distribution.samples
        sample_means = samples.mean(axis = 0)
        for mean in samples.mean(axis = 0):
            self.assertAlmostEqual(mean, 0.6, places = 2)
        sample_correlation_matrix = np.corrcoef(samples, rowvar = False)
        expected_correlation_matrix = np.eye(3)
        assert_allclose(sample_correlation_matrix, expected_correlation_matrix, atol = 1e-2)
        assert_allclose(population_distribution.correlation_matrix, expected_correlation_matrix)

    def test_when_probabilities_is_given_and_prevalence_and_population_are_not_and_correlation_is_zero(self):
        probabilities = [0.1, 0.3]
        population_distribution = PopulationDistribution(probabilities = probabilities,
                                                         correlation = 0,
                                                         num_samples = NUM_SAMPLES,
                                                         random_state = RANDOM_STATE)
        assert_array_equal(population_distribution.probabilities, probabilities)
        self.assertIsNone(population_distribution.cluster_correlations)
        self.assertEqual(population_distribution.population, 2)
        self.assertEqual(population_distribution.num_samples, NUM_SAMPLES)
        samples = population_distribution.samples
        for mean, prob in zip(samples.mean(axis = 0), probabilities):
            self.assertAlmostEqual(mean, prob, places = 2)
        sample_correlation_matrix = np.corrcoef(samples, rowvar = False)
        expected_correlation_matrix = np.eye(2)
        assert_allclose(sample_correlation_matrix, expected_correlation_matrix, atol = 1e-2)
        assert_allclose(population_distribution.correlation_matrix, expected_correlation_matrix)

    def test_when_probabilities_is_given_and_prevalence_and_population_are_not_and_cluster_correlations_is_empty_list(self):
        probabilities = [0.1, 0.3]
        population_distribution = PopulationDistribution(probabilities = probabilities,
                                                         cluster_correlations = [],
                                                         num_samples = NUM_SAMPLES,
                                                         random_state = RANDOM_STATE)
        assert_array_equal(population_distribution.probabilities, probabilities)
        self.assertIsNone(population_distribution.cluster_correlations)
        self.assertEqual(population_distribution.population, 2)
        self.assertEqual(population_distribution.num_samples, NUM_SAMPLES)
        samples = population_distribution.samples
        for mean, prob in zip(samples.mean(axis = 0), probabilities):
            self.assertAlmostEqual(mean, prob, places = 2)
        sample_correlation_matrix = np.corrcoef(samples, rowvar = False)
        expected_correlation_matrix = np.eye(2)
        assert_allclose(sample_correlation_matrix, expected_correlation_matrix, atol = 1e-2)
        assert_allclose(population_distribution.correlation_matrix, expected_correlation_matrix)

    def test_case_where_probabilities_is_given_and_correlation_is_given(self):
        probabilities = [0.1, 0.2, 0.3]
        population_distribution = PopulationDistribution(probabilities = probabilities,
                                                         correlation = 0.4,
                                                         num_samples = NUM_SAMPLES,
                                                         random_state = RANDOM_STATE)
        assert_array_equal(population_distribution.probabilities, probabilities)
        assert_array_equal(population_distribution.cluster_correlations, [(0.4, 3)])
        self.assertEqual(population_distribution.population, 3)
        self.assertEqual(population_distribution.num_samples, NUM_SAMPLES)
        samples = population_distribution.samples
        for mean, prob in zip(samples.mean(axis = 0), probabilities):
            self.assertAlmostEqual(mean, prob, places = 2)
        sample_correlation_matrix = np.corrcoef(samples, rowvar = False)
        expected_correlation_matrix = np.array([[ 1,  0.4, 0.4],
                                                [0.4,   1, 0.4],
                                                [0.4, 0.4,   1]])
        assert_allclose(sample_correlation_matrix, expected_correlation_matrix, atol = 1e-2)
        assert_allclose(population_distribution.correlation_matrix, expected_correlation_matrix)

    def test_case_where_probabilities_is_given_and_identity_correlation_matrix_is_given(self):
        probabilities = [0.1, 0.2, 0.3]
        correlation_matrix = np.eye(3)
        population_distribution = PopulationDistribution(probabilities = probabilities,
                                                         correlation_matrix = correlation_matrix,
                                                         num_samples = NUM_SAMPLES,
                                                         random_state = RANDOM_STATE)
        assert_array_equal(population_distribution.probabilities, probabilities)
        self.assertIsNone(population_distribution.cluster_correlations)
        self.assertEqual(population_distribution.population, 3)
        self.assertEqual(population_distribution.num_samples, NUM_SAMPLES)
        samples = population_distribution.samples
        for mean, prob in zip(samples.mean(axis = 0), probabilities):
            self.assertAlmostEqual(mean, prob, places = 2)
        sample_correlation_matrix = np.corrcoef(samples, rowvar = False)
        assert_allclose(sample_correlation_matrix, correlation_matrix, atol = 1e-2)
        assert_allclose(population_distribution.correlation_matrix, correlation_matrix)

    def test_case_where_probabilities_is_given_and_non_identity_correlation_matrix_is_given(self):
        probabilities = [0.1, 0.2, 0.3]
        correlation_matrix = np.array([[ 1,  0.4, 0.5],
                                       [0.4,   1, 0.6],
                                       [0.5, 0.6,   1]])
        population_distribution = PopulationDistribution(probabilities = probabilities,
                                                         correlation_matrix = correlation_matrix,
                                                         num_samples = NUM_SAMPLES,
                                                         random_state = RANDOM_STATE)
        assert_array_equal(population_distribution.probabilities, probabilities)
        self.assertIsNone(population_distribution.cluster_correlations)
        self.assertEqual(population_distribution.population, 3)
        self.assertEqual(population_distribution.num_samples, NUM_SAMPLES)
        samples = population_distribution.samples
        for mean, prob in zip(samples.mean(axis = 0), probabilities):
            self.assertAlmostEqual(mean, prob, places = 2)
        sample_correlation_matrix = np.corrcoef(samples, rowvar = False)
        assert_allclose(sample_correlation_matrix, correlation_matrix, atol = 1e-2)
        assert_allclose(population_distribution.correlation_matrix, correlation_matrix)

    def test_case_where_probabilities_is_given_and_cluster_correlations_has_same_number_of_elements_as_probabilities(self):
        probabilities = [0.1, 0.2, 0.3]
        population_distribution = PopulationDistribution(probabilities = probabilities,
                                                         cluster_correlations = np.array([[0.4, 3]]),
                                                         num_samples = NUM_SAMPLES,
                                                         random_state = RANDOM_STATE)
        assert_array_equal(population_distribution.probabilities, probabilities)
        assert_array_equal(population_distribution.cluster_correlations, [[0.4, 3]])
        self.assertEqual(population_distribution.population, 3)
        self.assertEqual(population_distribution.num_samples, NUM_SAMPLES)
        samples = population_distribution.samples
        for mean, prob in zip(samples.mean(axis = 0), probabilities):
            self.assertAlmostEqual(mean, prob, places = 2)
        sample_correlation_matrix = np.corrcoef(samples, rowvar = False)
        expected_correlation_matrix = np.array([[  1, 0.4, 0.4],
                                                [0.4,   1, 0.4],
                                                [0.4, 0.4,   1]])
        assert_allclose(sample_correlation_matrix, expected_correlation_matrix, atol = 1e-2)
        assert_allclose(population_distribution.correlation_matrix, expected_correlation_matrix)

    def test_case_where_probabilities_is_given_and_cluster_correlations_has_less_elements_than_probabilities(self):
        probabilities = [0.1, 0.2, 0.3]
        population_distribution = PopulationDistribution(probabilities = probabilities,
                                                         cluster_correlations = [[0.4, 2]],
                                                         num_samples = NUM_SAMPLES,
                                                         random_state = RANDOM_STATE)
        assert_array_equal(population_distribution.probabilities, probabilities)
        assert_array_equal(population_distribution.cluster_correlations, [[0.4, 2]])
        self.assertEqual(population_distribution.population, 3)
        self.assertEqual(population_distribution.num_samples, NUM_SAMPLES)
        samples = population_distribution.samples
        for mean, prob in zip(samples.mean(axis = 0), probabilities):
            self.assertAlmostEqual(mean, prob, places = 2)
        sample_correlation_matrix = np.corrcoef(samples, rowvar = False)
        expected_correlation_matrix = np.array([[  1, 0.4,  0],
                                                [0.4,   1,  0],
                                                [  0,   0,  1]])
        assert_allclose(sample_correlation_matrix, expected_correlation_matrix, atol = 1e-2)
        assert_allclose(population_distribution.correlation_matrix, expected_correlation_matrix)

    def test_case_where_probabilities_is_given_and_cluster_correlations_has_less_elements_than_probabilities_and_cluster_of_one(self):
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        population_distribution = PopulationDistribution(probabilities = probabilities,
                                                         cluster_correlations = [(0.15, 3), (0.25, 1), (0.35, 2)],
                                                         num_samples = NUM_SAMPLES,
                                                         random_state = RANDOM_STATE)
        assert_array_equal(population_distribution.probabilities, probabilities)
        assert_array_equal(population_distribution.cluster_correlations, [(0.15, 3), (0.25, 1), (0.35, 2)])
        self.assertEqual(population_distribution.population, 7)
        self.assertEqual(population_distribution.num_samples, NUM_SAMPLES)
        samples = population_distribution.samples
        for mean, prob in zip(samples.mean(axis = 0), probabilities):
            self.assertAlmostEqual(mean, prob, places = 2)
        sample_correlation_matrix = np.corrcoef(samples, rowvar = False)
        expected_correlation_matrix = np.array([[   1, 0.15, 0.15,   0,    0,    0,   0],
                                                [0.15,    1, 0.15,   0,    0,    0,   0],
                                                [0.15, 0.15,    1,   0,    0,    0,   0],
                                                [   0,    0,    0,   1,    0,    0,   0],
                                                [   0,    0,    0,   0,    1, 0.35,   0],
                                                [   0,    0,    0,   0, 0.35,    1,   0],
                                                [   0,    0,    0,   0,    0,    0,   1]])
        assert_allclose(sample_correlation_matrix, expected_correlation_matrix, atol = 1e-2)
        assert_allclose(population_distribution.correlation_matrix, expected_correlation_matrix)

    def test_when_inputs_specify_an_impossible_population_distribution(self):
        with self.assertRaises(ImpossiblePopulationDistribution):
            population_distribution = PopulationDistribution(probabilities = [0.05, 0.05, 0.001, 0.05, 0.001, 0.001],
                                                             cluster_correlations = [(0.25, 3), (0.25, 3)],
                                                             num_samples = NUM_SAMPLES,
                                                             random_state = RANDOM_STATE)