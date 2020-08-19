from unittest import TestCase
from .utils import (
    get_number_of_tests_exact,
    get_number_of_tests_simulated
)


NUM_TRIALS = 10 ** 5
RANDOM_STATE = 42


class PooledTestingTests(TestCase):

    def test_get_number_of_tests_exact_when_pool_sizes_is_array_like(self):
        expected_number_of_tests_per_person = get_number_of_tests_exact(prevalence = 0.03,
                                                                        sensitivity = 0.95,
                                                                        specificity = 0.99,
                                                                        pool_sizes = [3, 15])
        self.assertAlmostEqual(expected_number_of_tests_per_person[3], 0.4254, places = 4)
        self.assertAlmostEqual(expected_number_of_tests_per_person[15], 0.4214, places = 4)

    def test_get_number_of_tests_exact_when_pool_sizes_is_an_int(self):
        expected_number_of_tests_per_person = get_number_of_tests_exact(prevalence = 0.99,
                                                                        sensitivity = 0.6,
                                                                        specificity = 0.5,
                                                                        pool_sizes = 17)
        self.assertAlmostEqual(expected_number_of_tests_per_person[17], 0.6588, places = 4)

    def test_get_number_of_tests_simulated_when_pool_sizes_is_array_like_and_correlation_is_zero(self):
        expected_number_of_tests_per_person = get_number_of_tests_simulated(prevalence = 0.03,
                                                                            correlation = 0,
                                                                            sensitivity = 0.95,
                                                                            specificity = 0.99,
                                                                            pool_sizes = [3, 15],
                                                                            num_trials = NUM_TRIALS,
                                                                            random_state = RANDOM_STATE)
        self.assertAlmostEqual(expected_number_of_tests_per_person[3], 0.4254, places = 2)
        self.assertAlmostEqual(expected_number_of_tests_per_person[15], 0.4214, places = 2)

        # calling function again with same arguments except for `pool_sizes` in order to test `_memoise` decorator
        expected_number_of_tests_per_person = get_number_of_tests_simulated(prevalence = 0.03,
                                                                            correlation = 0,
                                                                            sensitivity = 0.95,
                                                                            specificity = 0.99,
                                                                            pool_sizes = [3, 17],
                                                                            num_trials = NUM_TRIALS,
                                                                            random_state = RANDOM_STATE)
        self.assertAlmostEqual(expected_number_of_tests_per_person[3], 0.4254, places = 2)
        self.assertAlmostEqual(expected_number_of_tests_per_person[17], 0.449, places = 2)
    
    def test_get_number_of_tests_simulated_when_pool_sizes_is_an_int_and_correlation_is_zero(self):
        expected_number_of_tests_per_person = get_number_of_tests_simulated(prevalence = 0.99,
                                                                            correlation = 0,
                                                                            sensitivity = 0.6,
                                                                            specificity = 0.5,
                                                                            pool_sizes = 17,
                                                                            num_trials = NUM_TRIALS,
                                                                            random_state = RANDOM_STATE)
        self.assertAlmostEqual(expected_number_of_tests_per_person[17], 0.6588, places = 2)

    def test_get_number_of_tests_simulated_when_pool_sizes_is_array_like_and_correlation_is_non_zero(self):
        expected_number_of_tests_per_person = get_number_of_tests_simulated(prevalence = 0.03,
                                                                            correlation = 0.5,
                                                                            sensitivity = 0.95,
                                                                            specificity = 0.99,
                                                                            pool_sizes = [3, 15],
                                                                            num_trials = NUM_TRIALS,
                                                                            random_state = RANDOM_STATE)
        self.assertAlmostEqual(expected_number_of_tests_per_person[3], 0.395, places = 2)
        self.assertAlmostEqual(expected_number_of_tests_per_person[15], 0.173, places = 2)

    def test_get_number_of_tests_simulated_when_pool_sizes_is_an_int_and_correlation_is_non_zero(self):
        expected_number_of_tests_per_person = get_number_of_tests_simulated(prevalence = 0.03,
                                                                            correlation = 0.5,
                                                                            sensitivity = 0.95,
                                                                            specificity = 0.99,
                                                                            pool_sizes = 17,
                                                                            num_trials = NUM_TRIALS,
                                                                            random_state = RANDOM_STATE)
        self.assertAlmostEqual(expected_number_of_tests_per_person[17], 0.169, places = 2)