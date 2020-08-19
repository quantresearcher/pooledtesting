import numpy as np
from unittest import TestCase
from ..populationdistribution import PopulationDistribution
from ..testingstrategy import (
    HighRiskLowRiskTwoStageTesting,
    Test,
    ThreeStageTesting,
    TwoDimensionalArrayTesting,
    TwoStageTesting,
)


NUM_TRIALS = 10 ** 5
RANDOM_STATE = 42


class TestTests(TestCase):

    def setUp(self):
        self.test = Test(sensitivity = 0.9,
                         specificity = 0.8,
                         random_state = RANDOM_STATE)

    def test_string_representation_does_not_raise_exception(self):
        str(self.test)

    def test_Test_when_pool_is_infected(self):
        results = [self.test.conduct([1])[0] for _ in range(NUM_TRIALS)]
        number_of_tests = sum(results) / NUM_TRIALS
        self.assertAlmostEqual(number_of_tests, self.test.sensitivity, places = 2)

    def test_Test_when_pool_is_not_infected(self):
        results = [self.test.conduct([0])[0] for _ in range(NUM_TRIALS)]
        number_of_tests = sum(results) / NUM_TRIALS
        self.assertAlmostEqual(number_of_tests, 1 - self.test.specificity, places = 2)

    def test_Test_when_pool_is_infected(self):
        pools = np.array([[1, 0],
                          [0, 1]])
        results11 = []
        results12 = []
        results21 = []
        results22 = []
        for _ in range(NUM_TRIALS):
            results = self.test.conduct(pools)
            results11.append(results[0, 0])
            results12.append(results[0, 1])
            results21.append(results[1, 0])
            results22.append(results[1, 1])
        self.assertAlmostEqual(sum(results11) / NUM_TRIALS, self.test.sensitivity, places = 2)
        self.assertAlmostEqual(sum(results22) / NUM_TRIALS, self.test.sensitivity, places = 2)
        self.assertAlmostEqual(sum(results12) / NUM_TRIALS, 1 - self.test.specificity, places = 2)
        self.assertAlmostEqual(sum(results21) / NUM_TRIALS, 1 - self.test.specificity, places = 2)
    

class TwoStageTestingTests(TestCase):

    def setUp(self):
        self.test = Test(sensitivity = 0.95,
                         specificity = 0.99,
                         random_state = RANDOM_STATE)
        self.strategy = TwoStageTesting(test = self.test,
                                        pool_size = 4)

    def test_string_representation_does_not_raise_exception(self):
        str(self.strategy)

    def test_when_pool_size_is_one(self):
        strategy = TwoStageTesting(test = self.test,
                                   pool_size = 1)
        population_distribution = PopulationDistribution(prevalence = 0.03,
                                                         population = 3,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        strategy.run(population_distribution)
        total_tests_conducted = strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertEqual(number_of_tests_per_person, 1)
        self.assertEqual(strategy.test, self.test)

    def test_when_population_is_less_than_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.03,
                                                         population = 3,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        with self.assertRaises(ValueError):
            self.strategy.run(population_distribution)

    def test_when_population_is_equal_to_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.03,
                                                         population = 4,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.37, places = 2)
        self.assertEqual(self.strategy.test, self.test)

    def test_when_population_is_a_multiple_of_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.03,
                                                         population = 12,
                                                         correlation = 0.5,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.318, places = 2)

    def test_when_population_is_not_a_multiple_of_pool_size_and_remainder_is_greater_than_one(self):
        population_distribution = PopulationDistribution(prevalence = 0.03,
                                                         population = 10,
                                                         correlation = 0.5,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.365, places = 2)

    def test_when_population_is_not_a_multiple_of_pool_size_and_remainder_is_one(self):
        population_distribution = PopulationDistribution(prevalence = 0.03,
                                                         population = 9,
                                                         correlation = 0.5,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.39, places = 2)


class ThreeStageTestingTests(TestCase):

    def setUp(self):
        self.test = Test(sensitivity = 0.95,
                         specificity = 0.99,
                         random_state = RANDOM_STATE)
        self.strategy = ThreeStageTesting(test = self.test,
                                          pool_sizes = [2, 3, 7])

    def test_string_representation_does_not_raise_exception(self):
        str(self.strategy)

    def test_when_stage_2_pool_sizes_are_all_one(self):
        """This is equivalent to two stage testing."""
        strategy = ThreeStageTesting(test = self.test,
                                     pool_sizes = [1, 1, 1, 1])
        population_distribution = PopulationDistribution(prevalence = 0.03,
                                                         population = 12,
                                                         correlation = 0.5,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        strategy.run(population_distribution)
        total_tests_conducted = strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.318, places = 2)
        self.assertEqual(strategy.test, self.test)

    def test_when_population_is_less_than_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.03,
                                                         population = 10,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        with self.assertRaises(ValueError):
            self.strategy.run(population_distribution)

    def test_when_population_is_equal_to_stage_1_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.03,
                                                         population = 12,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.29, places = 2)
        self.assertEqual(self.strategy.test, self.test)

    def test_when_population_is_a_multiple_of_stage_1_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.03,
                                                         population = 24,
                                                         correlation = 0.5,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.169, places = 2)

    def test_when_population_is_a_multiple_of_stage_1_pool_size_and_at_least_one_stage2_pool_size_is_1(self):
        strategy = ThreeStageTesting(test = self.test,
                                     pool_sizes = [3, 2, 1, 1])
        population_distribution = PopulationDistribution(prevalence = 0.03,
                                                         population = 21,
                                                         correlation = 0.5,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        strategy.run(population_distribution)
        total_tests_conducted = strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.22, places = 2)

    def test_when_population_is_not_a_multiple_of_stage_1_pool_size_and_remainder_is_less_than_first_stage2_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.3,
                                                         population = 25,
                                                         correlation = 0.5,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.72, places = 2)

    def test_when_population_is_not_a_multiple_of_stage_1_pool_size_and_remainder_is_equal_to_first_stage2_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.3,
                                                         population = 26,
                                                         correlation = 0.5,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.74, places = 2)

    def test_when_population_is_not_a_multiple_of_stage_1_pool_size_and_remainder_is_between_first_and_second_stage2_pool_sizes(self):
        population_distribution = PopulationDistribution(prevalence = 0.3,
                                                         population = 27,
                                                         correlation = 0.5,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.73, places = 2)

    def test_when_population_is_not_a_multiple_of_stage_1_pool_size_and_remainder_is_equal_to_sum_of_first_two_stage2_pool_sizes(self):
        population_distribution = PopulationDistribution(prevalence = 0.3,
                                                         population = 29,
                                                         correlation = 0.5,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.72, places = 2)

    def test_when_population_is_not_a_multiple_of_stage_1_pool_size_and_remainder_is_one_more_than_sum_of_first_two_stage2_pool_sizes_but_more_than_one_less_than_sum_of_all_stage2_pool_sizes(self):
        population_distribution = PopulationDistribution(prevalence = 0.3,
                                                         population = 30,
                                                         correlation = 0.5,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.72, places = 2)

    def test_when_population_is_not_a_multiple_of_stage_1_pool_size_and_remainder_is_one_less_than_sum_of_all_stage2_pool_sizes(self):
        population_distribution = PopulationDistribution(prevalence = 0.3,
                                                         population = 35,
                                                         correlation = 0.5,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.71, places = 2)


class TwoDimensionalArrayTestingWithSquareGridTestsWithoutTestingMasterPools(TestCase):

    def setUp(self):
        self.test = Test(sensitivity = 0.95,
                         specificity = 0.99,
                         random_state = RANDOM_STATE)
        self.strategy = TwoDimensionalArrayTesting(test = self.test,
                                                   num_rows = 4,
                                                   num_columns = 4,
                                                   test_master_pools = False)

    def test_string_representation_does_not_raise_exception(self):
        str(self.strategy)

    def test_when_population_is_less_than_grid_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.1,
                                                         population = 12,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        with self.assertRaises(ValueError):
            self.strategy.run(population_distribution)

    def test_when_population_is_equal_to_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.1,
                                                         population = 16,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.63, places = 2)
        self.assertEqual(self.strategy.test, self.test)

    def test_when_population_is_a_multiple_of_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.1,
                                                         population = 48,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.63, places = 2)

    def test_when_population_is_not_a_multiple_of_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.1,
                                                         population = 20,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.70, places = 2)


class TwoDimensionalArrayTestingWithNonSquareGridTestsWithoutTestingMasterPools(TestCase):

    def setUp(self):
        self.test = Test(sensitivity = 0.95,
                         specificity = 0.99,
                         random_state = RANDOM_STATE)
        self.strategy = TwoDimensionalArrayTesting(test = self.test,
                                                   num_rows = 3,
                                                   num_columns = 4,
                                                   test_master_pools = False)

    def test_string_representation_does_not_raise_exception(self):
        str(self.strategy)

    def test_when_population_is_less_than_grid_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.1,
                                                         population = 10,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        with self.assertRaises(ValueError):
            self.strategy.run(population_distribution)

    def test_when_population_is_equal_to_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.1,
                                                         population = 12,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.69, places = 2)
        self.assertEqual(self.strategy.test, self.test)

    def test_when_population_is_a_multiple_of_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.1,
                                                         population = 36,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.69, places = 2)

    def test_when_population_is_not_a_multiple_of_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.1,
                                                         population = 30,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.75, places = 2)


class TwoDimensionalArrayTestingWithSquareGridTestsWithTestingMasterPools(TestCase):

    def setUp(self):
        self.test = Test(sensitivity = 0.95,
                         specificity = 0.99,
                         random_state = RANDOM_STATE)
        self.strategy = TwoDimensionalArrayTesting(test = self.test,
                                                   num_rows = 4,
                                                   num_columns = 4,
                                                   test_master_pools = True)

    def test_string_representation_does_not_raise_exception(self):
        str(self.strategy)

    def test_when_population_is_less_than_grid_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.1,
                                                         population = 12,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        with self.assertRaises(ValueError):
            self.strategy.run(population_distribution)

    def test_when_population_is_equal_to_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.1,
                                                         population = 16,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.57, places = 2)
        self.assertEqual(self.strategy.test, self.test)

    def test_when_population_is_a_multiple_of_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.1,
                                                         population = 48,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.57, places = 2)

    def test_when_population_is_not_a_multiple_of_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.1,
                                                         population = 20,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.65, places = 2)


class TwoDimensionalArrayTestingWithNonSquareGridTestsWithTestingMasterPools(TestCase):

    def setUp(self):
        self.test = Test(sensitivity = 0.95,
                         specificity = 0.99,
                         random_state = RANDOM_STATE)
        self.strategy = TwoDimensionalArrayTesting(test = self.test,
                                                   num_rows = 3,
                                                   num_columns = 4,
                                                   test_master_pools = True)

    def test_string_representation_does_not_raise_exception(self):
        str(self.strategy)

    def test_when_population_is_less_than_grid_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.1,
                                                         population = 10,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        with self.assertRaises(ValueError):
            self.strategy.run(population_distribution)

    def test_when_population_is_equal_to_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.1,
                                                         population = 12,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.57, places = 2)
        self.assertEqual(self.strategy.test, self.test)

    def test_when_population_is_a_multiple_of_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.1,
                                                         population = 36,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.57, places = 2)

    def test_when_population_is_not_a_multiple_of_pool_size(self):
        population_distribution = PopulationDistribution(prevalence = 0.1,
                                                         population = 30,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        self.strategy.run(population_distribution)
        total_tests_conducted = self.strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.66, places = 2)


class HighRiskLowRiskTwoStageTestingTests(TestCase):

    def setUp(self):
        self.test = Test(sensitivity = 1,
                         specificity = 1,
                         random_state = RANDOM_STATE)
        
    def test_number_of_tests_per_person_when_there_are_no_correlations(self):
        population_distribution = PopulationDistribution(probabilities = [0.001] * 100 + [0.05] * 100,
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        strategy = HighRiskLowRiskTwoStageTesting(test = self.test,
                                                  pool_sizes = [34, 5],
                                                  threshold = 0.01)
        str(strategy) # test that no exception is thrown
        strategy.run(population_distribution)
        total_tests_conducted = strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.24, places = 2)
        self.assertEqual(strategy.test, self.test)

    def test_number_of_tests_per_person_when_there_are_correlations(self):
        population_distribution = PopulationDistribution(probabilities = [0.001] * 100 + [0.05] * 100,
                                                         cluster_correlations = [(0.25, 100), (0.25, 100)],
                                                         num_samples = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
        strategy = HighRiskLowRiskTwoStageTesting(test = self.test,
                                                  pool_sizes = [40, 10],
                                                  threshold = 0.01)
        str(strategy) # test that no exception is thrown
        strategy.run(population_distribution)
        total_tests_conducted = strategy.total_tests_conducted
        number_of_tests_per_person = total_tests_conducted / (population_distribution.population * NUM_TRIALS)
        self.assertAlmostEqual(number_of_tests_per_person, 0.18, places = 2)
        self.assertEqual(strategy.test, self.test)




if __name__ == '__main__':
    from unittest import main
    main()