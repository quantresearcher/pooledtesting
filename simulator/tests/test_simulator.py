import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal
from time import time
from unittest import TestCase
from ..populationdistribution import PopulationDistribution
from ..simulator import Simulator
from ..testingstrategy import Test, TwoStageTesting


RANDOM_STATE = 42
NUM_TRIALS = 10 ** 4


class SimulatorTests(TestCase):

    def setUp(self):
        self.population_distribution = PopulationDistribution(prevalence = 0.01,
                                                              population = 24,
                                                              correlation = 0.5,
                                                              num_samples = NUM_TRIALS,
                                                              random_state = RANDOM_STATE)
        self.test = Test(sensitivity = 0.95,
                         specificity = 0.99,
                         random_state = RANDOM_STATE)

    def test_Simulator_when_a_single_strategy_is_supplied(self):
        strategy = TwoStageTesting(test = self.test, pool_size = 4)
        simulator = Simulator(population_distribution = self.population_distribution,
                              strategies = strategy)
        start_time = time()
        simulator.run()
        end_time = time()
        self.assertAlmostEqual(simulator.run_time, end_time - start_time, places = 2)
        tests_per_person = simulator.tests_per_person
        assert_series_equal(tests_per_person,
                            pd.Series(data = [0.28],
                                      index = [str(strategy)]),
                            check_index_type = False,
                            atol = 1e-2)
        ax = simulator.plot()
        y = ax.lines[-1].get_xydata()[:, 1]
        assert_array_equal(y, tests_per_person)

    def test_Simulator_when_a_list_of_strategies_is_supplied(self):
        strategies = [
            TwoStageTesting(test = self.test, pool_size = 3),
            TwoStageTesting(test = self.test, pool_size = 4),
            TwoStageTesting(test = self.test, pool_size = 2)
        ]
        simulator = Simulator(population_distribution = self.population_distribution,
                              strategies = strategies)
        for max_workers in [1, 4]:
            start_time = time()
            simulator.run(max_workers = max_workers)
            end_time = time()
            self.assertAlmostEqual(simulator.run_time, end_time - start_time, places = 2)
            tests_per_person = simulator.tests_per_person
            assert_series_equal(tests_per_person,
                                pd.Series(data = [0.361, 0.28, 0.524],
                                          index = [str(strategy) for strategy in strategies]),
                                check_index_type = False,
                                atol = 1e-2)
            ax = simulator.plot()
            y = ax.lines[-1].get_xydata()[:, 1]
            assert_array_equal(y, tests_per_person)