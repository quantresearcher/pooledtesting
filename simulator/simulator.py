"""Simulator for running multiple testing strategies."""
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from time import time
from .testingstrategy import TestingStrategy


class Simulator:
    """Runs testing strategies over simulated populations to calculate the
    expected number of tests per person for each strategy.

    Parameters
    ----------
    population_distribution : PopulationDistribution
        Contians the population samples to run the testing strategies on.
    strategies : TestingStrategy or list of TestingStrategy
        The testing strategies to run.

    Attributes
    ----------
    _population_distribution : see Parameters
    _strategies : list of TestingStrategy
        A list of TestingStrategy instances to run.
    _tests_conducted : dict
        A dict of the total number of tests conducted for each strategy keyed
        by the string representations of the TestingStrategy instance.
    _runtime : float
        The total number of seconds taken to run all the strategies.

    """
    def __init__(self,
                 population_distribution,
                 strategies):
        self._population_distribution = population_distribution
        self._strategies = [strategies] if isinstance(strategies, TestingStrategy) else strategies
        self._tests_conducted = dict()
        self._run_time = 0
        

    @property
    def tests_per_person(self):
        """Returns a Pandas Series of the average number of tests per person
        indexed by testing strategy.
        """
        return pd.Series(self._tests_conducted) / self._population_distribution.samples.size


    @property
    def run_time(self):
        """Returns the total number of seconds taken to run all the strategies."""
        return self._run_time
    

    def run(self, max_workers = None):
        """Runs the simulation.

        Parameters
        ----------
        max_workers : int, optional
            The maximum number of threads to run in parallel. If None is
            given, it will default to the number of cores on the machine.

        Notes
        -----
        Since the TestingStrategy instances make heavy use of NumPy
        computations and since the Global Interpreter Lock is released during
        NumPy computations, using multiple threads will achieve a speedup. On
        the other hand, using multiple processes will result in a significant
        slowdown compared to using just a single process.

        """
        self._tests_conducted.clear()
        start_time = time()
        with ThreadPoolExecutor(max_workers = max_workers or cpu_count()) as executor:
            for strategy, total_tests_conducted in executor.map(self._run_single_strategy, self._strategies):
                self._tests_conducted[str(strategy)] = total_tests_conducted
        self._run_time = time() - start_time


    def _run_single_strategy(self, strategy):
        """Returns the total tests conducted when `strategy` is run.

        Parameters
        ----------
        strategy : TestingStrategy
            The strategy to run.

        Returns
        -------
        2-tuple
            The first element is `strategy` and the second element is the
            number of tests conducted.

        """
        strategy.run(population_distribution = self._population_distribution)
        return strategy, strategy.total_tests_conducted
        

    def plot(self, figsize = (14, 7)):
        """Plots the average number of tests per person for each strategy.

        Parameters
        ----------
        figsize : 2-tuple of ints, optional
            The dimensions of the figure in inches, where the first element is
            the width and the second element is the height.
        
        Returns
        -------
        Matplotlib Axes
            The axes containing the plot.

        """
        x_ticks = np.arange(1, len(self._tests_conducted) + 1)
        tests_per_person = pd.Series(data = self.tests_per_person.values,
                                     index = x_ticks)
        ax = tests_per_person.plot(figsize = figsize)
        ax.set_xticks(x_ticks)
        ax.set_xlabel('Strategy')
        ax.set_ylabel(f'Average number of tests per person')
        optimal_strategy = tests_per_person.idxmin()
        ax.set_title(f'Optimal strategy: {optimal_strategy}         ' \
                     f'Reduction in tests: {round(100 * (1 - tests_per_person[optimal_strategy]), 1)}%')
        return ax