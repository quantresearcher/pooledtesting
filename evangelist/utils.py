"""A collection of functions for calculating the expected number of tests
required in two-stage hierarchical testing.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from collections import defaultdict
from functools import wraps
from inspect import getfullargspec
from simulator.utils import simulate_correlated_bernoulli


def _memoise(func):
    """A decorator that stores the result of calls to functions that contain a
    `pool_sizes` argument.

    This speeds up subsequent calls to the decorated function that re-use a
    subset of `pool_sizes` while leaving other arguments unchanged.
    
    Parameters
    ----------
    func : function
        The function should contain an argument named `pool_sizes` whose type
        is either an int or an array. The function must return a dict whose
        keys are the element(s) in `pool_sizes`.
    
    Returns
    -------
    function
        The decorated function.
        
    """
    cache = defaultdict(lambda: defaultdict(dict)) # keys are decorated functions

    @wraps(func)
    def wrapper(*args, **kwargs):
        argument_names = getfullargspec(func).args
        kwargs.update(dict(zip(argument_names, args)))
        other_args = tuple(kwargs[argument_name] for argument_name in argument_names if argument_name != 'pool_sizes')
        pool_size_cache = cache[func][other_args]
        pool_sizes = kwargs['pool_sizes']
        if isinstance(pool_sizes, int):
            pool_sizes = [pool_sizes]
        new_pool_sizes = [pool_size for pool_size in pool_sizes if pool_size not in pool_size_cache]
        if len(new_pool_sizes):
            new_args = tuple(kwargs[argument_name] if argument_name != 'pool_sizes' else new_pool_sizes for argument_name in argument_names)
            pool_size_cache.update(func(*new_args))
        return {pool_size: pool_size_cache[pool_size] for pool_size in pool_sizes}
    
    return wrapper


@_memoise
def get_number_of_tests_simulated(prevalence,
                                  correlation,
                                  sensitivity,
                                  specificity,
                                  pool_sizes,
                                  num_trials,
                                  random_state = None):
    """Returns a dict of the expected number of tests per person keyed by pool
    size when two-stage hierarchical testing is carried out.

    The expected number of tests are calculated based on running `num_trials`
    simulations.
    
    Parameters
    ----------
    prevalence : float
        The proportion of the population who have the disease.
    correlation : float
        The correlation of the infection statuses between each pair of people.
    sensitivity : float
        The sensitivity of the diagnostic test (true positive rate).
    specificity : float
        The specificity of the diagnostic test (true negative rate).
    pool_sizes : int or array-like of ints
        The pool size for the first stage of testing specified as an int.
        Multiple pool sizes can be given as an array-like of ints. All pool
        sizes must be greater than 1.
    num_trials : int
        The number of simulations to run.
    random_state : int, optional
        If an int is given, `random_state` is the seed that will be used by
        the random number generator.
    
    Returns
    -------
    dict
        A dict of the average number of tests per person over the `num_trials`
        simulations keyed by the element(s) in `pool_sizes`.
        
    """
    pool_sizes = np.array(pool_sizes)
    populations = simulate_correlated_bernoulli(p = (prevalence,) * max(pool_sizes),
                                                correlation = correlation,
                                                num_samples = num_trials,
                                                random_state = random_state)
    true_positive_pools = np.sign(populations.cumsum(axis = 1)).sum(axis = 0)[pool_sizes - 1]
    true_negative_pools = num_trials - true_positive_pools
    num_positive_tests = true_positive_pools * sensitivity + true_negative_pools * (1 - specificity)
    total_tests = num_trials + num_positive_tests * pool_sizes
    expected_number_of_tests = total_tests / num_trials
    expected_number_of_tests_per_person = expected_number_of_tests / pool_sizes
    return dict(zip(pool_sizes, expected_number_of_tests_per_person))


def get_number_of_tests_exact(prevalence,
                              sensitivity,
                              specificity,
                              pool_sizes):
    """Returns a dict of the expected number of tests per person needed for
    two-stage hierarchical testing keyed by pool size.

    The expected number of tests are calculated exactly using a formula.
    However, this function is limited to the case where there is no correlaton
    in the infection statuses among people.
    
    Parameters
    ----------
    prevalence : float
        The proportion of the population who have the disease.
    sensitivity : float
        The sensitivity of the diagnostic test (true positive rate).
    specificity : float
        The specificity of the diagnostic test (true negative rate).
    pool_sizes : int or array-like
        The pool size for the first stage of testing specified as an int.
        Multiple pool sizes can be given as an array of ints. All pool
        sizes must be greater than 1.
    
    Returns
    -------
    dict
        A dict of the expected number of tests per person keyed by the
        pool size(s) in `pool_sizes`.
        
    """
    if isinstance(pool_sizes, int):
        pool_sizes = [pool_sizes]
    pool_sizes = np.array(pool_sizes)
    no_disease_probability = (1 - prevalence) ** pool_sizes
    positive_test_probability = sensitivity * (1 - no_disease_probability) + (1 - specificity) * no_disease_probability
    expected_number_of_tests = 1 + pool_sizes * positive_test_probability
    expected_number_of_tests_per_person = expected_number_of_tests / pool_sizes
    return dict(zip(pool_sizes, expected_number_of_tests_per_person))