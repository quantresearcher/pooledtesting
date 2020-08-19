"""A miscellaneous collection of functions."""
import numpy as np
from collections.abc import Iterable
from functools import lru_cache, wraps
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal, norm


def count_not_none(*args):
    """Returns the number of objects in `args` that are not None.

    Parameters
    ----------
    args : variable-length argument list
        The objects to check if None or not.

    Returns
    -------
    int

    """
    return sum(arg is not None for arg in args)


def freeze_iterable_arguments(func):
    """A decorator that converts all the iterable array-like arguments of
    `func` into tuples.

    Parameters
    ----------
    func : function
        The function must not have any string arguments.
    
    Returns
    -------
    function
        The decorated function.

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = [tuple(arg) if isinstance(arg, Iterable) else arg for arg in args]
        kwargs = {k: tuple(v) if isinstance(v, Iterable) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    
    return wrapper


@freeze_iterable_arguments
@lru_cache(maxsize = None)
def _find_correlation(x, target_probability):
    """Returns the correlation of the multivariate gaussian distribution with
    mean [0, 0] for which the CDF evaluated at `x` is `target_probability`.

    Parameters
    ----------
    x : 1D array-like
        An array of two floats.
    target_probability : float
        A number between 0 and 1.

    Returns
    -------
    float
        The correlation of a 2D multivariate gaussian distribution.

    """
    def func(correlation,
             x,
             target_probability):
        """The objective function to minimise."""
        correlation_matrix = [[1, correlation],
                              [correlation, 1]]
        probability = multivariate_normal.cdf(x,
                                              mean = [0, 0],
                                              cov = correlation_matrix)
        return abs(probability - target_probability)

    result = minimize_scalar(func,
                             args = (x, target_probability),
                             bounds = (0, 1),
                             method = 'bounded')
    return result.x


def simulate_correlated_bernoulli(p,
                                  correlation,
                                  num_samples,
                                  random_state = None):
    """Returns a `num_samples` by `len(p)` array of samples drawn from
    Bernoulli distributions that have non-negative pairwise correlations of
    `correlation`.
    
    Parameters
    ----------
    p : array of floats
        An array of at least length 2 containing the probabilities of each
        Bernoulli distribution.
    correlation : 2D array-like or float
        A positive definite matrix containing the Pearson correlations between
        each pair of variables represented in `p`. The number of rows/columns
        must be equal to the length of `p`. If the pairwise correlation is the
        same for all variables, then a float can be provided. All correlations
        must be non-negative.
    num_samples : int
        The number of samples to generate.
    random_state : int or None, optional
        If an int is given, `random_state` is the seed that will be used by
        the random number generator. If None is given, then the random number
        generator is the RandomState instance used by `numpy.random`.
        
    Returns
    -------
    2D NumPy ndarray
        A `num_samples` by `len(p)` array containing ints that are either 1s
        or 0s. The means of the columns should be approximately equal to `p`
        and the pairwise correlations of the columns should correspond to
        `correlation`.
        
    Raises
    ------
    ValueError
        * If `correlation` is a negative number.
        * If `correlation` is a matrix containing at least one negative number.
        * If `correlation` is a square matrix with the same number of
          rows/columns as the number of elements in `p`.
        * If `correlation` is a non-symmetric matrix.
        * If `correlation` is not a positive semi-definite matrix.
    ImpossibleDistribution
        If no Bernoulli samples consistent with the given `p` and
        `correlation` can be generated.
        
    Notes
    -----
    This is an implementation of the algorithm proposed by Emrich and
    Piedmonte (1991) in their paper "A method for generating high-dimensional
    multivariate binary variables".
    
    An easy-to-understand description of the algorithm can be found in the
    paper "Generating nonnegatively correlated binary random variates".
    
    """
    p = np.array(p)
    identical_probabilities_and_correlations = (p == p[0]).all()
    if isinstance(correlation, (float, int, np.float, np.integer)):
        if correlation < 0:
            raise ValueError('`correlation` must be non-negative.')
        correlation = np.full(shape = (len(p), len(p)),
                              fill_value = correlation)
        np.fill_diagonal(correlation, 1)
    else:
        correlation = np.array(correlation)
        if not len(p) == correlation.shape[0] == correlation.shape[1]:
            raise ValueError('`p` and `correlation` must have the same length.')
        elif not np.allclose(correlation, correlation.T):
            raise ValueError('`correlation` must be symmetric.')
        elif (correlation < 0).any():
            raise ValueError('`correlation` must contain only non-negative values.')
        elif not (np.linalg.eigvals(correlation) >= 0).all():
            raise ValueError('`correlation` must be positive semi-definite.')
        if identical_probabilities_and_correlations:
            scalars = correlation[np.tril_indices_from(correlation, -1)]
            identical_probabilities_and_correlations = (scalars == scalars[0]).all()

    # Step 1: determine mean vector for multivariate gaussian distribution
    multivariate_gaussian_mean = norm().ppf(p)

    # Step 2: determine correlation matrix for multivariate gaussian distribution
    multivariate_gaussian_correlation = np.eye(len(p))
    if identical_probabilities_and_correlations: # separating out this case for efficiency
        bernoulli_joint_prob = correlation[0, 1] * p[0] * (1 - p[0]) + p[0] ** 2
        multivariate_gaussian_correlation[np.tril_indices_from(multivariate_gaussian_correlation, -1)] = _find_correlation(multivariate_gaussian_mean[[0, 1]], bernoulli_joint_prob)
    else:
        outer_prod = np.outer(p, p)
        bernoulli_joint_prob = correlation * (outer_prod * np.outer(1 - p, 1 - p)) ** .5 + outer_prod
        for indices in zip(*np.tril_indices_from(multivariate_gaussian_correlation, -1)): # this loop is a major bottleneck for large matrices
            multivariate_gaussian_correlation[indices] = _find_correlation(multivariate_gaussian_mean[list(indices)], bernoulli_joint_prob[indices])
    multivariate_gaussian_correlation += np.tril(multivariate_gaussian_correlation, -1).T

    # Step 3: draw samples from multivariate gaussian distribution
    random_number_generator = np.random.RandomState(random_state)
    try:
        samples = random_number_generator.multivariate_normal(mean = multivariate_gaussian_mean,
                                                              cov = multivariate_gaussian_correlation,
                                                              size = num_samples,
                                                              check_valid = 'raise')
    except ValueError:
        raise ImpossibleDistribution('No Bernoulli samples consistent with the given `p` and `correlation` can be generated.')

    # Step 4: transform normal distribution to bernoulli distribution
    samples[samples > 0] = 1
    samples[samples <= 0] = 0
    return samples.astype(int)


class ImpossibleDistribution(ValueError):
    """Raised when no valid probability distribution with the given parameters
    exists.
    """