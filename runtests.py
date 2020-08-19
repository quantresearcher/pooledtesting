"""Runs all the unit tests in this package."""
import os
import unittest


def run_all_tests():
    """Runs all test cases in the package."""
    loader = unittest.TestLoader()
    directory = os.path.dirname(__file__)
    suite = loader.discover(start_dir = directory)
    runner = unittest.TextTestRunner()
    return runner.run(suite)


if __name__ == '__main__':
    run_all_tests()