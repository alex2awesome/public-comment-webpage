# Autometrics Test Guide

This folder contains pytest-based tests for the Autometrics package. The tests validate functionality for datasets, metrics, and caching mechanisms.

## Running Tests

### Prerequisites

Make sure you have pytest installed:

```bash
pip install pytest
```

### Running All Tests

To run all tests, navigate to the autometrics root directory and run:

```bash
pytest
```

Or alternatively:

```bash
python -m pytest
```

### Running Specific Test Files

To run tests from a specific file:

```bash
pytest test/test_dataset.py
```

### Running Specific Test Functions

To run a specific test function:

```bash
pytest test/test_dataset.py::test_get_subset_normal
```

### Running Tests with Markers

Some tests are marked with specific categories:

- `slow`: Tests that take more time to run
- `integration`: Integration tests that test multiple components together

To run only tests with a specific marker:

```bash
pytest -m slow
```

To skip tests with a specific marker:

```bash
pytest -m "not slow"
```

### Verbose Output

To see more details about the tests being run:

```bash
pytest -v
```

### Test Coverage

To see test coverage (requires pytest-cov plugin):

```bash
pip install pytest-cov
pytest --cov=autometrics
```

For a detailed HTML coverage report:

```bash
pytest --cov=autometrics --cov-report=html
```

## Test Structure

- `test_dataset.py`: Tests for Dataset and PairwiseDataset functionality
- `test_pairwise_metric.py`: Tests for PairwiseMetric
- `test_pairwise_multi_metric.py`: Tests for PairwiseMultiMetric
- `test_cache.py`: Tests for metric caching
- `test_init_param_caching.py`: Tests for caching with different initialization parameters

## Fixtures

Common test fixtures are defined in `conftest.py`, which provides:

- `cleanup_cache`: Handles setup and teardown of cache directories
- Various data fixtures used across multiple test files

## Adding New Tests

When adding new tests:

1. Follow the pytest conventions
2. Use appropriate fixtures for setup and teardown
3. Add meaningful assertions to validate functionality
4. Group related tests in the same file
5. Add appropriate markers for slow or integration tests 