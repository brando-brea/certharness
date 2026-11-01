import pytest

from certharness.mutate import mutate_all
from certharness.synthetic import generate


@pytest.fixture(scope="session")
def originals():
    return generate(24, seed=7)


@pytest.fixture(scope="session")
def mutants(originals):
    return mutate_all(originals, seed=7)
