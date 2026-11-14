import pytest

from src.features import build_features
from src.ingest import clean, load_raw


@pytest.fixture(scope="session")
def raw():
    return load_raw()


@pytest.fixture(scope="session")
def flex(raw):
    return clean(raw)


@pytest.fixture(scope="session")
def feats(flex):
    return build_features(flex)
