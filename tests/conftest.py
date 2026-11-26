from __future__ import annotations

from pathlib import Path

import pytest

from funnelshift import synth
from funnelshift.definitions import Funnel, load_funnel
from funnelshift.run import FunnelRun

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def trial() -> Funnel:
    return load_funnel(ROOT / "funnels" / "trial.yaml")


@pytest.fixture(scope="session")
def trial_21(trial: Funnel) -> Funnel:
    return trial.model_copy(update={"window_days": 21})


@pytest.fixture(scope="session")
def data() -> synth.SynthData:
    return synth.generate(seed=7)


@pytest.fixture(scope="session")
def events_path(data: synth.SynthData, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synth.write_dataset(data, tmp_path_factory.mktemp("synth"))


@pytest.fixture(scope="session")
def run14(events_path: Path, trial: Funnel) -> FunnelRun:
    return FunnelRun.from_parquet(events_path, trial)


@pytest.fixture(scope="session")
def run21(events_path: Path, trial_21: Funnel) -> FunnelRun:
    return FunnelRun.from_parquet(events_path, trial_21)
