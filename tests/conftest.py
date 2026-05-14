import os

import pytest

os.environ["OFFLINE_DEMO_MODE"] = "1"
os.environ["AUTO_REVIEW_DECISION"] = "skip"


@pytest.fixture(autouse=True)
def reset_singletons():
    import audit.config as _cfg
    import audit.judge as _judge
    _cfg.reset_config()
    _judge.reset_judge()
    yield
    _cfg.reset_config()
    _judge.reset_judge()
