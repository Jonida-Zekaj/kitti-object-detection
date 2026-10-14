from utils.config import load_config

def test_load_default_config():
    cfg = load_config("configs/default.yaml")
    assert "paths" in cfg
    assert cfg["dataset"]["target_size"] == [416, 416]
    assert "Car" in cfg["dataset"]["classes"]

    # goal: verify that the default configuration loads correctly