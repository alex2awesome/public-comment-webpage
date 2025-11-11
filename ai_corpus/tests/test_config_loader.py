from pathlib import Path

import pytest

from ai_corpus.config.loader import ConfigError, load_config


def test_load_default_config():
    cfg = load_config()
    assert "global" in cfg
    assert "regulations_gov" in cfg


def test_load_custom_path(tmp_path: Path):
    test_cfg = tmp_path / "config.yaml"
    test_cfg.write_text("global:\n  user_agent: tester\ncustom:\n  value: 1\n", encoding="utf-8")
    cfg = load_config(test_cfg)
    assert cfg["global"]["user_agent"] == "tester"
    assert cfg["custom"]["value"] == 1


def test_missing_config(tmp_path: Path):
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError):
        load_config(missing)
