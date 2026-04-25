"""Tests for ~/.arrayview/config.toml read/write and environment detection."""

import os
import textwrap

import pytest

from arrayview._config import (
    get_viewer_colormaps,
    get_viewer_rounded_panes,
    get_window_default,
    load_config,
    save_config,
)


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Redirect config to a temp directory."""
    config_dir = tmp_path / ".arrayview"
    config_dir.mkdir()
    config_file = config_dir / "config.toml"
    monkeypatch.setattr("arrayview._config.CONFIG_PATH", str(config_file))
    return config_file


class TestLoadConfig:
    def test_missing_file_returns_empty(self, tmp_config):
        assert load_config() == {}

    def test_reads_toml(self, tmp_config):
        tmp_config.write_text(textwrap.dedent("""\
            [window]
            default = "browser"
            terminal = "native"
        """))
        cfg = load_config()
        assert cfg["window"]["default"] == "browser"
        assert cfg["window"]["terminal"] == "native"

    def test_malformed_file_returns_empty(self, tmp_config):
        tmp_config.write_text("not valid [[[ toml")
        assert load_config() == {}


class TestSaveConfig:
    def test_creates_file(self, tmp_config):
        save_config({"window": {"default": "browser"}})
        assert tmp_config.exists()
        cfg = load_config()
        assert cfg["window"]["default"] == "browser"

    def test_roundtrip(self, tmp_config):
        original = {"window": {"default": "native", "vscode": "vscode", "jupyter": "inline"}}
        save_config(original)
        assert load_config() == original


class TestGetWindowDefault:
    def test_no_config_returns_none(self, tmp_config, monkeypatch):
        monkeypatch.delenv("ARRAYVIEW_WINDOW", raising=False)
        assert get_window_default("terminal") is None

    def test_env_specific(self, tmp_config, monkeypatch):
        monkeypatch.delenv("ARRAYVIEW_WINDOW", raising=False)
        tmp_config.write_text(textwrap.dedent("""\
            [window]
            terminal = "browser"
        """))
        assert get_window_default("terminal") == "browser"

    def test_falls_back_to_default(self, tmp_config, monkeypatch):
        monkeypatch.delenv("ARRAYVIEW_WINDOW", raising=False)
        tmp_config.write_text(textwrap.dedent("""\
            [window]
            default = "browser"
        """))
        assert get_window_default("terminal") == "browser"

    def test_env_specific_overrides_default(self, tmp_config, monkeypatch):
        monkeypatch.delenv("ARRAYVIEW_WINDOW", raising=False)
        tmp_config.write_text(textwrap.dedent("""\
            [window]
            default = "browser"
            terminal = "native"
        """))
        assert get_window_default("terminal") == "native"

    def test_env_var_overrides_config(self, tmp_config, monkeypatch):
        tmp_config.write_text(textwrap.dedent("""\
            [window]
            terminal = "native"
        """))
        monkeypatch.setenv("ARRAYVIEW_WINDOW", "browser")
        assert get_window_default("terminal") == "browser"

    def test_invalid_value_returns_none(self, tmp_config, monkeypatch):
        monkeypatch.delenv("ARRAYVIEW_WINDOW", raising=False)
        tmp_config.write_text(textwrap.dedent("""\
            [window]
            terminal = "invalid_mode"
        """))
        assert get_window_default("terminal") is None


class TestGetViewerColormaps:
    def test_no_config_returns_none(self, tmp_config):
        assert get_viewer_colormaps() is None

    def test_returns_list_from_config(self, tmp_config):
        tmp_config.write_text(textwrap.dedent("""\
            [viewer]
            colormaps = ["gray", "viridis", "plasma"]
        """))
        assert get_viewer_colormaps() == ["gray", "viridis", "plasma"]

    def test_empty_list_returns_none(self, tmp_config):
        tmp_config.write_text(textwrap.dedent("""\
            [viewer]
            colormaps = []
        """))
        assert get_viewer_colormaps() is None

    def test_missing_viewer_section_returns_none(self, tmp_config):
        tmp_config.write_text(textwrap.dedent("""\
            [window]
            default = "browser"
        """))
        assert get_viewer_colormaps() is None


class TestGetViewerRoundedPanes:
    def test_no_config_returns_none(self, tmp_config):
        assert get_viewer_rounded_panes() is None

    def test_returns_bool_from_config(self, tmp_config):
        tmp_config.write_text(textwrap.dedent("""\
            [viewer]
            rounded_panes = false
        """))
        assert get_viewer_rounded_panes() is False

    def test_accepts_bool_like_string(self, tmp_config):
        tmp_config.write_text(textwrap.dedent("""\
            [viewer]
            rounded_panes = "on"
        """))
        assert get_viewer_rounded_panes() is True

    def test_invalid_value_returns_none(self, tmp_config):
        tmp_config.write_text(textwrap.dedent("""\
            [viewer]
            rounded_panes = "sometimes"
        """))
        assert get_viewer_rounded_panes() is None


class TestDetectEnvironment:
    def test_jupyter(self, monkeypatch):
        monkeypatch.setattr("arrayview._platform._in_jupyter", lambda: True)
        monkeypatch.setattr("arrayview._platform._in_vscode_terminal", lambda: False)
        monkeypatch.setattr("arrayview._platform._is_julia_env", lambda: False)
        from arrayview._platform import detect_environment

        assert detect_environment() == "jupyter"

    def test_vscode(self, monkeypatch):
        monkeypatch.setattr("arrayview._platform._in_jupyter", lambda: False)
        monkeypatch.setattr("arrayview._platform._in_vscode_terminal", lambda: True)
        monkeypatch.setattr("arrayview._platform._is_julia_env", lambda: False)
        from arrayview._platform import detect_environment

        assert detect_environment() == "vscode"

    def test_julia(self, monkeypatch):
        monkeypatch.setattr("arrayview._platform._in_jupyter", lambda: False)
        monkeypatch.setattr("arrayview._platform._in_vscode_terminal", lambda: False)
        monkeypatch.setattr("arrayview._platform._is_julia_env", lambda: True)
        from arrayview._platform import detect_environment

        assert detect_environment() == "julia"

    def test_ssh(self, monkeypatch):
        monkeypatch.setattr("arrayview._platform._in_jupyter", lambda: False)
        monkeypatch.setattr("arrayview._platform._in_vscode_terminal", lambda: False)
        monkeypatch.setattr("arrayview._platform._is_julia_env", lambda: False)
        monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 5678 5.6.7.8 22")
        from arrayview._platform import detect_environment

        assert detect_environment() == "ssh"

    def test_terminal_default(self, monkeypatch):
        monkeypatch.setattr("arrayview._platform._in_jupyter", lambda: False)
        monkeypatch.setattr("arrayview._platform._in_vscode_terminal", lambda: False)
        monkeypatch.setattr("arrayview._platform._is_julia_env", lambda: False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        from arrayview._platform import detect_environment

        assert detect_environment() == "terminal"
