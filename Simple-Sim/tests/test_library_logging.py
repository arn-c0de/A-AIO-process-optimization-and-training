"""Library reports via logging and neutral exceptions; the CLI layer formats and adds flag hints."""

import logging

import pytest

from simple_sim.cli_utils import PROFILE_MISMATCH_HINT, UsageError, configure_cli_logging, with_hint
from simple_sim.manifest import ProfileMismatchError, RendererMismatchError, check_checkpoint_profile, check_renderer_version

CKPT = {"component_profile": {"profile_id": "a@1", "profile_hash": "sha256:aaa"}}


def test_profile_mismatch_raises_neutral_error():
    with pytest.raises(ProfileMismatchError, match="Cannot evaluate model on different component type") as ei:
        check_checkpoint_profile(CKPT, "b@1", "sha256:bbb", action="evaluate model on")
    assert "--allow" not in str(ei.value)
    assert (ei.value.checkpoint_profile, ei.value.dataset_profile) == ("a@1", "b@1")


def test_profile_mismatch_allowed_and_hash_mismatch_log_warnings(caplog):
    with caplog.at_level(logging.WARNING, logger="simple_sim.manifest"):
        assert check_checkpoint_profile(CKPT, "b@1", "sha256:bbb", action="x", allow_mismatch=True) == "a@1"
    msgs = [r.getMessage() for r in caplog.records]
    assert any("component profile mismatch" in m for m in msgs) and any("Profile hash mismatch" in m for m in msgs)


def test_legacy_checkpoint_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="simple_sim.manifest"):
        assert check_checkpoint_profile({}, "b@1", "h", action="x") is None
    assert "legacy checkpoint" in caplog.text


def test_renderer_mismatch_error_carries_versions():
    with pytest.raises(RendererMismatchError) as ei:
        check_renderer_version({"generator": {"renderer_version": "1"}}, "2")
    assert (ei.value.dataset_version, ei.value.current_version) == ("1", "2")
    assert "--allow" not in str(ei.value)


def test_with_hint_appends_cli_advice():
    def fail():
        check_checkpoint_profile(CKPT, "b@1", "h", action="x")

    with pytest.raises(UsageError) as ei:
        with_hint(fail, ProfileMismatchError, PROFILE_MISMATCH_HINT)()
    assert str(ei.value).endswith("\n\n" + PROFILE_MISMATCH_HINT)
    assert with_hint(lambda: 5, ProfileMismatchError, "x")() == 5


def test_cli_logging_prints_plain_lines_and_prefixed_warnings(capsys):
    configure_cli_logging()
    configure_cli_logging()  # idempotent: one handler
    log = logging.getLogger("simple_sim.test_probe")
    log.info("hello")
    log.warning("\nwatch out")
    out = capsys.readouterr().out
    assert out == "hello\n\nWARNING: watch out\n"


def test_library_is_silent_without_configuration():
    root = logging.getLogger("simple_sim")
    assert any(isinstance(h, logging.NullHandler) for h in root.handlers)
