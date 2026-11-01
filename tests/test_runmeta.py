import subprocess

import pytest

from certharness.runmeta import FreezeError, freeze_status, require_frozen


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "src/certharness/graders").mkdir(parents=True)
    (tmp_path / "src/certharness/graders/code.py").write_text("x = 1\n")
    (tmp_path / "other.py").write_text("y = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    _git(tmp_path, "tag", "frozen")
    return tmp_path


def test_missing_tag(repo):
    s = freeze_status("nope", root=repo)
    assert not s["ok"] and "does not exist" in s["reason"]


def test_intact_and_unrelated_changes_ok(repo):
    (repo / "other.py").write_text("y = 2\n")
    _git(repo, "commit", "-qam", "unrelated")
    assert require_frozen("frozen", root=repo)["ok"]


def test_uncommitted_grader_edit_refused(repo):
    (repo / "src/certharness/graders/code.py").write_text("x = 2\n")
    with pytest.raises(FreezeError, match="code.py"):
        require_frozen("frozen", root=repo)


def test_committed_grader_edit_and_new_file_refused(repo):
    (repo / "src/certharness/graders/code.py").write_text("x = 3\n")
    _git(repo, "commit", "-qam", "tweak")
    assert not freeze_status("frozen", root=repo)["ok"]
    _git(repo, "tag", "frozen2")
    (repo / "src/certharness/graders/new.py").write_text("")
    s = freeze_status("frozen2", root=repo)
    assert not s["ok"] and "new.py" in s["reason"]
