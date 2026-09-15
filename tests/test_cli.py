"""CLI errors and paths must agree with the underlying scientific operations."""
from pathlib import Path

from wnt_pinn.cli import main


def root_fixture(tmp_path):
    (tmp_path / 'AGENTS.md').write_text('test fixture\n')
    (tmp_path / 'pyproject.toml').write_text('[project]\nname="test"\n')
    return tmp_path


def test_integrity_failure_has_nonzero_cli_exit(tmp_path, monkeypatch):
    from wnt_pinn import artifacts
    root = root_fixture(tmp_path)
    monkeypatch.setattr(artifacts, 'verify', lambda *args, **kwargs: {'status': 'error', 'failures': ['bad hash']})
    assert main(['--root', str(root), 'artifacts', 'verify']) == 1


def test_resume_is_relative_to_explicit_root(tmp_path, monkeypatch):
    from wnt_pinn import runs
    root = root_fixture(tmp_path)
    (root / 'configs').mkdir()
    (root / 'configs/check.json').write_text('{"pipeline": "integral"}\n')
    captured = {}
    def execute(config, **kwargs):
        captured.update(config=config, **kwargs)
        return root / 'runs/id'
    monkeypatch.setattr(runs, 'run_experiment', execute)
    assert main(['--root', str(root), 'run', 'configs/check.json', '--resume', 'runs/id']) == 0
    assert captured['config'] == root / 'configs/check.json'
    assert captured['resume'] == root / 'runs/id'
    assert isinstance(captured['root'], Path)
