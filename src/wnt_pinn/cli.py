"""Command-line entry points for experiments, results and publications."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import uuid
import subprocess
import json
from pathlib import Path
import sys


def repository_root(value=None):
    if value:
        root = Path(value).expanduser().resolve()
        if not (root / 'AGENTS.md').is_file():
            raise ValueError(f'Not a research checkout: {root}')
        return root
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        for path in (base, *base.parents):
            if (path / 'AGENTS.md').is_file() and (path / 'pyproject.toml').is_file():
                return path
    raise ValueError('Run from a research checkout or supply --root PATH.')


def parser():
    result = argparse.ArgumentParser(prog='wnt-pinn', description=__doc__)
    result.add_argument('--root', help='Path to the research checkout')
    commands = result.add_subparsers(dest='command', required=True)
    reproduce = commands.add_parser('reproduce', help='Recompute registered saved results')
    selection = reproduce.add_mutually_exclusive_group()
    selection.add_argument('--all', action='store_true')
    selection.add_argument('--result', action='append', dest='results')
    reproduce.add_argument('--output', type=Path)
    commands.add_parser('verify', help='Verify result registry and source hashes')
    run = commands.add_parser('run', help='Run a validated experiment configuration')
    run.add_argument('config', nargs='?', type=Path)
    run.add_argument('--config', dest='config_option', type=Path)
    run.add_argument('--resume', type=Path)
    artifacts = commands.add_parser('artifacts', help='Inventory, archive and restore run artifacts')
    actions = artifacts.add_subparsers(dest='action', required=True)
    inv = actions.add_parser('inventory')
    inv.add_argument('--output', type=Path)
    inv.add_argument('--run', action='append', dest='include_runs')
    fetch = actions.add_parser('fetch')
    fetch.add_argument('destination')
    fetch.add_argument('--result', action='append', dest='results')
    fetch.add_argument('--include-upstream', action='store_true')
    for name in ('archive', 'restore', 'verify'):
        action = actions.add_parser(name)
        action.add_argument('--select', action='append')
        if name != 'verify':
            action.add_argument('destination', help='External store path or HTTPS URL for restore')
    publications = commands.add_parser('publications', help='Build publications from cached inputs')
    pubs = publications.add_subparsers(dest='action', required=True)
    build = pubs.add_parser('build')
    selection = build.add_mutually_exclusive_group(required=True)
    selection.add_argument('--all', action='store_true')
    selection.add_argument('--id', action='append', dest='ids')
    build.add_argument('--output', type=Path)
    build.add_argument('--figures-only', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        root = repository_root(args.root)
        if args.command == 'reproduce':
            from .reporting import reproduce
            receipt = reproduce(root, result_ids=args.results, output_dir=args.output)
        elif args.command == 'verify':
            from .reporting import verify_registry
            receipt = verify_registry(root)
        elif args.command == 'run':
            from .runs import run_experiment
            config = args.config_option or args.config
            if config is None:
                raise ValueError('Provide an experiment configuration path.')
            config = config if config.is_absolute() else root / config
            resume = args.resume
            if resume is not None and not resume.is_absolute():
                resume = root / resume
            output = run_experiment(config, root=root, resume=resume)
            receipt = {'ok': True, 'run_directory': str(output)}
        elif args.command == 'artifacts':
            from . import artifacts
            if args.action == 'inventory':
                receipt = artifacts.inventory(root, output_path=args.output, include_runs=args.include_runs)
            elif args.action == 'fetch':
                receipt = artifacts.restore_required_inputs(root, args.destination, result_ids=args.results, include_upstream=args.include_upstream)
            elif args.action == 'verify':
                receipt = artifacts.verify(root, selected=args.select)
            else:
                receipt = getattr(artifacts, args.action)(root, args.destination, selected=args.select)
        elif args.command == 'publications':
            from .reporting.publications import build_publication
            ids = args.ids
            if args.all:
                catalog = json.loads((root / 'publications/catalog.json').read_text())
                rows = catalog.get('publications', catalog)
                ids = list(rows) if isinstance(rows, dict) else [row['id'] for row in rows]
            receipt = {'ok': True, 'publications': []}
            base_output = args.output or root / 'build/publications' / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
            for identifier in ids:
                output = base_output / identifier
                item = build_publication(root, identifier, output, figures_only=args.figures_only)
                receipt['publications'].append(item)
                if item.get('ok') is False:
                    receipt['ok'] = False
        print(json.dumps(receipt, indent=2, default=str))
        return 1 if receipt.get('ok') is False or receipt.get('status') in {'failed', 'error'} else 0
    except (ValueError, OSError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        print(f'wnt-pinn: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
