"""Run an experiment from a checkout, after installing the project."""
from pathlib import Path
import argparse

from wnt_pinn.runs import RunFailed, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    try:
        directory = run_experiment(args.config, root=args.root, resume=args.resume)
    except (RunFailed, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(directory)


if __name__ == "__main__":
    main()
