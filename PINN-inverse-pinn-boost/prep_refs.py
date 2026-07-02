"""Build the (regime, condition) reference trajectories once and pickle them,
so the 4 parallel regime processes each just load their slice."""
import pickle
import sys

from reference import generate_references

if __name__ == "__main__":
    out = sys.argv[1]
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
    refs = generate_references(T=T)
    with open(out, "wb") as fh:
        pickle.dump(refs, fh)
    print(f"wrote refs cache -> {out}")
