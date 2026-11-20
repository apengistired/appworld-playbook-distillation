"""Render the N highest-ranked entries from an ACE playbook."""
import argparse
from pathlib import Path

from ace_offline import Playbook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("n", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("n must be positive")

    playbook = Playbook.load(args.source)
    ranked = sorted(
        playbook.bullets,
        key=lambda entry: (entry["helpful"] - entry["harmful"], entry["helpful"]),
        reverse=True,
    )
    playbook.bullets = ranked[:args.n]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(playbook.render() + "\n")
    print(f"Wrote {len(playbook.bullets)} entries to {args.output}")


if __name__ == "__main__":
    main()
