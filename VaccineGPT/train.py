"""Compatibility launcher for both repository-root and VaccineGPT/ invocations."""

try:
    from .vaccinegpt.cli import main
except ImportError:  # Executed as ``python train.py`` rather than imported.
    from vaccinegpt.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
