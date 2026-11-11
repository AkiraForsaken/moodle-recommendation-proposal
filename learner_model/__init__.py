"""Population-separated learner modeling pipeline."""

__all__ = ["main"]


def main():
    from .builder import main as builder_main

    return builder_main()
