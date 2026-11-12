"""Learner profiles built from one course's own Moodle evidence."""

__all__ = ["main"]


def main():
    from .builder import main as builder_main

    return builder_main()
