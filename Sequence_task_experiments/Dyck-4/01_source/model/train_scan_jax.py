#!/usr/bin/env python3
"""Public CLI entry point for the serialized JAX SCAN trainer."""

if __package__:
    from .train_scan_jax_impl import build_parser, main
else:
    from train_scan_jax_impl import build_parser, main


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
