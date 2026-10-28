#!/usr/bin/env python3
"""Merged Dyck task using the original single-block A1F1E3 trainer."""
import argparse
from pathlib import Path

if __package__:
    from .train_scan_jax_impl import build_parser as _base_parser, _validate_args
    from .data_jax import _dyck_root
else:
    from train_scan_jax_impl import build_parser as _base_parser, _validate_args
    from data_jax import _dyck_root


def build_parser():
    parser = _base_parser()
    parser.description = __doc__
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for command, child in action.choices.items():
            if command == "pretrain-nonlinear":
                continue
            child.set_defaults(
                data_root=Path(__file__).resolve().parent.parent / "length5" / "length5" / "DyckLanguage",
                attention_layers=1, ffn_layers=1, entangling_layers=3,
                learning_rate=0.003, max_decode_steps=6,
            )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        if hasattr(args, "data_root") and _dyck_root(args.data_root) is None:
            raise ValueError("--data-root must point directly to the merged DyckLanguage directory.")
        return int(args.handler(args))
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError, KeyError, IndexError, FloatingPointError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
