"""Command-line interface: ``aqyn <command> ...``."""

from __future__ import annotations

import argparse
import importlib

COMMANDS = {
    "prepare": ("prepare", "download LJSpeech and encode it with Mimi"),
    "train": ("train", "train a model from a YAML config"),
    "synth": ("synthesize", "synthesize speech from text"),
    "eval": ("evaluate", "evaluate a checkpoint or the Mimi ceiling"),
    "bench": ("bench", "measure training speed and memory on this GPU"),
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="aqyn", description="Aqyn: streaming TTS with CfC over Mimi")
    sub = ap.add_subparsers(dest="command", required=True, metavar="command")
    for name, (module, help_text) in COMMANDS.items():
        mod = importlib.import_module(f"aqyn.{module}")
        p = sub.add_parser(
            name,
            help=help_text,
            description=mod.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        mod.add_args(p)
        p.set_defaults(_run=mod.run)
    args = ap.parse_args(argv)
    args._run(args)


if __name__ == "__main__":
    main()
