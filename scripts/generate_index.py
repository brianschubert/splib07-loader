#!/usr/bin/env python3
import argparse
import lzma
import pathlib
import pickle
import sys

from splib07._index import Splib07Index


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("library_path", type=pathlib.Path)
    parser.add_argument("output_path", type=pathlib.Path)
    return parser


def _main(cli_args: list[str]) -> None:
    args = _make_parser().parse_args(cli_args)

    index = Splib07Index.generate_index(args.library_path)
    with lzma.open(args.output_path, "wb") as fd:
        pickle.dump(index, fd)


if __name__ == "__main__":
    _main(sys.argv[1:])
