from typing import Dict, List, Optional, Union
from pathlib import Path
from math import floor, log10
import random
import string
import json
from argparse import Namespace


def random_code(n: int = 6) -> str:
    return ''.join([random.choice(string.ascii_letters + string.digits)
                    for _ in range(int(n))])


def ten_digit(number: float) -> int:
    return int(floor(log10(number)) + 1)


def sec2time(seconds: float) -> str:
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // 3600
    timestamp = '%dh:%dm:%ds' % (h, m, s)
    return timestamp


def read_args(filename: Union[str, Path]) -> Namespace:
    args = Namespace()
    with open(filename, 'r') as fp:
        args.__dict__.update(json.load(fp))
    return args


def write_args(filename: Union[str, Path], args: Namespace, indent: int = 2) -> None:
    with open(filename, 'w') as fp:
        json.dump(args.__dict__, fp, indent=indent)


__all__ = [
    "random_code",
    "ten_digit",
    "sec2time",
    "read_args",
    "write_args",
]
