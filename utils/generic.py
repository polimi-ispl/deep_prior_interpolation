from math import floor, log10
import random
import string


def random_code(n=6):
    return ''.join([random.choice(string.ascii_letters + string.digits)
                    for _ in range(int(n))])


def ten_digit(number):
    return int(floor(log10(number)) + 1)


def sec2time(seconds):
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // 3600
    timestamp = '%dh:%dm:%ds' % (h, m, s)
    return timestamp


__all__ = [
    "random_code",
    "ten_digit",
    "sec2time"
]
