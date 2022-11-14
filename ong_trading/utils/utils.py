"""
Some utilities
"""
import pandas as pd


def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)


def fmt_dt(dt: pd.Timestamp) -> str:
    """Formats a date for printing"""
    # return dt.strftime("%x")
    return dt.strftime("%d%b%y")


if __name__ == '__main__':
    for t in .5, 5, 50, 500, 5000, 50000, 500000:
        print(t, format_time(t))


