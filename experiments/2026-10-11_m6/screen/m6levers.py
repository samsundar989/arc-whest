"""M6.4 lever patches (exact text replacements, in the style of price.py).

lateage(L0): the old-tier join bar becomes _SHARED_AGE + 1 from layer L0 on, so lineages born at
layer >= L0 - 5 stay exact for 6 layers instead of 5. L0 = 13 is "age 6 only for lineages born
at layer >= 8". Every use of the bar (join test and append test) reads the same per-layer value,
so join and birth stay consistent within a layer.
"""


def _rep(s, a, b, count=1):
    n = s.count(a)
    if n != count:
        raise SystemExit(f"m6levers: anchor occurs {n} times, expected {count}:\n{a}")
    return s.replace(a, b)


def lateage(L0, base=5):
    """`base` = the file's _SHARED_AGE (6 after price.age(6), giving age 7 late)."""
    def p(s):
        s = _rep(s, f"_SHARED_AGE = {base}\n",
                 f"_SHARED_AGE = {base}\n"
                 "#: M6.4: late lineages stay exact one generation longer (bar +1 from this layer on).\n"
                 f"_AGE6_FROM = {int(L0)}\n\n\n"
                 "def _age_bar():\n"
                 "    return _SHARED_AGE + 1 if _CUR_LAYER >= _AGE6_FROM else _SHARED_AGE\n")
        s = _rep(s, "(tw.k - tw.ko) >= _SHARED_AGE:", "(tw.k - tw.ko) >= _age_bar():", 2)
        s = _rep(s, "(tw.k - tw.ko) < _SHARED_AGE or", "(tw.k - tw.ko) < _age_bar() or", 2)
        return s
    return p
