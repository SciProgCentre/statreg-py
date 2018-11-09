#!/usr/bin/python
"""
Utils for memoization of class methods
"""

import functools
import inspect

def memoize_null(f):
    """
    Decorator for memoization of function which doesn't take any
    arguments
    """
    table_field = '_memo_' + f.__name__
    #
    @functools.wraps(f)
    def fun(self) :
        try:
            memo = getattr(self, table_field)
        except AttributeError:
            memo = f(self)
            setattr(self, table_field, memo)
        return memo
    return fun


def memoize(f):
    """
    Decorator for memoization of function which doesn't take only
    keyword arguments
    """
    table_field = '_memo_' + f.__name__
    argnames    = list(inspect.signature(f).parameters)
    #
    @functools.wraps(f)
    def fun(self, *arg, **kwd) :
        try:
            memo = getattr(self, table_field)
        except AttributeError:
            memo = {}
            setattr(self, table_field, memo)
        # Normalize positional arguments as keyword arguments. It's quite costly
        dct = kwd.copy()
        for i,v in enumerate(arg, 1):
            # If argument is invalid call function and die
            if i >= len(argnames) :
                return f(self, *arg, **kwd)
            k = argnames[i]
            if k in dct :
                return f(self, *arg, **kwd)
            dct[k] = v
        key = frozenset(dct.items())
        if key not in memo:
            memo[key] = f(self, *arg, **kwd)
        return memo[key]
    return fun
