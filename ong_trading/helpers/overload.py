"""
Functions to overload a class method so it calls one or other function according to method arguments
It works exactly like functools.singledispatch or functools.singledispatchmethod but 40% faster
It is based on https://stackoverflow.com/a/25344433
"""
import functools


# Original function from stackoverflow
def multidispatch(*types):
    def register(function):
        name = function.__name__
        mm = multidispatch.registry.get(name)
        if mm is None:
            @functools.wraps(function)
            def wrapper(self, *args):
                types = tuple(arg.__class__ for arg in args)
                function = wrapper.typemap.get(types)
                if function is None:
                    raise TypeError("no match")
                return function(self, *args)
            wrapper.typemap = {}
            mm = multidispatch.registry[name] = wrapper
        if types in mm.typemap:
            raise TypeError("duplicate registration")
        mm.typemap[types] = function
        return mm
    return register


multidispatch.registry = {}


# Modded function to work with just one argument
def singledispatch(typ):
    def register(function):
        name = function.__name__
        mm = singledispatch.registry.get(name)
        if mm is None:
            @functools.wraps(function)
            def wrapper(self, arg):
                # types = tuple(arg.__class__ for arg in args)
                types = type(arg)
                function = wrapper.typemap.get(types)
                if function is None:
                    raise TypeError("no match")
                return function(self, arg)
            wrapper.typemap = {}
            mm = singledispatch.registry[name] = wrapper
        if typ in mm.typemap:
            raise TypeError("duplicate registration")
        mm.typemap[typ] = function
        return mm
    return register


singledispatch.registry = {}
