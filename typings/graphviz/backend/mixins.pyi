"""
This type stub file was generated by pyright.
"""

from .. import parameters

"""Mixin classes used by Base subclasses to inherit backend functionality."""
__all__ = ['Render', 'Pipe', 'Unflatten', 'View']
class Render(parameters.Parameters):
    """Parameters for calling and calling ``graphviz.render()``."""
    ...


class Pipe(parameters.Parameters):
    """Parameters for calling and calling ``graphviz.pipe()``."""
    _get_format = ...
    _get_filepath = ...


class Unflatten:
    ...


class View:
    """Open filepath with its default viewing application
        (platform-specific)."""
    _view_darwin = ...
    _view_freebsd = ...
    _view_linux = ...
    _view_windows = ...


