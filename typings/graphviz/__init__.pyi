"""
This type stub file was generated by pyright.
"""

from ._defaults import set_default_engine, set_default_format, set_jupyter_format
from .backend import DOT_BINARY, UNFLATTEN_BINARY, pipe, pipe_lines, pipe_lines_string, pipe_string, render, unflatten, version, view
from .exceptions import CalledProcessError, DotSyntaxWarning, ExecutableNotFound, FileExistsError, FormatSuffixMismatchWarning, RequiredArgumentError, UnknownSuffixWarning
from .graphs import Digraph, Graph
from .jupyter_integration import SUPPORTED_JUPYTER_FORMATS
from .parameters import ENGINES, FORMATS, FORMATTERS, RENDERERS
from .quoting import escape, nohtml
from .sources import Source

"""Assemble DOT source code and render it with Graphviz.

Example:
    >>> import graphviz  # doctest: +NO_EXE
    >>> dot = graphviz.Digraph(comment='The Round Table')

    >>> dot.node('A', 'King Arthur')
    >>> dot.node('B', 'Sir Bedevere the Wise')
    >>> dot.node('L', 'Sir Lancelot the Brave')

    >>> dot.edges(['AB', 'AL'])

    >>> dot.edge('B', 'L', constraint='false')

    >>> print(dot)  #doctest: +NORMALIZE_WHITESPACE
    // The Round Table
    digraph {
        A [label="King Arthur"]
        B [label="Sir Bedevere the Wise"]
        L [label="Sir Lancelot the Brave"]
        A -> B
        A -> L
        B -> L [constraint=false]
    }
"""
__all__ = ['ENGINES', 'FORMATS', 'RENDERERS', 'FORMATTERS', 'DOT_BINARY', 'UNFLATTEN_BINARY', 'SUPPORTED_JUPYTER_FORMATS', 'Graph', 'Digraph', 'Source', 'escape', 'nohtml', 'render', 'pipe', 'pipe_string', 'pipe_lines', 'pipe_lines_string', 'unflatten', 'version', 'view', 'ExecutableNotFound', 'CalledProcessError', 'RequiredArgumentError', 'FileExistsError', 'UnknownSuffixWarning', 'FormatSuffixMismatchWarning', 'DotSyntaxWarning', 'set_default_engine', 'set_default_format', 'set_jupyter_format']
__title__ = ...
__version__ = ...
__author__ = ...
__license__ = ...
__copyright__ = ...
ENGINES = ...
FORMATS = ...
RENDERERS = ...
FORMATTERS = ...
SUPPORTED_JUPYTER_FORMATS = ...
DOT_BINARY = ...
UNFLATTEN_BINARY = ...
ExecutableNotFound = ...
CalledProcessError = ...
RequiredArgumentError = ...
FileExistsError = ...
UnknownSuffixWarning = ...
FormatSuffixMismatchWarning = ...
DotSyntaxWarning = ...