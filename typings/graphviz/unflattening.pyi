"""
This type stub file was generated by pyright.
"""

import typing
import graphviz
from . import _tools, backend, base, encoding

"""Pipe source through the Graphviz *unflatten* preprocessor."""
__all__ = ['Unflatten']
class Unflatten(encoding.Encoding, base.Base, backend.Unflatten):
    """Pipe source through the Graphviz *unflatten* preprocessor."""
    @_tools.deprecate_positional_args(supported_number=1)
    def unflatten(self, stagger: typing.Optional[int] = ..., fanout: bool = ..., chain: typing.Optional[int] = ...) -> graphviz.Source:
        """Return a new :class:`.Source` instance with the source
            piped through the Graphviz *unflatten* preprocessor.

        Args:
            stagger: Stagger the minimum length
                of leaf edges between 1 and this small integer.
            fanout: Fanout nodes with indegree = outdegree = 1
                when staggering (requires ``stagger``).
            chain: Form disconnected nodes into chains
                of up to this many nodes.

        Returns:
            Prepocessed DOT source code (improved layout aspect ratio).

        Raises:
            graphviz.RequiredArgumentError: If ``fanout`` is given
                but ``stagger`` is None.
            graphviz.ExecutableNotFound: If the Graphviz ``unflatten`` executable
                is not found.
            graphviz.CalledProcessError: If the returncode (exit status)
                of the unflattening 'unflatten' subprocess is non-zero.

        See also:
            Upstream documentation:
            https://www.graphviz.org/pdf/unflatten.1.pdf
        """
        ...
    


