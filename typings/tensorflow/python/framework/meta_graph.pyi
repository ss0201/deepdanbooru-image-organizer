"""
This type stub file was generated by pyright.
"""

"""MetaGraph and related functions."""
_UNBOUND_INPUT_PREFIX = ...
_COMPAT_COLLECTION_LIST = ...
def ops_used_by_graph_def(graph_def): # -> list[Unknown]:
  """Collect the list of ops used by a graph.

  Does not validate that the ops are all registered.

  Args:
    graph_def: A `GraphDef` proto, as from `graph.as_graph_def()`.

  Returns:
    A list of strings, each naming an op used by the graph.
  """
  ...

def stripped_op_list_for_graph(graph_def): # -> OpList:
  """Collect the stripped OpDefs for ops used by a graph.

  This function computes the `stripped_op_list` field of `MetaGraphDef` and
  similar protos.  The result can be communicated from the producer to the
  consumer, which can then use the C++ function
  `RemoveNewDefaultAttrsFromGraphDef` to improve forwards compatibility.

  Args:
    graph_def: A `GraphDef` proto, as from `graph.as_graph_def()`.

  Returns:
    An `OpList` of ops used by the graph.
  """
  ...

SAVE_AND_RESTORE_OPS = ...
def add_collection_def(meta_graph_def, key, graph=..., export_scope=..., exclude_nodes=..., override_contents=...):
  """Adds a collection to MetaGraphDef protocol buffer.

  Args:
    meta_graph_def: MetaGraphDef protocol buffer.
    key: One of the GraphKeys or user-defined string.
    graph: The `Graph` from which to get collections.
    export_scope: Optional `string`. Name scope to remove.
    exclude_nodes: An iterable of nodes or `string` node names to omit from the
      collection, or None.
    override_contents: An iterable of values to place in the collection,
      ignoring the current values (if set).
  """
  ...

def strip_graph_default_valued_attrs(meta_graph_def): # -> None:
  """Strips default valued attributes for node defs in given MetaGraphDef.

  This method also sets `meta_info_def.stripped_default_attrs` in the given
  `MetaGraphDef` proto to True.

  Args:
    meta_graph_def: `MetaGraphDef` protocol buffer

  Returns:
    None.
  """
  ...

def create_meta_graph_def(meta_info_def=..., graph_def=..., saver_def=..., collection_list=..., graph=..., export_scope=..., exclude_nodes=..., clear_extraneous_savers=..., strip_default_attrs=...): # -> MetaGraphDef:
  """Construct and returns a `MetaGraphDef` protocol buffer.

  Args:
    meta_info_def: `MetaInfoDef` protocol buffer.
    graph_def: `GraphDef` protocol buffer.
    saver_def: `SaverDef` protocol buffer.
    collection_list: List of string keys to collect.
    graph: The `Graph` to create `MetaGraphDef` out of.
    export_scope: Optional `string`. Name scope to remove.
    exclude_nodes: An iterable of nodes or `string` node names to omit from all
      collection, or None.
    clear_extraneous_savers: Remove any preexisting SaverDefs from the SAVERS
        collection.  Note this method does not alter the graph, so any
        extraneous Save/Restore ops should have been removed already, as needed.
    strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the NodeDefs. For a detailed guide, see
        [Stripping Default-Valued Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).

  Returns:
    MetaGraphDef protocol buffer.

  Raises:
    TypeError: If the arguments are not of the correct proto buffer type.
  """
  ...

def read_meta_graph_file(filename): # -> MetaGraphDef:
  """Reads a file containing `MetaGraphDef` and returns the protocol buffer.

  Args:
    filename: `meta_graph_def` filename including the path.

  Returns:
    A `MetaGraphDef` protocol buffer.

  Raises:
    IOError: If the file doesn't exist, or cannot be successfully parsed.
  """
  ...

def import_scoped_meta_graph(meta_graph_or_file, clear_devices=..., graph=..., import_scope=..., input_map=..., unbound_inputs_col_name=..., restore_collections_predicate=...):
  """Recreates a `Graph` saved in a `MetaGraphDef` proto.

  This function takes a `MetaGraphDef` protocol buffer as input. If
  the argument is a file containing a `MetaGraphDef` protocol buffer ,
  it constructs a protocol buffer from the file content. The function
  then adds all the nodes from the `graph_def` field to the
  current graph, recreates the desired collections, and returns a dictionary of
  all the Variables imported into the name scope.

  In combination with `export_scoped_meta_graph()`, this function can be used to

  * Serialize a graph along with other Python objects such as `QueueRunner`,
    `Variable` into a `MetaGraphDef`.

  * Restart training from a saved graph and checkpoints.

  * Run inference from a saved graph and checkpoints.

  Args:
    meta_graph_or_file: `MetaGraphDef` protocol buffer or filename (including
      the path) containing a `MetaGraphDef`.
    clear_devices: Boolean which controls whether to clear device information
      from graph_def. Default false.
    graph: The `Graph` to import into. If `None`, use the default graph.
    import_scope: Optional `string`. Name scope into which to import the
      subgraph. If `None`, the graph is imported to the root name scope.
    input_map: A dictionary mapping input names (as strings) in `graph_def` to
      `Tensor` objects. The values of the named input tensors in the imported
      graph will be re-mapped to the respective `Tensor` values.
    unbound_inputs_col_name: Collection name for looking up unbound inputs.
    restore_collections_predicate: a predicate on collection names. A collection
      named c (i.e whose key is c) will be restored iff
      1) `restore_collections_predicate(c)` is True, and
      2) `c != unbound_inputs_col_name`.

  Returns:
    A dictionary of all the `Variables` imported into the name scope.

  Raises:
    ValueError: If the graph_def contains unbound inputs.
  """
  ...

def import_scoped_meta_graph_with_return_elements(meta_graph_or_file, clear_devices=..., graph=..., import_scope=..., input_map=..., unbound_inputs_col_name=..., restore_collections_predicate=..., return_elements=...):
  """Imports graph from `MetaGraphDef` and returns vars and return elements.

  This function takes a `MetaGraphDef` protocol buffer as input. If
  the argument is a file containing a `MetaGraphDef` protocol buffer ,
  it constructs a protocol buffer from the file content. The function
  then adds all the nodes from the `graph_def` field to the
  current graph, recreates the desired collections, and returns a dictionary of
  all the Variables imported into the name scope.

  In combination with `export_scoped_meta_graph()`, this function can be used to

  * Serialize a graph along with other Python objects such as `QueueRunner`,
    `Variable` into a `MetaGraphDef`.

  * Restart training from a saved graph and checkpoints.

  * Run inference from a saved graph and checkpoints.

  Args:
    meta_graph_or_file: `MetaGraphDef` protocol buffer or filename (including
      the path) containing a `MetaGraphDef`.
    clear_devices: Boolean which controls whether to clear device information
      from graph_def. Default false.
    graph: The `Graph` to import into. If `None`, use the default graph.
    import_scope: Optional `string`. Name scope into which to import the
      subgraph. If `None`, the graph is imported to the root name scope.
    input_map: A dictionary mapping input names (as strings) in `graph_def` to
      `Tensor` objects. The values of the named input tensors in the imported
      graph will be re-mapped to the respective `Tensor` values.
    unbound_inputs_col_name: Collection name for looking up unbound inputs.
    restore_collections_predicate: a predicate on collection names. A collection
      named c (i.e whose key is c) will be restored iff
      1) `restore_collections_predicate(c)` is True, and
      2) `c != unbound_inputs_col_name`.
    return_elements:  A list of strings containing operation names in the
      `MetaGraphDef` that will be returned as `Operation` objects; and/or
      tensor names in `MetaGraphDef` that will be returned as `Tensor` objects.

  Returns:
    A tuple of (
      dictionary of all the `Variables` imported into the name scope,
      list of `Operation` or `Tensor` objects from the `return_elements` list).

  Raises:
    ValueError: If the graph_def contains unbound inputs.

  """
  ...

def export_scoped_meta_graph(filename=..., graph_def=..., graph=..., export_scope=..., as_text=..., unbound_inputs_col_name=..., clear_devices=..., saver_def=..., clear_extraneous_savers=..., strip_default_attrs=..., save_debug_info=..., **kwargs): # -> tuple[MetaGraphDef, dict[Unknown, Unknown]]:
  """Returns `MetaGraphDef` proto. Optionally writes it to filename.

  This function exports the graph, saver, and collection objects into
  `MetaGraphDef` protocol buffer with the intention of it being imported
  at a later time or location to restart training, run inference, or be
  a subgraph.

  Args:
    filename: Optional filename including the path for writing the
      generated `MetaGraphDef` protocol buffer.
    graph_def: `GraphDef` protocol buffer.
    graph: The `Graph` to export. If `None`, use the default graph.
    export_scope: Optional `string`. Name scope under which to extract
      the subgraph. The scope name will be stripped from the node definitions
      for easy import later into new name scopes. If `None`, the whole graph
      is exported.
    as_text: If `True`, writes the `MetaGraphDef` as an ASCII proto.
    unbound_inputs_col_name: Optional `string`. If provided, a string collection
      with the given name will be added to the returned `MetaGraphDef`,
      containing the names of tensors that must be remapped when importing the
      `MetaGraphDef`.
    clear_devices: Boolean which controls whether to clear device information
      before exporting the graph.
    saver_def: `SaverDef` protocol buffer.
    clear_extraneous_savers: Remove any Saver-related information from the
        graph (both Save/Restore ops and SaverDefs) that are not associated
        with the provided SaverDef.
    strip_default_attrs: Set to true if default valued attributes must be
      removed while exporting the GraphDef.
    save_debug_info: If `True`, save the GraphDebugInfo to a separate file,
      which in the same directory of filename and with `_debug` added before the
      file extension.
    **kwargs: Optional keyed arguments, including meta_info_def and
        collection_list.

  Returns:
    A `MetaGraphDef` proto and dictionary of `Variables` in the exported
    name scope.

  Raises:
    ValueError: When the `GraphDef` is larger than 2GB.
    ValueError: When executing in Eager mode and either `graph_def` or `graph`
      is undefined.
  """
  ...

def copy_scoped_meta_graph(from_scope, to_scope, from_graph=..., to_graph=...):
  """Copies a sub-meta_graph from one scope to another.

  Args:
    from_scope: `String` name scope containing the subgraph to be copied.
    to_scope: `String` name scope under which the copied subgraph will reside.
    from_graph: Optional `Graph` from which to copy the subgraph. If `None`, the
      default graph is use.
    to_graph: Optional `Graph` to which to copy the subgraph. If `None`, the
      default graph is used.

  Returns:
    A dictionary of `Variables` that has been copied into `to_scope`.

  Raises:
    ValueError: If `from_scope` and `to_scope` are the same while
      `from_graph` and `to_graph` are also the same.
  """
  ...

