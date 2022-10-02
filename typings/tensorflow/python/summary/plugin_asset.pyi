"""
This type stub file was generated by pyright.
"""

import abc
import six

"""TensorBoard Plugin asset abstract class.

TensorBoard plugins may need to provide arbitrary assets, such as
configuration information for specific outputs, or vocabulary files, or sprite
images, etc.

This module contains methods that allow plugin assets to be specified at graph
construction time. Plugin authors define a PluginAsset which is treated as a
singleton on a per-graph basis. The PluginAsset has an assets method which
returns a dictionary of asset contents. The tf.compat.v1.summary.FileWriter
(or any other Summary writer) will serialize these assets in such a way that
TensorBoard can retrieve them.
"""
_PLUGIN_ASSET_PREFIX = ...
def get_plugin_asset(plugin_asset_cls, graph=...):
  """Acquire singleton PluginAsset instance from a graph.

  PluginAssets are always singletons, and are stored in tf Graph collections.
  This way, they can be defined anywhere the graph is being constructed, and
  if the same plugin is configured at many different points, the user can always
  modify the same instance.

  Args:
    plugin_asset_cls: The PluginAsset class
    graph: (optional) The graph to retrieve the instance from. If not specified,
      the default graph is used.

  Returns:
    An instance of the plugin_asset_class

  Raises:
    ValueError: If we have a plugin name collision, or if we unexpectedly find
      the wrong number of items in a collection.
  """
  ...

def get_all_plugin_assets(graph=...): # -> list[Unknown]:
  """Retrieve all PluginAssets stored in the graph collection.

  Args:
    graph: Optionally, the graph to get assets from. If unspecified, the default
      graph is used.

  Returns:
    A list with all PluginAsset instances in the graph.

  Raises:
    ValueError: if we unexpectedly find a collection with the wrong number of
      PluginAssets.

  """
  ...

@six.add_metaclass(abc.ABCMeta)
class PluginAsset:
  """This abstract base class allows TensorBoard to serialize assets to disk.

  Plugin authors are expected to extend the PluginAsset class, so that it:
  - has a unique plugin_name
  - provides an assets method that returns an {asset_name: asset_contents}
    dictionary. For now, asset_contents are strings, although we may add
    StringIO support later.

  LifeCycle of a PluginAsset instance:
  - It is constructed when get_plugin_asset is called on the class for
    the first time.
  - It is configured by code that follows the calls to get_plugin_asset
  - When the containing graph is serialized by the
    tf.compat.v1.summary.FileWriter, the writer calls assets and the
    PluginAsset instance provides its contents to be written to disk.
  """
  plugin_name = ...
  @abc.abstractmethod
  def assets(self):
    """Provide all of the assets contained by the PluginAsset instance.

    The assets method should return a dictionary structured as
    {asset_name: asset_contents}. asset_contents is a string.

    This method will be called by the tf.compat.v1.summary.FileWriter when it
    is time to write the assets out to disk.
    """
    ...
  

