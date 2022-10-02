"""
This type stub file was generated by pyright.
"""

import collections
from tensorflow.python.util.tf_export import tf_export

"""A SessionRunHook extends `session.run()` calls for the `MonitoredSession`.

SessionRunHooks are useful to track training, report progress, request early
stopping and more. SessionRunHooks use the observer pattern and notify at the
following points:
 - when a session starts being used
 - before a call to the `session.run()`
 - after a call to the `session.run()`
 - when the session closed

A SessionRunHook encapsulates a piece of reusable/composable computation that
can piggyback a call to `MonitoredSession.run()`. A hook can add any
ops-or-tensor/feeds to the run call, and when the run call finishes with success
gets the outputs it requested. Hooks are allowed to add ops to the graph in
`hook.begin()`. The graph is finalized after the `begin()` method is called.

There are a few pre-defined hooks:
 - StopAtStepHook: Request stop based on global_step
 - CheckpointSaverHook: saves checkpoint
 - LoggingTensorHook: outputs one or more tensor values to log
 - NanTensorHook: Request stop if given `Tensor` contains Nans.
 - SummarySaverHook: saves summaries to a summary writer

For more specific needs, you can create custom hooks:
  class ExampleHook(SessionRunHook):
    def begin(self):
      # You can add ops to the graph here.
      print('Starting the session.')
      self.your_tensor = ...

    def after_create_session(self, session, coord):
      # When this is called, the graph is finalized and
      # ops can no longer be added to the graph.
      print('Session created.')

    def before_run(self, run_context):
      print('Before calling session.run().')
      return SessionRunArgs(self.your_tensor)

    def after_run(self, run_context, run_values):
      print('Done running one step. The value of my tensor: %s',
            run_values.results)
      if you-need-to-stop-loop:
        run_context.request_stop()

    def end(self, session):
      print('Done with the session.')

To understand how hooks interact with calls to `MonitoredSession.run()`,
look at following code:
  with MonitoredTrainingSession(hooks=your_hooks, ...) as sess:
    while not sess.should_stop():
      sess.run(your_fetches)

Above user code leads to following execution:
  call hooks.begin()
  sess = tf.compat.v1.Session()
  call hooks.after_create_session()
  while not stop is requested:
    call hooks.before_run()
    try:
      results = sess.run(merged_fetches, feed_dict=merged_feeds)
    except (errors.OutOfRangeError, StopIteration):
      break
    call hooks.after_run()
  call hooks.end()
  sess.close()

Note that if sess.run() raises OutOfRangeError or StopIteration then
hooks.after_run() will not be called but hooks.end() will still be called.
If sess.run() raises any other exception then neither hooks.after_run() nor
hooks.end() will be called.
"""
@tf_export(v1=["train.SessionRunHook"])
class SessionRunHook:
  """Hook to extend calls to MonitoredSession.run()."""
  def begin(self): # -> None:
    """Called once before using the session.

    When called, the default graph is the one that will be launched in the
    session.  The hook can modify the graph by adding new operations to it.
    After the `begin()` call the graph will be finalized and the other callbacks
    can not modify the graph anymore. Second call of `begin()` on the same
    graph, should not change the graph.
    """
    ...
  
  def after_create_session(self, session, coord): # -> None:
    """Called when new TensorFlow session is created.

    This is called to signal the hooks that a new session has been created. This
    has two essential differences with the situation in which `begin` is called:

    * When this is called, the graph is finalized and ops can no longer be added
        to the graph.
    * This method will also be called as a result of recovering a wrapped
        session, not only at the beginning of the overall session.

    Args:
      session: A TensorFlow Session that has been created.
      coord: A Coordinator object which keeps track of all threads.
    """
    ...
  
  def before_run(self, run_context): # -> None:
    """Called before each call to run().

    You can return from this call a `SessionRunArgs` object indicating ops or
    tensors to add to the upcoming `run()` call.  These ops/tensors will be run
    together with the ops/tensors originally passed to the original run() call.
    The run args you return can also contain feeds to be added to the run()
    call.

    The `run_context` argument is a `SessionRunContext` that provides
    information about the upcoming `run()` call: the originally requested
    op/tensors, the TensorFlow Session.

    At this point graph is finalized and you can not add ops.

    Args:
      run_context: A `SessionRunContext` object.

    Returns:
      None or a `SessionRunArgs` object.
    """
    ...
  
  def after_run(self, run_context, run_values): # -> None:
    """Called after each call to run().

    The `run_values` argument contains results of requested ops/tensors by
    `before_run()`.

    The `run_context` argument is the same one send to `before_run` call.
    `run_context.request_stop()` can be called to stop the iteration.

    If `session.run()` raises any exceptions then `after_run()` is not called.

    Args:
      run_context: A `SessionRunContext` object.
      run_values: A SessionRunValues object.
    """
    ...
  
  def end(self, session): # -> None:
    """Called at the end of session.

    The `session` argument can be used in case the hook wants to run final ops,
    such as saving a last checkpoint.

    If `session.run()` raises exception other than OutOfRangeError or
    StopIteration then `end()` is not called.
    Note the difference between `end()` and `after_run()` behavior when
    `session.run()` raises OutOfRangeError or StopIteration. In that case
    `end()` is called but `after_run()` is not called.

    Args:
      session: A TensorFlow Session that will be soon closed.
    """
    ...
  


@tf_export(v1=["train.SessionRunArgs"])
class SessionRunArgs(collections.namedtuple("SessionRunArgs", ["fetches", "feed_dict", "options"])):
  """Represents arguments to be added to a `Session.run()` call.

  Args:
    fetches: Exactly like the 'fetches' argument to Session.Run().
      Can be a single tensor or op, a list of 'fetches' or a dictionary
      of fetches.  For example:
        fetches = global_step_tensor
        fetches = [train_op, summary_op, global_step_tensor]
        fetches = {'step': global_step_tensor, 'summ': summary_op}
      Note that this can recurse as expected:
        fetches = {'step': global_step_tensor,
                   'ops': [train_op, check_nan_op]}
    feed_dict: Exactly like the `feed_dict` argument to `Session.Run()`
    options: Exactly like the `options` argument to `Session.run()`, i.e., a
      config_pb2.RunOptions proto.
  """
  def __new__(cls, fetches, feed_dict=..., options=...): # -> Self@SessionRunArgs:
    ...
  


@tf_export(v1=["train.SessionRunContext"])
class SessionRunContext:
  """Provides information about the `session.run()` call being made.

  Provides information about original request to `Session.Run()` function.
  SessionRunHook objects can stop the loop by calling `request_stop()` of
  `run_context`. In the future we may use this object to add more information
  about run without changing the Hook API.
  """
  def __init__(self, original_args, session) -> None:
    """Initializes SessionRunContext."""
    ...
  
  @property
  def original_args(self): # -> Unknown:
    """A `SessionRunArgs` object holding the original arguments of `run()`.

    If user called `MonitoredSession.run(fetches=a, feed_dict=b)`, then this
    field is equal to SessionRunArgs(a, b).

    Returns:
     A `SessionRunArgs` object
    """
    ...
  
  @property
  def session(self): # -> Unknown:
    """A TensorFlow session object which will execute the `run`."""
    ...
  
  @property
  def stop_requested(self): # -> bool:
    """Returns whether a stop is requested or not.

    If true, `MonitoredSession` stops iterations.
    Returns:
      A `bool`
    """
    ...
  
  def request_stop(self): # -> None:
    """Sets stop requested field.

    Hooks can use this function to request stop of iterations.
    `MonitoredSession` checks whether this is called or not.
    """
    ...
  


@tf_export(v1=["train.SessionRunValues"])
class SessionRunValues(collections.namedtuple("SessionRunValues", ["results", "options", "run_metadata"])):
  """Contains the results of `Session.run()`.

  In the future we may use this object to add more information about result of
  run without changing the Hook API.

  Args:
    results: The return values from `Session.run()` corresponding to the fetches
      attribute returned in the RunArgs. Note that this has the same shape as
      the RunArgs fetches.  For example:
        fetches = global_step_tensor
        => results = nparray(int)
        fetches = [train_op, summary_op, global_step_tensor]
        => results = [None, nparray(string), nparray(int)]
        fetches = {'step': global_step_tensor, 'summ': summary_op}
        => results = {'step': nparray(int), 'summ': nparray(string)}
    options: `RunOptions` from the `Session.run()` call.
    run_metadata: `RunMetadata` from the `Session.run()` call.
  """
  ...

