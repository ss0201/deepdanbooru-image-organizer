"""
This type stub file was generated by pyright.
"""

import threading
from abc import abstractmethod
from tensorflow.python.util.tf_export import keras_export

"""Utilities for file download and caching."""
if True:
    def urlretrieve(url, filename, reporthook=..., data=...): # -> None:
        """Replacement for `urlretrieve` for Python 2.

        Under Python 2, `urlretrieve` relies on `FancyURLopener` from legacy
        `urllib` module, known to have issues with proxy management.

        Args:
            url: url to retrieve.
            filename: where to store the retrieved data locally.
            reporthook: a hook function that will be called once on
              establishment of the network connection and once after each block
              read thereafter. The hook will be passed three arguments; a count
              of blocks transferred so far, a block size in bytes, and the total
              size of the file.
            data: `data` argument passed to `urlopen`.
        """
        ...
    
else:
    ...
def is_generator_or_sequence(x): # -> bool:
    """Check if `x` is a Keras generator type."""
    ...

@keras_export("keras.utils.get_file")
def get_file(fname=..., origin=..., untar=..., md5_hash=..., file_hash=..., cache_subdir=..., hash_algorithm=..., extract=..., archive_format=..., cache_dir=...):
    """Downloads a file from a URL if it not already in the cache.

    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.keras/datasets/example.txt`.

    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.

    Example:

    ```python
    path_to_downloaded_file = tf.keras.utils.get_file(
        "flower_photos",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
        untar=True)
    ```

    Args:
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location. If `None`, the
            name of the file at `origin` will be used.
        origin: Original URL of the file.
        untar: Deprecated in favor of `extract` argument.
            boolean, whether the file should be decompressed
        md5_hash: Deprecated in favor of `file_hash` argument.
            md5 hash of the file for verification
        file_hash: The expected hash string of the file after download.
            The sha256 and md5 hash algorithms are both supported.
        cache_subdir: Subdirectory under the Keras cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        hash_algorithm: Select the hash algorithm to verify the file.
            options are `'md5'`, `'sha256'`, and `'auto'`.
            The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
            Options are `'auto'`, `'tar'`, `'zip'`, and `None`.
            `'tar'` includes tar, tar.gz, and tar.bz files.
            The default `'auto'` corresponds to `['tar', 'zip']`.
            None or an empty list will return no matches found.
        cache_dir: Location to store cached files, when None it
            defaults to the default directory `~/.keras/`.

    Returns:
        Path to the downloaded file
    """
    ...

def validate_file(fpath, file_hash, algorithm=..., chunk_size=...): # -> bool:
    """Validates a file against a sha256 or md5 hash.

    Args:
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    Returns:
        Whether the file is valid
    """
    ...

class ThreadsafeIter:
    """Wrap an iterator with a lock and propagate exceptions to all threads."""
    def __init__(self, it) -> None:
        ...
    
    def __iter__(self): # -> Self@ThreadsafeIter:
        ...
    
    def next(self):
        ...
    
    def __next__(self):
        ...
    


def threadsafe_generator(f): # -> (*a: Unknown, **kw: Unknown) -> ThreadsafeIter:
    ...

@keras_export("keras.utils.Sequence")
class Sequence:
    """Base object for fitting to a sequence of data, such as a dataset.

    Every `Sequence` must implement the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement
    `on_epoch_end`.
    The method `__getitem__` should return a complete batch.

    Notes:

    `Sequence` are a safer way to do multiprocessing. This structure guarantees
    that the network will only train once
     on each sample per epoch which is not the case with generators.

    Examples:

    ```python
    from skimage.io import imread
    from skimage.transform import resize
    import numpy as np
    import math

    # Here, `x_set` is list of path to the images
    # and `y_set` are the associated classes.

    class CIFAR10Sequence(tf.keras.utils.Sequence):

        def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size

        def __len__(self):
            return math.ceil(len(self.x) / self.batch_size)

        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) *
            self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) *
            self.batch_size]

            return np.array([
                resize(imread(file_name), (200, 200))
                   for file_name in batch_x]), np.array(batch_y)
    ```
    """
    @abstractmethod
    def __getitem__(self, index):
        """Gets batch at position `index`.

        Args:
            index: position of the batch in the Sequence.

        Returns:
            A batch
        """
        ...
    
    @abstractmethod
    def __len__(self):
        """Number of batch in the Sequence.

        Returns:
            The number of batches in the Sequence.
        """
        ...
    
    def on_epoch_end(self): # -> None:
        """Method called at the end of every epoch."""
        ...
    
    def __iter__(self): # -> Generator[Unknown, None, None]:
        """Create a generator that iterate over the Sequence."""
        ...
    


def iter_sequence_infinite(seq): # -> Generator[Unknown, None, None]:
    """Iterates indefinitely over a Sequence.

    Args:
      seq: `Sequence` instance.

    Yields:
      Batches of data from the `Sequence`.
    """
    ...

_SHARED_SEQUENCES = ...
_SEQUENCE_COUNTER = ...
_DATA_POOLS = ...
_WORKER_ID_QUEUE = ...
_WORKER_IDS = ...
_FORCE_THREADPOOL = ...
_FORCE_THREADPOOL_LOCK = threading.RLock()
def dont_use_multiprocessing_pool(f): # -> (*args: Unknown, **kwargs: Unknown) -> Unknown:
    ...

def get_pool_class(use_multiprocessing): # -> (processes: int | None = ..., initializer: ((...) -> object) | None = ..., initargs: Iterable[Any] = ...) -> Any:
    ...

def get_worker_id_queue():
    """Lazily create the queue to track worker ids."""
    ...

def init_pool(seqs): # -> None:
    ...

def get_index(uid, i):
    """Get the value from the Sequence `uid` at index `i`.

    To allow multiple Sequences to be used at the same time, we use `uid` to
    get a specific one. A single Sequence would cause the validation to
    overwrite the training Sequence.

    Args:
        uid: int, Sequence identifier
        i: index

    Returns:
        The value at index `i`.
    """
    ...

@keras_export("keras.utils.SequenceEnqueuer")
class SequenceEnqueuer:
    """Base class to enqueue inputs.

    The task of an Enqueuer is to use parallelism to speed up preprocessing.
    This is done with processes or threads.

    Example:

    ```python
        enqueuer = SequenceEnqueuer(...)
        enqueuer.start()
        datas = enqueuer.get()
        for data in datas:
            # Use the inputs; training, evaluating, predicting.
            # ... stop sometime.
        enqueuer.stop()
    ```

    The `enqueuer.get()` should be an infinite stream of data.
    """
    def __init__(self, sequence, use_multiprocessing=...) -> None:
        ...
    
    def is_running(self): # -> bool:
        ...
    
    def start(self, workers=..., max_queue_size=...): # -> None:
        """Starts the handler's workers.

        Args:
            workers: Number of workers.
            max_queue_size: queue size
                (when full, workers could block on `put()`)
        """
        ...
    
    def stop(self, timeout=...): # -> None:
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        Args:
            timeout: maximum time to wait on `thread.join()`
        """
        ...
    
    def __del__(self): # -> None:
        ...
    
    @abstractmethod
    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.
        # Returns
            Generator yielding tuples `(inputs, targets)`
                or `(inputs, targets, sample_weights)`.
        """
        ...
    


@keras_export("keras.utils.OrderedEnqueuer")
class OrderedEnqueuer(SequenceEnqueuer):
    """Builds a Enqueuer from a Sequence.

    Args:
        sequence: A `tf.keras.utils.data_utils.Sequence` object.
        use_multiprocessing: use multiprocessing if True, otherwise threading
        shuffle: whether to shuffle the data at the beginning of each epoch
    """
    def __init__(self, sequence, use_multiprocessing=..., shuffle=...) -> None:
        ...
    
    def get(self): # -> Generator[Unknown, None, None]:
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        Yields:
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        ...
    


def init_pool_generator(gens, random_seed=..., id_queue=...): # -> None:
    """Initializer function for pool workers.

    Args:
      gens: State which should be made available to worker processes.
      random_seed: An optional value with which to seed child processes.
      id_queue: A multiprocessing Queue of worker ids. This is used to indicate
        that a worker process was created by Keras and can be terminated using
        the cleanup_all_keras_forkpools utility.
    """
    ...

def next_sample(uid):
    """Gets the next value from the generator `uid`.

    To allow multiple generators to be used at the same time, we use `uid` to
    get a specific one. A single generator would cause the validation to
    overwrite the training generator.

    Args:
        uid: int, generator identifier

    Returns:
        The next value of generator `uid`.
    """
    ...

@keras_export("keras.utils.GeneratorEnqueuer")
class GeneratorEnqueuer(SequenceEnqueuer):
    """Builds a queue out of a data generator.

    The provided generator can be finite in which case the class will throw
    a `StopIteration` exception.

    Args:
        generator: a generator function which yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        random_seed: Initial seed for workers,
            will be incremented by one for each worker.
    """
    def __init__(self, generator, use_multiprocessing=..., random_seed=...) -> None:
        ...
    
    def get(self): # -> Generator[Unknown, None, None]:
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        Yields:
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        ...
    


@keras_export("keras.utils.pad_sequences", "keras.preprocessing.sequence.pad_sequences")
def pad_sequences(sequences, maxlen=..., dtype=..., padding=..., truncating=..., value=...): # -> ndarray[Unknown, Unknown]:
    """Pads sequences to the same length.

    This function transforms a list (of length `num_samples`)
    of sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence in the list.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` until they are `num_timesteps` long.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.

    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding or removing values from the beginning of the sequence is the
    default.

    >>> sequence = [[1], [2, 3], [4, 5, 6]]
    >>> tf.keras.preprocessing.sequence.pad_sequences(sequence)
    array([[0, 0, 1],
           [0, 2, 3],
           [4, 5, 6]], dtype=int32)

    >>> tf.keras.preprocessing.sequence.pad_sequences(sequence, value=-1)
    array([[-1, -1,  1],
           [-1,  2,  3],
           [ 4,  5,  6]], dtype=int32)

    >>> tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post')
    array([[1, 0, 0],
           [2, 3, 0],
           [4, 5, 6]], dtype=int32)

    >>> tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=2)
    array([[0, 1],
           [2, 3],
           [5, 6]], dtype=int32)

    Args:
        sequences: List of sequences (each sequence is a list of integers).
        maxlen: Optional Int, maximum length of all sequences. If not provided,
            sequences will be padded to the length of the longest individual
            sequence.
        dtype: (Optional, defaults to `"int32"`). Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, "pre" or "post" (optional, defaults to `"pre"`):
            pad either before or after each sequence.
        truncating: String, "pre" or "post" (optional, defaults to `"pre"`):
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value. (Optional, defaults to 0.)

    Returns:
        Numpy array with shape `(len(sequences), maxlen)`

    Raises:
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    ...

