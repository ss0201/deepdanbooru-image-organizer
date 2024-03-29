"""
This type stub file was generated by pyright.
"""

"""Utility object to handler partial batches for TPUStrategy."""
class PartialBatchPaddingHandler:
    """A container that holds info about partial batches for `predict()`."""
    def __init__(self, output_shape) -> None:
        ...
    
    def get_real_batch_size(self, dataset_batch):
        """Returns the number of elements in a potentially partial batch."""
        ...
    
    def update_mask(self, padding_mask, dataset_batch):
        """Calculate and cache the amount of padding required for a batch."""
        ...
    
    def pad_batch(self, *dataset_batch_elements): # -> tuple[Unknown, ...] | dict[Unknown, Unknown]:
        """Pads the batch dimension of a tensor to the complete batch size."""
        ...
    
    def apply_mask(self, prediction_result): # -> Any | list[Unknown]:
        """Removes prediction output that corresponds to padded input."""
        ...
    


