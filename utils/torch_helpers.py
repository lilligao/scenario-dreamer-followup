import numpy as np
import torch
from typing import Tuple, Dict, Optional

def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data

def remove_prefix(state_dict: Dict, prefix: str) -> Dict:
    result = dict()

    for k, v in state_dict.items():
        tokens = k.split('.')

        if tokens[0] == prefix:
            tokens = tokens[1:]

        key = '.'.join(tokens)
        result[key] = v

    return result

def load_encoder_weights(model, ckpt_path, prefix='backbone'):
    """
    Loads encoder weights from a checkpoint, removing a given prefix.

    Args:
        model (torch.nn.Module): The encoder model to load weights into.
        ckpt_path (str): Path to the checkpoint file.
        prefix (str): Prefix to remove from state_dict keys.

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    checkpoint = torch.load(ckpt_path) #, map_location='cpu'
    state_dict = remove_prefix(checkpoint['state_dict'], prefix)
    model.load_state_dict(state_dict)
    return model
