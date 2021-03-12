import random as pyrandom
from torch import nn
from collections.abc import Callable
from typing import Union
from loguru import logger
import torch
import numpy as np

layer_types_not_initialized = set()


def get_init_weights(func: Callable, bias: Union[bool, float] = 0.01) -> Callable:
    """
    Constructs a weight initialization function to be applied to a pytorch module object

    Args:
        func: callable
        bias: if True, apply the func to the bias. Otherwise, set the bias to the float value. Do not pass false.

    Returns:
        callable

    """

    def init_weights(m):
        if not type(m) == nn.BatchNorm2d:
            try:
                try:
                    func(m.weight)
                    if type(bias) == bool and bias:
                        func(m.bias)
                    else:
                        if type(bias) == bool and not bias:
                            logger.warning(
                                "Received false as apply to bias, assuming bias to be 0.00"
                            )
                            apply_to_bias = 0.0
                        m.bias.data.fill_(apply_to_bias)

                except AttributeError as e:
                    layer_types_not_initialized.add(str(e))

            except:
                print(m)
                raise

    return init_weights


def initialize_null(model: torch.nn.Module) -> torch.nn.Module:
    """
    Sets all weights to negative 100, so we can check if weights were set correctly via a future
    initialization method.

    Args:
        model:

    Returns:
        model
    """

    def zero_out(m):
        return torch.nn.init.constant_(m, -100.0)

    init_weights = get_init_weights(zero_out, bias=True)
    model.apply(init_weights)
    return model


def assert_non_null_weights(statedict: dict):
    """
    Checks if a statedict has weights defined as null (-100)
    Args:
        statedict:

    Returns:

    """
    for w in statedict.items():
        assert not np.any(w[1].cpu().numpy() == -100.0), (
            w[0] + "contains parameters that were likely not initialized"
        )
