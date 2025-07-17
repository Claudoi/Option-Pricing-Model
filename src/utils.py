# Utils Placeholder
import numpy as np
from src.constants import VALID_OPTION_TYPES, VALID_BARRIER_TYPES, ERROR_OPTION_TYPE, ERROR_BARRIER_TYPE, ERROR_INVALID_INPUTS



def validate_option_type(option_type):
    """
    Raises an error if the option type is invalid.
    """
    if option_type not in VALID_OPTION_TYPES:
        raise ValueError(ERROR_OPTION_TYPE)



def validate_barrier_type(barrier_type):
    """
    Raises an error if the barrier type is invalid.
    """
    if barrier_type not in VALID_BARRIER_TYPES:
        raise ValueError(ERROR_BARRIER_TYPE)



def validate_positive_inputs(*args):
    """
    Checks that all numeric inputs are strictly positive.
    """
    if any(param <= 0 for param in args):
        raise ValueError(ERROR_INVALID_INPUTS)



def calculate_payoff(ST, K, option_type):
    """
    Calculates the option payoff at maturity for European call or put.
    """
    if option_type == "call":
        return np.maximum(ST - K, 0)
    elif option_type == "put":
        return np.maximum(K - ST, 0)
    else:
        raise ValueError(ERROR_OPTION_TYPE)
