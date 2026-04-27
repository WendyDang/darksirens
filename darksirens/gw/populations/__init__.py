# populations/__init__.py
from .registry import (
    get_model, 
    pop_model_parser, 
    pop_model_prior_parser, 
    get_fixed_population_params
)

__all__ = [
    "get_model", 
    "pop_model_parser", 
    "pop_model_prior_parser", 
    "get_fixed_population_params"
]