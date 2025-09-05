"""
Dynamic Rollout Class Loader for RL2

This module provides utilities for dynamically loading and instantiating
rollout classes based on configuration, with built-in tracking of custom usage.
"""

import importlib
import logging

def is_custom_rollout(config):
    return hasattr(config, 'rollout_class') and config.rollout_class is not None

def Rollout(config):
    if not is_custom_rollout(config):
        from RL2.workers.rollout import Rollout
        return Rollout(config)
    
    return _create_rollout(config)

def _create_rollout(config):
    rollout_class_path = config.rollout_class
    
    if '.' not in rollout_class_path:
        raise ValueError(
            f"rollout_class must be a fully qualified class name "
            f"(e.g., 'module.Class'), got: '{rollout_class_path}'"
        )
    
    try:
        module_path, class_name = rollout_class_path.rsplit('.', 1)
        
        module = importlib.import_module(module_path)
        rollout_class = getattr(module, class_name)
        
        if not isinstance(rollout_class, type):
            raise TypeError(f"'{rollout_class_path}' is not a class")
        
        from RL2.workers.rollout import Rollout as BaseRollout
        if not issubclass(rollout_class, BaseRollout):
            raise TypeError(
                f"'{rollout_class_path}' must inherit from RL2.workers.rollout.Rollout"
            )
        
        logging.info(f"Successfully loaded custom rollout class: {rollout_class_path}")
        return rollout_class(config)
        
    except ImportError as e:
        raise ImportError(
            f"Could not import rollout class '{rollout_class_path}': {e}"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_path}': {e}"
        ) from e