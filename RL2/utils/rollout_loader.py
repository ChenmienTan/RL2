"""
Dynamic Rollout Class Loader for RL2

This module provides utilities for dynamically loading custom rollout classes
based on configuration, enabling flexible rollout implementation selection.
"""

import importlib
import logging


def load_rollout_class(config):
    """
    Load rollout class dynamically based on configuration.
    
    Args:
        config: Rollout configuration object that may contain 'rollout_class' attribute
        
    Returns:
        Rollout class (uninstantiated)
        
    Raises:
        ImportError: If the specified module cannot be imported
        AttributeError: If the specified class doesn't exist in the module
        TypeError: If the specified class doesn't inherit from base Rollout
    """
    # Default to standard rollout if no custom class specified
    if not hasattr(config, 'rollout_class') or config.rollout_class is None:
        from RL2.workers.rollout import Rollout
        logging.debug("Using default Rollout class: RL2.workers.rollout.Rollout")
        return Rollout
    
    # Parse the fully qualified class name
    rollout_class_path = config.rollout_class
    if '.' not in rollout_class_path:
        raise ValueError(
            f"rollout_class must be a fully qualified class name "
            f"(e.g., 'module.Class'), got: '{rollout_class_path}'"
        )
    
    try:
        # Split module path and class name
        module_path, class_name = rollout_class_path.rsplit('.', 1)
        
        # Dynamic import
        module = importlib.import_module(module_path)
        rollout_class = getattr(module, class_name)
        
        # Validation: ensure it's a callable class
        if not isinstance(rollout_class, type):
            raise TypeError(f"'{rollout_class_path}' is not a class")
        
        # Validation: ensure it inherits from base Rollout
        from RL2.workers.rollout import Rollout as BaseRollout
        if not issubclass(rollout_class, BaseRollout):
            raise TypeError(
                f"'{rollout_class_path}' must inherit from RL2.workers.rollout.Rollout"
            )
        
        logging.info(f"Successfully loaded custom rollout class: {rollout_class_path}")
        return rollout_class
        
    except ImportError as e:
        raise ImportError(
            f"Could not import rollout class '{rollout_class_path}': {e}"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_path}': {e}"
        ) from e


def uses_custom_rollout(config):
    """
    Check if using custom rollout class vs default dataset-based rollout.
    
    Args:
        config: Rollout configuration object
        
    Returns:
        bool: True if custom rollout class is specified, False otherwise
    """
    return (hasattr(config, 'rollout_class') and 
            config.rollout_class is not None)