"""
Rollout Wrapper for RL2

This module provides a wrapper that automatically selects the appropriate
rollout implementation based on configuration, enabling seamless GEM integration
without modifying core RL2 code.
"""

import logging

# Import the standard rollout
from RL2.workers.rollout import Rollout as StandardRollout

# Try to import GEM rollout if available
try:
    from RL2.workers.gem_rollout import GEMRollout
    GEM_AVAILABLE = True
except ImportError:
    GEM_AVAILABLE = False
    GEMRollout = None


def Rollout(config):
    """
    Factory function that returns the appropriate Rollout worker
    based on configuration.
    
    This allows GEM environments to be used without modifying
    the PPO trainer or other core RL2 components.
    
    Args:
        config: Configuration object
        
    Returns:
        Either StandardRollout or GEMRollout instance
    """
    # Check if GEM environment is requested
    if hasattr(config, 'use_gem_env') and config.use_gem_env:
        if not GEM_AVAILABLE:
            raise ImportError(
                "GEM rollout worker is not available. Please install GEM:\n"
                "pip install gem-rl\n"
                "Or install from source:\n"
                "git clone https://github.com/axonrl/gem.git && cd gem && pip install -e ."
            )
        logging.info("Using GEM rollout worker for environment interaction")
        return GEMRollout(config)
    else:
        # Standard RL2 rollout behavior
        return StandardRollout(config)