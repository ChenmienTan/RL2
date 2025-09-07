"""
Dynamic Rollout Class Loader for RL2

This module provides utilities for dynamically loading and instantiating
rollout classes based on configuration. Uses AgentRollout when agent_class 
is specified, otherwise uses regular Rollout.
"""

import logging

def has_agent_class(config):
    return hasattr(config, 'agent_class') and config.agent_class is not None

def Rollout(config):
    if has_agent_class(config):
        from RL2.utils.agent import AgentRollout
        logging.info(f"Using AgentRollout with agent_class: {config.agent_class}")
        return AgentRollout(config)
    else:
        from RL2.workers.rollout import Rollout
        return Rollout(config)