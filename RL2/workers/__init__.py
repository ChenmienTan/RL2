from .base import Worker

def initialize_actor(config, train):
    """
    Initialize actor(s) for training.

    Returns:
        - Single actor if shared_policy=true or multi_agent disabled
        - Dict of actors if shared_policy=false
    """
    from hydra.core.hydra_config import HydraConfig
    hydra_config = HydraConfig.get()
    backend = hydra_config.runtime.choices.get(
        "actor" if train else "ref_actor"
    )

    # Check if multi-agent with independent policy
    is_multi_agent = hasattr(config, 'multi_agent') and config.multi_agent.enabled
    is_independent = is_multi_agent and not config.multi_agent.shared_policy

    if backend == "fsdp":
        from .fsdp.actor import FSDPActor
        actor_class = FSDPActor
    elif backend == "megatron":
        from .megatron.actor import MegatronActor
        actor_class = MegatronActor
    else:
        raise NotImplementedError

    # Independent policy: create dict of actors
    if is_independent:
        # Will be populated lazily when we see agent IDs
        return {}

    # Shared policy or single agent: create one actor
    return actor_class(config, train)

def initialize_critic(config):

    from hydra.core.hydra_config import HydraConfig
    hydra_config = HydraConfig.get()
    backend = hydra_config.runtime.choices.get("critic")
    if backend == "fsdp":
        from .fsdp.critic import FSDPCritic
        return FSDPCritic(config)
    elif backend == "megatron":
        from .megatron.critic import MegatronCritic
        return MegatronCritic(config)
    else:
        raise NotImplementedError

def initialize_rollout(config):

    from .rollout import Rollout
    return Rollout(config)
