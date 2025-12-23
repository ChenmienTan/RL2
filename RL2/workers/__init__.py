from .base import Worker

def initialize_actor(config, train):

    from hydra.core.hydra_config import HydraConfig
    hydra_config = HydraConfig.get()
    backend = hydra_config.runtime.choices.get(
        "actor" if train else "ref_actor"
    )
    if backend == "fsdp":
        from .fsdp.actor import FSDPActor
        return FSDPActor(config, train)
    elif backend == "megatron":
        from .megatron.actor import MegatronActor
        return MegatronActor(config, train)
    else:
        raise NotImplementedError

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
