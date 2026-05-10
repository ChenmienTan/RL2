from typing import Dict, Any, List, Callable, Tuple, Optional
from omegaconf import OmegaConf, DictConfig
import os
import json
import asyncio
import inspect
from enum import Enum
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import count
import torch
from transformers import AutoTokenizer
from RL2.datasets import get_tensor_dict, BaseDataset
from RL2.utils.communication import async_request


DEFAULT_AGENT_ID = "agent_0"
EPISODE_COUNTER = count()
PROMPT_GROUP_COUNTER = count()


class SampleStatus(Enum):

    RUNNING = "running"
    ABORTED = "aborted"
    DONE = "done"


@dataclass
class AgentTrajectory:

    state_text: str = ""
    action_text: str = ""
    state_dict: Dict[str, List[int | float]] = field(default_factory=dict)
    state_dicts: List[Dict[str, List[int | float]]] = field(default_factory=list)
    turn: int = 0
    metrics: Dict[str, List[float | int | bool]] = field(
        default_factory=lambda: defaultdict(list)
    )
    previous_action_text: str = ""
    previous_response_length: int = 0
    done: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "state_text": self.state_text,
            "action_text": self.action_text,
            "state_dict": self.state_dict,
            "state_dicts": self.state_dicts,
            "turn": self.turn,
            "metrics": dict(self.metrics),
            "previous_action_text": self.previous_action_text,
            "previous_response_length": self.previous_response_length,
            "done": self.done
        }


@dataclass
class Sample:
    Status = SampleStatus

    # for initialization
    sample: Dict[str, Any] = field(default_factory=dict)
    episode_id: int = field(default_factory=lambda: next(EPISODE_COUNTER))
    prompt_group_id: int = 0
    response_id: int = 0
    multi_agent: bool = False
    agent_ids: List[str] = field(default_factory=lambda: [DEFAULT_AGENT_ID])
    current_agent: str = DEFAULT_AGENT_ID
    shared_info: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[float | int | bool]] = field(
        default_factory=lambda: defaultdict(list)
    )
    agents: Dict[str, AgentTrajectory] = field(default_factory=dict)
    status: SampleStatus = SampleStatus.RUNNING

    # for logging
    def __post_init__(self):
        self._ensure_agent(self.current_agent)

    def to_json(self) -> Dict[str, Any]:
        return {
            "sample": self.sample,
            "episode_id": self.episode_id,
            "prompt_group_id": self.prompt_group_id,
            "response_id": self.response_id,
            "multi_agent": self.multi_agent,
            "agent_ids": self.agent_ids,
            "current_agent": self.current_agent,
            "shared_info": self.shared_info,
            "metrics": dict(self.metrics),
            "status": self.status.value,
            "agents": {
                agent_id: agent.to_json()
                for agent_id, agent in self.agents.items()
            }
        }

    def _ensure_agent(self, agent_id: str) -> AgentTrajectory:
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentTrajectory()
        return self.agents[agent_id]

    @property
    def agent_id(self) -> str:
        return self.current_agent

    @agent_id.setter
    def agent_id(self, value: str):
        self.current_agent = value
        self._ensure_agent(value)

    @property
    def agent_done(self) -> bool:
        return self._ensure_agent(self.current_agent).done

    @agent_done.setter
    def agent_done(self, value: bool):
        self._ensure_agent(self.current_agent).done = value

    @property
    def active_agent(self) -> AgentTrajectory:
        return self._ensure_agent(self.current_agent)

    @property
    def state_text(self) -> str:
        return self.active_agent.state_text

    @state_text.setter
    def state_text(self, value: str):
        self.active_agent.state_text = value

    @property
    def action_text(self) -> str:
        return self.active_agent.action_text

    @action_text.setter
    def action_text(self, value: str):
        self.active_agent.action_text = value

    @property
    def state_dict(self) -> Dict[str, List[int | float]]:
        return self.active_agent.state_dict

    @state_dict.setter
    def state_dict(self, value: Dict[str, List[int | float]]):
        self.active_agent.state_dict = value

    @property
    def state_dicts(self) -> List[Dict[str, List[int | float]]]:
        return self.active_agent.state_dicts

    @state_dicts.setter
    def state_dicts(self, value: List[Dict[str, List[int | float]]]):
        self.active_agent.state_dicts = value

    @property
    def turn(self) -> int:
        return self.active_agent.turn

    @turn.setter
    def turn(self, value: int):
        self.active_agent.turn = value

    @property
    def previous_action_text(self) -> str:
        return self.active_agent.previous_action_text

    @previous_action_text.setter
    def previous_action_text(self, value: str):
        self.active_agent.previous_action_text = value

    @property
    def previous_response_length(self) -> int:
        return self.active_agent.previous_response_length

    @previous_response_length.setter
    def previous_response_length(self, value: int):
        self.active_agent.previous_response_length = value

    def use_agent(self, agent_id: str):
        self.current_agent = agent_id
        self._ensure_agent(agent_id)


def initialize_state_dict(
    tokenizer: AutoTokenizer,
    state_text: str
) -> Dict[str, List[int | float]]:
        
    state = tokenizer.encode(state_text, add_special_tokens=False)
    return {
        "states": state,
        "actions": len(state) * [0],
        "action_mask": len(state) * [0],
        "logps": len(state) * [0.0],
        "rewards": len(state) * [0.0]
    }

def add_llm_response(sample: Sample, response: Dict[str, Any]):

    # `previous_action_text` is non-empty if aborted before
    sample.action_text = sample.previous_action_text + response["text"]

    # encode(decode(tokens)) may not be identical to tokens. Therefore, 
    # token-in-token-out is necessary to guanartee that tokens fed into 
    # training and inference engines are identical
    # https://github.com/OpenRLHF/OpenRLHF/pull/1094
    # https://github.com/THUDM/slime/pull/117
    meta_info = response["meta_info"]
    if "output_token_logprobs" in meta_info and len(meta_info["output_token_logprobs"][0]) == 3: # TODO: is this condition correct?
        logp, action, _ = map(list, zip(*meta_info["output_token_logprobs"]))
        sample.state_dict["states"].extend(action)
        sample.state_dict["actions"].extend(action)
        sample.state_dict["action_mask"].extend(len(action) * [1])
        sample.state_dict["logps"].extend(logp)
        sample.state_dict["rewards"].extend(len(action) * [0.0])
        # actual rewards will be overwritten in `add_env_response`

    finish_reason = meta_info["finish_reason"]["type"]
    if finish_reason == "abort":
        # User may mask action tokens to avoid off-policy training
        sample.status = SampleStatus.ABORTED
        sample.previous_action_text = sample.action_text
        sample.previous_response_length += meta_info["completion_tokens"]
        return
        
    sample.turn += 1
    response_length = (
        sample.previous_response_length + meta_info["completion_tokens"]
    )
    sample.active_agent.metrics["response_length"].append(response_length)
    sample.active_agent.metrics["length_clip_ratio"].append(
        finish_reason == "length"
    )
    sample.metrics["response_length"].append(response_length)
    sample.metrics["length_clip_ratio"].append(finish_reason == "length")
    sample.metrics[f"response_length/{sample.current_agent}"].append(
        response_length
    )
    sample.metrics[f"length_clip_ratio/{sample.current_agent}"].append(
        finish_reason == "length"
    )

    # reset if not aborted
    sample.previous_action_text = ""
    sample.previous_response_length = 0

def _append_if_trainable(agent: AgentTrajectory):
    if not agent.state_dict:
        return
    if sum(agent.state_dict.get("action_mask", [])) == 0 and len(agent.state_dicts) > 0:
        return
    agent.state_dicts.append(agent.state_dict)

def _update_agent_observation(
    tokenizer: AutoTokenizer,
    agent: AgentTrajectory,
    next_state: str,
    prefix: str
):
    if next_state.startswith(prefix):
        state_dict_delta = initialize_state_dict(
            tokenizer,
            next_state[len(prefix):]
        )
        for k, v in state_dict_delta.items():
            agent.state_dict[k].extend(v)
    else:
        _append_if_trainable(agent)
        agent.state_dict = initialize_state_dict(tokenizer, next_state)
    agent.state_text = next_state
    agent.action_text = ""

def _finalize_agent(sample: Sample, agent_id: str):
    agent = sample._ensure_agent(agent_id)
    if agent.done:
        return
    _append_if_trainable(agent)
    agent.done = True

def _get_agent_episode_reward(agent: AgentTrajectory) -> float:
    return float(sum([
        state_dict["rewards"][-1]
        for state_dict in agent.state_dicts
        if len(state_dict["rewards"]) > 0
    ]))

def _collect_episode_metrics(sample: Sample):
    if sample.metrics.get("_episode_metrics_collected"):
        return

    total_reward = float(sample.shared_info["global_reward"]) if "global_reward" in sample.shared_info else 0.0
    total_turns = 0
    total_score = 0.0
    has_score = False
    for agent_id in sample.agent_ids:
        agent = sample._ensure_agent(agent_id)
        reward = _get_agent_episode_reward(agent)
        turns = agent.turn
        score = float(sum(agent.metrics.get("scores", [])))

        if "global_reward" not in sample.shared_info:
            total_reward += reward
        total_turns += turns
        total_score += score
        has_score = has_score or ("scores" in agent.metrics)

        sample.metrics[f"rewards/{agent_id}"].append(reward)
        sample.metrics[f"turns/{agent_id}"].append(turns)
        sample.metrics[f"agent_done/{agent_id}"].append(agent.done)
        if "scores" in agent.metrics:
            sample.metrics[f"scores/{agent_id}"].append(score)

    sample.metrics["turns"].append(total_turns)
    sample.metrics["rewards"].append(total_reward)
    if has_score:
        sample.metrics["scores"].append(total_score)
    sample.metrics["_episode_metrics_collected"] = [True]

def _is_multi_agent_payload(response: Dict[str, Any]) -> bool:
    return any(
        key in response
        for key in (
            "agent_ids", "current_agent", "next_observations",
            "done_agents", "shared_info", "rewards", "scores"
        )
    ) and isinstance(response.get("next_observations", {}), dict)

def _normalize_multi_agent_response(
    sample: Sample,
    response: Dict[str, Any]
) -> Dict[str, Any]:
    rewards = response.get("rewards", {})
    if isinstance(rewards, (int, float)):
        rewards = {sample.current_agent: float(rewards)}
    rewards = {str(k): float(v) for k, v in rewards.items()}

    scores = response.get("scores", {})
    if isinstance(scores, (int, float)):
        scores = {sample.current_agent: float(scores)}
    scores = {str(k): float(v) for k, v in scores.items()}

    next_observations = response.get("next_observations", {})
    if response.get("next_state") is not None:
        next_observations = {
            **next_observations,
            sample.current_agent: response["next_state"]
        }

    done_agents = set(str(agent_id) for agent_id in response.get("done_agents", []))
    if response.get("done", False):
        done_agents.update(sample.agent_ids)
    for agent_id, next_state in next_observations.items():
        if next_state is None:
            done_agents.add(str(agent_id))

    return {
        "agent_ids": [str(agent_id) for agent_id in response.get("agent_ids", sample.agent_ids)],
        "current_agent": response.get("current_agent"),
        "next_observations": {
            str(agent_id): next_state
            for agent_id, next_state in next_observations.items()
            if next_state is not None
        },
        "rewards": rewards,
        "scores": scores,
        "done": response.get("done", False),
        "done_agents": list(done_agents),
        "shared_info": deepcopy(response.get("shared_info", sample.shared_info)),
        "extra_info": deepcopy(response.get("extra_info", sample.sample.get("extra_info", {})))
    }

def _get_next_turn_agent(sample: Sample, requested_agent: Optional[str], agent_order: str) -> Optional[str]:
    if requested_agent is not None and not sample._ensure_agent(requested_agent).done:
        return requested_agent
    available_agents = [
        agent_id for agent_id in sample.agent_ids
        if not sample._ensure_agent(agent_id).done
    ]
    if len(available_agents) == 0:
        return None
    if agent_order == "env_driven":
        return available_agents[0]
    if sample.current_agent in available_agents:
        current_idx = available_agents.index(sample.current_agent)
        return available_agents[(current_idx + 1) % len(available_agents)]
    return available_agents[0]

def _add_multi_agent_response(
    config: DictConfig,
    tokenizer: AutoTokenizer,
    sample: Sample,
    response: Dict[str, Any]
):
    response = _normalize_multi_agent_response(sample, response)
    sample.multi_agent = True
    sample.agent_ids = response["agent_ids"]
    sample.shared_info = response["shared_info"]
    sample.sample["extra_info"] = response["extra_info"]

    current_agent_id = sample.current_agent
    current_agent = sample._ensure_agent(current_agent_id)
    current_agent.state_dict["rewards"][-1] = response["rewards"].get(
        current_agent_id, 0.0
    )
    if current_agent_id in response["scores"]:
        current_agent.metrics["scores"].append(response["scores"][current_agent_id])

    for agent_id, score in response["scores"].items():
        if agent_id == current_agent_id:
            continue
        sample._ensure_agent(agent_id).metrics["scores"].append(score)

    previous_state = current_agent.state_text
    previous_action = current_agent.action_text
    for agent_id, next_state in response["next_observations"].items():
        agent = sample._ensure_agent(agent_id)
        prefix = previous_state + previous_action if agent_id == current_agent_id else agent.state_text
        if not agent.state_dict:
            agent.state_dict = initialize_state_dict(tokenizer, next_state)
        else:
            _update_agent_observation(tokenizer, agent, next_state, prefix)

    for agent_id in response["done_agents"]:
        _finalize_agent(sample, agent_id)

    next_agent = _get_next_turn_agent(
        sample,
        response["current_agent"],
        config.multi_agent.agent_order
    )
    if response["done"] or next_agent is None:
        sample.status = SampleStatus.DONE
        _collect_episode_metrics(sample)
        return
    sample.use_agent(next_agent)

def add_env_response(
    config: DictConfig,
    tokenizer: AutoTokenizer,
    sample: Sample,
    response: Dict[str, Any]
):
    if _is_multi_agent_payload(response):
        _add_multi_agent_response(config, tokenizer, sample, response)
        return

    sample.state_dict["rewards"][-1] = response["reward"]
    if "score" in response:
        sample.active_agent.metrics["scores"].append(float(response["score"]))

    if response["done"]:

        sample.status = SampleStatus.DONE
        sample.state_dicts.append(sample.state_dict)
        sample.active_agent.done = True
        _collect_episode_metrics(sample)
        return

    if response["next_state"].startswith(sample.state_text + sample.action_text):
        state_dict_delta = initialize_state_dict(
            tokenizer,
            response["next_state"][len(sample.state_text + sample.action_text):]
        )
        for k, v in state_dict_delta.items():
            sample.state_dict[k].extend(v)
    else:
        # If the previous state is not a prefix of the next state, the trajectory will 
        # contain multiple sequences
        sample.state_dicts.append(sample.state_dict)
        sample.state_dict = initialize_state_dict(
            tokenizer, response["next_state"]
        )
    sample.state_text = response["next_state"]

def _initialize_default_agent(
    config: DictConfig,
    tokenizer: AutoTokenizer,
    sample: Sample
):
    sample.multi_agent = False
    sample.agent_ids = [DEFAULT_AGENT_ID]
    sample.use_agent(DEFAULT_AGENT_ID)
    sample.shared_info = {}
    if config.apply_chat_template:
        sample.state_text = tokenizer.apply_chat_template(
            sample.sample[config.messages_key],
            add_generation_prompt=True,
            tokenize=False
        )
    else:
        sample.state_text = sample.sample[config.prompt_key]
    sample.state_dict = initialize_state_dict(tokenizer, sample.state_text)

def _normalize_initial_multi_agent_state(
    config: DictConfig,
    tokenizer: AutoTokenizer,
    sample: Sample,
    init_response: Dict[str, Any]
):
    if not _is_multi_agent_payload(init_response):
        raise ValueError(
            "Multi-agent environments must return `agent_ids` and "
            "`next_observations` from `reset` or dataset sample."
        )
    response = _normalize_multi_agent_response(sample, init_response)
    sample.multi_agent = True
    sample.agent_ids = response["agent_ids"]
    sample.shared_info = response["shared_info"]
    sample.sample["extra_info"] = response["extra_info"]
    for agent_id in sample.agent_ids:
        sample.use_agent(agent_id)
        next_state = response["next_observations"].get(agent_id)
        if next_state is None:
            sample.agent_done = True
            continue
        sample.state_text = next_state
        sample.state_dict = initialize_state_dict(tokenizer, next_state)
    next_agent = _get_next_turn_agent(
        sample,
        response["current_agent"],
        config.multi_agent.agent_order
    )
    if next_agent is None or response["done"]:
        sample.status = SampleStatus.DONE
        _collect_episode_metrics(sample)
        return
    sample.use_agent(next_agent)

async def _call_env_function(fn: Callable, **kwargs):
    signature = inspect.signature(fn)
    params = signature.parameters.values()
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in params
    )
    if accepts_kwargs:
        filtered_kwargs = kwargs
    else:
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k in signature.parameters
        }
    response = fn(**filtered_kwargs)
    if inspect.isawaitable(response):
        response = await response
    return response

def build_env_generate(
    step_fn: Callable,
    reset_fn: Optional[Callable] = None
) -> Callable:
    async def _env_step(sample: Sample) -> Dict[str, Any]:
        return await _call_env_function(
            step_fn,
            sample=sample,
            state=sample.state_text,
            action=sample.action_text,
            extra_info=deepcopy(sample.sample.get("extra_info", {})),
            agent_id=sample.current_agent,
            agent_states={
                agent_id: sample._ensure_agent(agent_id).state_text
                for agent_id in sample.agent_ids
            },
            shared_info=deepcopy(sample.shared_info),
            episode_id=sample.episode_id
        )

    async def _env_reset(
        config: DictConfig,
        tokenizer: AutoTokenizer,
        sample: Sample
    ) -> Optional[Dict[str, Any]]:
        if reset_fn is None:
            return None
        return await _call_env_function(
            reset_fn,
            sample=sample,
            raw_sample=sample.sample,
            config=config,
            tokenizer=tokenizer,
            extra_info=deepcopy(sample.sample.get("extra_info", {})),
            episode_id=sample.episode_id
        )

    async def _generate(
        config: DictConfig,
        tokenizer: AutoTokenizer,
        router_url: str,
        sample: Sample
    ):
        return await base_generate(
            config,
            tokenizer,
            router_url,
            sample,
            env_step_fn=_env_step,
            env_reset_fn=_env_reset
        )
    return _generate

async def base_generate(
    config: DictConfig,
    tokenizer: AutoTokenizer,
    router_url: str,
    sample: Sample,
    env_step_fn: Callable,
    env_reset_fn: Optional[Callable] = None
):
    """
    A typical generate function where user only needs to provide the `env_step` 
    function. User may provide their own `generate` function for advanced use.
    """
    sampling_params = OmegaConf.to_container(config.sampling_params)

    match sample.status:

        case SampleStatus.RUNNING:

            if getattr(config, "multi_agent", None) and config.multi_agent.enabled:
                init_response = None
                if env_reset_fn is not None:
                    init_response = await env_reset_fn(
                        config, tokenizer, sample
                    )
                elif _is_multi_agent_payload(sample.sample):
                    init_response = sample.sample
                if init_response is None:
                    raise ValueError(
                        "Multi-agent rollout is enabled but the environment does "
                        "not provide a `reset` function or dataset-level initial "
                        "multi-agent observations."
                    )
                _normalize_initial_multi_agent_state(
                    config, tokenizer, sample, init_response
                )
            else:
                _initialize_default_agent(config, tokenizer, sample)

        case SampleStatus.ABORTED:
            sample.status = SampleStatus.RUNNING

        case SampleStatus.DONE:
            # User may treat this case as `RUNNING` to avoid off-policy training
            return

    while True:
        if sample.status == SampleStatus.DONE:
            return

        # TODO: set `max_tokens`
        response = await async_request(
            router_url,
            "generate",
            json={
                "input_ids": sample.state_dict["states"],
                "sampling_params": {
                    **sampling_params,
                    "max_new_tokens": sampling_params["max_new_tokens"] - sample.previous_response_length,
                    "no_stop_trim": True
                },
                "return_logprob": True
            }
        )
        add_llm_response(sample, response)
        if sample.status == SampleStatus.ABORTED:
            return
        
        response = await env_step_fn(sample)
        add_env_response(config, tokenizer, sample, response)
        if sample.status == SampleStatus.DONE:
            return


class SampleGroup:

    def __init__(
        self,
        config: DictConfig,
        tokenizer: AutoTokenizer,
        sample: Dict[str, Any]
    ):

        self.config = config
        self.tokenizer = tokenizer
        self.prompt_group_id = next(PROMPT_GROUP_COUNTER)
        self.samples = [
            Sample(
                sample=deepcopy(sample),
                prompt_group_id=self.prompt_group_id,
                response_id=response_id
            )
            for response_id in range(config.responses_per_prompt)
        ]

    async def generate(self, router_url: str, generate_fn: Callable) -> "SampleGroup":
        """
        This function packs the generation tasks of samples within a group into a single task so that they will return togather.
        """
        await asyncio.gather(*(
            generate_fn(self.config, self.tokenizer, router_url, sample)
            for sample in self.samples
        ))
        return self
    
    def print(self):

        sample = self.samples[0]
        print("\n")
        if sample.multi_agent:
            for agent_id in sample.agent_ids:
                agent = sample._ensure_agent(agent_id)
                print(f"[{agent_id}] {agent.state_text}{agent.action_text}")
        else:
            print(sample.state_text + sample.action_text)
        print("[Reward]", sample.metrics["rewards"][0])

    def save(self, step):
        
        data = [sample.to_json() for sample in self.samples]
        os.makedirs(self.config.save_dir, exist_ok=True)
        with open(f"{self.config.save_dir}/step{step}.jsonl", "a") as f:
            f.write(json.dumps(data) + "\n")

    def to_all_tensor_dicts_and_metrics(self) -> Tuple[List[List[Dict[str, torch.Tensor]]], Dict[str, List[float | int | bool]]]:
        
        all_tensor_dicts, metrics = [], defaultdict(list)
        for sample in self.samples:
            team_reward = float(
                sample.shared_info.get(
                    "global_reward",
                    sum(_get_agent_episode_reward(agent) for agent in sample.agents.values())
                )
            ) if sample.multi_agent else None
            value_target = sample.shared_info.get("state_value_target")

            for agent_idx, agent_id in enumerate(sample.agent_ids):
                sample.use_agent(agent_id)
                tensor_dicts = []
                for state_dict in sample.state_dicts:
                    if sum(state_dict["action_mask"][1:]) == 0:
                        continue
                    tensor_dict = get_tensor_dict(
                        state_dict["states"],
                        state_dict["actions"],
                        state_dict["action_mask"],
                    )
                    seq_len = tensor_dict["states"].shape[0]
                    tensor_dict["llm_logps"] = torch.FloatTensor(
                        state_dict["logps"][1:]
                    )
                    tensor_dict["rewards"] = torch.FloatTensor(
                        state_dict["rewards"][1:]
                    )
                    tensor_dict["agent_id"] = torch.full(
                        (seq_len,), agent_idx, dtype=torch.long
                    )
                    tensor_dict["episode_id"] = torch.full(
                        (seq_len,), sample.episode_id, dtype=torch.long
                    )
                    tensor_dict["adv_group_id"] = torch.full(
                        (seq_len,),
                        sample.prompt_group_id * 1024 + agent_idx,
                        dtype=torch.long
                    )
                    tensor_dict["agent_done"] = torch.full(
                        (seq_len,),
                        int(sample.agent_done),
                        dtype=torch.long
                    )
                    tensor_dict["shared_mask"] = torch.zeros(
                        seq_len, dtype=torch.long
                    )
                    if sample.multi_agent:
                        action_positions = torch.where(
                            tensor_dict["action_mask"] > 0
                        )[0]
                        if len(action_positions) > 0:
                            last_action = action_positions[-1]
                            tensor_dict["shared_mask"][last_action] = 1
                        if team_reward is not None:
                            tensor_dict["team_rewards"] = torch.zeros(
                                seq_len, dtype=torch.float32
                            )
                            if len(action_positions) > 0:
                                tensor_dict["team_rewards"][last_action] = team_reward
                        if value_target is not None:
                            tensor_dict["state_value_targets"] = torch.zeros(
                                seq_len, dtype=torch.float32
                            )
                            if len(action_positions) > 0:
                                tensor_dict["state_value_targets"][last_action] = float(value_target)
                    tensor_dicts.append(tensor_dict)
                if len(tensor_dicts) > 0:
                    all_tensor_dicts.append(tensor_dicts)
            for k, v in sample.metrics.items():
                if k.startswith("_"):
                    continue
                metrics[k].extend(v)
        return all_tensor_dicts, metrics


class RLDataset(BaseDataset):

    def __getitem__(self, idx: int) -> SampleGroup:

        sample = self.dataset[idx]
        return SampleGroup(self.config, self.tokenizer, sample)

    def collate_fn(self, batch: Tuple[SampleGroup]) -> SampleGroup:
        return batch[0]