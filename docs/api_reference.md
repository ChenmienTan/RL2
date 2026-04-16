# API Reference

## Multi-Agent Environment API

### Environment Functions

#### `reset(sample, tokenizer, extra_info, **kwargs)`

Initialize the environment for a new episode.

**Parameters:**
- `sample` (SampleGroup): Input data sample containing the task
- `tokenizer` (AutoTokenizer): Tokenizer for formatting prompts
- `extra_info` (Dict): Additional metadata
- `**kwargs`: Additional arguments (e.g., `sample` object)

**Returns:** `Dict[str, Any]`
```python
{
    "agent_ids": List[str],           # ["agent_1", "agent_2", ...]
    "current_agent": str,              # "agent_1"
    "next_observations": {             # Initial prompts
        "agent_1": str,
        "agent_2": str,
    },
    "rewards": {                       # Initial rewards (usually 0)
        "agent_1": 0.0,
        "agent_2": 0.0,
    },
    "done": bool,                      # False initially
    "done_agents": [],                 # Empty initially
    "shared_info": Dict,               # Shared state
    "extra_info": Dict                 # Metadata
}
```

**Example:**
```python
async def reset(sample, tokenizer, extra_info, **kwargs):
    question = sample.sample["question"]
    
    return {
        "agent_ids": ["planner", "solver"],
        "current_agent": "planner",
        "next_observations": {
            "planner": f"Question: {question}\nPlan the solution:",
            "solver": f"Question: {question}\nWait for planner's notes."
        },
        "rewards": {"planner": 0.0, "solver": 0.0},
        "done": False,
        "done_agents": [],
        "shared_info": {},
        "extra_info": extra_info
    }
```

---

#### `step(state, action, extra_info, agent_id, agent_states, shared_info, **kwargs)`

Process one agent's action and transition to the next state.

**Parameters:**
- `state` (str): Current agent's observation/state
- `action` (str): Current agent's generated action/response
- `extra_info` (Dict): Metadata
- `agent_id` (str): ID of the agent that just acted
- `agent_states` (Dict[str, str]): All agents' current states
- `shared_info` (Dict): Shared information between agents
- `**kwargs`: Additional arguments (e.g., `sample` object)

**Returns:** `Dict[str, Any]`
```python
{
    "current_agent": str,              # Next agent to act (if not done)
    "next_observations": {             # Updated observations
        "next_agent": str,
    },
    "rewards": {                       # Rewards for this step
        "agent_1": float,
        "agent_2": float,
    },
    "scores": {                        # Optional: separate scores
        "agent_1": float,
        "agent_2": float,
    },
    "done": bool,                      # Episode finished?
    "done_agents": List[str],          # Newly finished agents
    "shared_info": Dict,               # Updated shared state
    "extra_info": Dict                 # Updated metadata
}
```

**Example:**
```python
async def step(state, action, extra_info, agent_id, agent_states, shared_info, **kwargs):
    if agent_id == "planner":
        # Planner finished, pass notes to solver
        solver_state = agent_states["solver"] + f"\nPlanner's notes: {action}\nNow solve:"
        return {
            "current_agent": "solver",
            "next_observations": {"solver": solver_state},
            "rewards": {"planner": 0.0, "solver": 0.0},
            "done": False,
            "done_agents": ["planner"],
            "shared_info": {"plan": action},
            "extra_info": extra_info
        }
    else:
        # Solver finished, calculate reward
        reward = evaluate_solution(action, kwargs["sample"])
        return {
            "rewards": {"planner": reward, "solver": reward},
            "scores": {"planner": reward, "solver": reward},
            "done": True,
            "done_agents": ["planner", "solver"],
            "shared_info": shared_info,
            "extra_info": extra_info
        }
```

---

### Configuration API

#### Multi-Agent Config

```yaml
rollout:
  env_path: str                        # Path to environment file
  
  multi_agent:
    enabled: bool                      # Enable multi-agent mode
    shared_policy: bool                # Share model between agents
    reward_mode: str                   # "team" | "individual" | "competitive"
    agent_order: str                   # "env_driven" | "round_robin" | "random"
    max_turns: int                     # Optional: max turns per episode
    
  train:
    path: str                          # Dataset path
    prompts_per_rollout: int           # Prompts per rollout
    responses_per_prompt: int          # Responses per prompt
    sampling_params:
      max_new_tokens: int
      temperature: float
      top_p: float
      stop: List[str]                  # Stop tokens
      
  test:
    path: str                          # Test dataset path
    prompts_per_rollout: int           # Optional: override for test
```

#### Actor Config

```yaml
actor:
  model_name: str                      # HuggingFace model name or path
  max_length_per_device: int           # Max sequence length
  use_liger_kernel: bool               # Use Liger kernel optimization
  tp_size: int                         # Tensor parallelism size
  
  kl:
    coef: float                        # KL divergence coefficient
    
  optimizer:
    lr: float                          # Learning rate
    weight_decay: float
    
  scheduler:
    type: str                          # "cosine" | "linear" | "constant"
```

#### Trainer Config

```yaml
trainer:
  total_steps: int                     # Total training steps
  test_freq: int                       # Test every N steps
  save_freq: int                       # Save checkpoint every N steps
  project: str                         # WandB project name
  experiment_name: str                 # Experiment name
  eval_only: bool                      # Evaluation only mode
```

---

## Utility Functions

### Communication Utils

#### `sync_request(url, endpoint, method="POST", max_trials=3, retry_delay=1, **kwargs)`

Make synchronous HTTP request with retry logic.

**Parameters:**
- `url` (str): Base URL
- `endpoint` (str): API endpoint
- `method` (str): HTTP method ("GET" or "POST")
- `max_trials` (int): Maximum retry attempts
- `retry_delay` (float): Delay between retries in seconds
- `**kwargs`: Additional arguments for requests (headers, json, etc.)

**Returns:** Response data (JSON or text)

**Example:**
```python
from RL2.utils.communication import sync_request

# Health check
response = sync_request("http://localhost:8000", "health", "GET")

# Register worker
response = sync_request(
    "http://localhost:8000",
    "workers",
    method="POST",
    headers={"Content-Type": "application/json"},
    json={"url": "http://localhost:8001"}
)
```

---

#### `async_request(url, endpoint, method="POST", **kwargs)`

Make asynchronous HTTP request.

**Parameters:**
- `url` (str): Base URL
- `endpoint` (str): API endpoint
- `method` (str): HTTP method
- `**kwargs`: Additional arguments for aiohttp

**Returns:** Awaitable response data

**Example:**
```python
from RL2.utils.communication import async_request

response = await async_request(
    "http://localhost:8000",
    "generate",
    method="POST",
    json={"prompt": "Hello", "max_tokens": 100}
)
```

---

### Logging Utils

#### `progress_bar(iterable, desc=None, total=None)`

Create a progress bar for iterations.

**Parameters:**
- `iterable`: Iterable to wrap
- `desc` (str): Description text
- `total` (int): Total iterations

**Returns:** tqdm progress bar

**Example:**
```python
from RL2.utils.logging import progress_bar

for batch in progress_bar(dataloader, desc="Training"):
    # Training code
    pass
```

---

#### `gather_and_log(metrics, step, prefix="train")`

Gather metrics from all ranks and log to WandB.

**Parameters:**
- `metrics` (Dict[str, float]): Metrics to log
- `step` (int): Current step
- `prefix` (str): Metric prefix

**Example:**
```python
from RL2.utils.logging import gather_and_log

metrics = {
    "loss": 0.5,
    "reward": 0.8,
    "entropy": 0.3
}
gather_and_log(metrics, step=100, prefix="train")
```

---

## Dataset API

### RLDataset

Base class for RL datasets.

```python
from RL2.datasets import RLDataset

class MyDataset(RLDataset):
    def __init__(self, config, tokenizer, dataset):
        super().__init__(config, tokenizer, dataset)
        
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # Process sample
        return processed_sample
```

---

### SampleGroup

Container for a group of samples.

**Attributes:**
- `sample` (Dict): Original sample data
- `meta_info` (Dict): Metadata (logprobs, etc.)
- `extra_info` (Dict): Additional information

**Example:**
```python
from RL2.datasets import SampleGroup

sample_group = SampleGroup(
    sample={"question": "What is 2+2?", "answer": "4"},
    meta_info={"logprobs": [...]},
    extra_info={"source": "math_dataset"}
)
```

---

## Training API

### PPOTrainer

Main trainer class for PPO algorithm.

```python
from RL2.trainer.ppo import PPOTrainer
from omegaconf import DictConfig

config = DictConfig({...})
trainer = PPOTrainer(config)
trainer.fit()
```

**Methods:**
- `fit()`: Run training loop
- `test()`: Run evaluation
- `save_checkpoint(step)`: Save model checkpoint

---

### Rollout

Handles environment rollouts and data collection.

```python
from RL2.workers import initialize_rollout

rollout = initialize_rollout(config.rollout)
data = await rollout(actor, samples)
```

**Methods:**
- `__call__(actor, samples)`: Perform rollout
- `_launch_server_process()`: Start inference server
- `_launch_router_process()`: Start router

---

## Actor API

### FSDPActor

FSDP-based actor for policy network.

```python
from RL2.workers.fsdp import FSDPActor

actor = FSDPActor(config, train=True)
actor.prepare_scheduler(total_steps)
```

**Methods:**
- `forward(input_ids, ...)`: Forward pass
- `generate(prompts, ...)`: Generate responses
- `save_checkpoint(path)`: Save model
- `load_checkpoint(path)`: Load model

---

## Server API

### ServerArgs

Configuration for SGLang inference server.

```python
from sglang.srt.server_args import ServerArgs

server_args = ServerArgs(
    model_path="Qwen/Qwen2.5-3B-Instruct",
    host="0.0.0.0",
    port=8000,
    tp_size=1,
    enable_memory_saver=False,
    log_level="error"
)
```

---

### RouterArgs

Configuration for SGLang router.

```python
from sglang_router.launch_router import RouterArgs

router_args = RouterArgs(
    host="0.0.0.0",
    port=8080,
    log_level="error",
    prometheus_port=9090
)
```

---

## Environment Examples

### Minimal Environment

```python
async def reset(sample, tokenizer, extra_info, **kwargs):
    return {
        "agent_ids": ["agent_1"],
        "current_agent": "agent_1",
        "next_observations": {"agent_1": "Start task"},
        "rewards": {"agent_1": 0.0},
        "done": False,
        "done_agents": [],
        "shared_info": {},
        "extra_info": extra_info
    }

async def step(state, action, extra_info, agent_id, agent_states, shared_info, **kwargs):
    reward = 1.0 if "correct" in action.lower() else 0.0
    return {
        "rewards": {"agent_1": reward},
        "scores": {"agent_1": reward},
        "done": True,
        "done_agents": ["agent_1"],
        "shared_info": shared_info,
        "extra_info": extra_info
    }
```

---

### Two-Agent Collaboration

```python
AGENT_A = "agent_a"
AGENT_B = "agent_b"

async def reset(sample, tokenizer, extra_info, **kwargs):
    task = sample.sample["task"]
    return {
        "agent_ids": [AGENT_A, AGENT_B],
        "current_agent": AGENT_A,
        "next_observations": {
            AGENT_A: f"Task: {task}\nYou go first:",
            AGENT_B: f"Task: {task}\nWait for agent A:"
        },
        "rewards": {AGENT_A: 0.0, AGENT_B: 0.0},
        "done": False,
        "done_agents": [],
        "shared_info": {"history": []},
        "extra_info": extra_info
    }

async def step(state, action, extra_info, agent_id, agent_states, shared_info, **kwargs):
    history = shared_info["history"] + [f"{agent_id}: {action}"]
    
    if agent_id == AGENT_A:
        # A finished, B's turn
        next_state = agent_states[AGENT_B] + f"\nAgent A said: {action}\nYour turn:"
        return {
            "current_agent": AGENT_B,
            "next_observations": {AGENT_B: next_state},
            "rewards": {AGENT_A: 0.0, AGENT_B: 0.0},
            "done": False,
            "done_agents": [AGENT_A],
            "shared_info": {"history": history},
            "extra_info": extra_info
        }
    else:
        # B finished, calculate reward
        reward = evaluate_collaboration(history)
        return {
            "rewards": {AGENT_A: reward, AGENT_B: reward},
            "scores": {AGENT_A: reward, AGENT_B: reward},
            "done": True,
            "done_agents": [AGENT_A, AGENT_B],
            "shared_info": {"history": history},
            "extra_info": extra_info
        }
```

---

### Three-Agent Sequential

```python
AGENTS = ["agent_1", "agent_2", "agent_3"]

async def reset(sample, tokenizer, extra_info, **kwargs):
    task = sample.sample["task"]
    return {
        "agent_ids": AGENTS,
        "current_agent": AGENTS[0],
        "next_observations": {
            agent: f"Task: {task}\nWait for your turn." for agent in AGENTS
        },
        "rewards": {agent: 0.0 for agent in AGENTS},
        "done": False,
        "done_agents": [],
        "shared_info": {"current_idx": 0, "results": []},
        "extra_info": extra_info
    }

async def step(state, action, extra_info, agent_id, agent_states, shared_info, **kwargs):
    current_idx = shared_info["current_idx"]
    results = shared_info["results"] + [action]
    
    if current_idx < len(AGENTS) - 1:
        # Next agent's turn
        next_idx = current_idx + 1
        next_agent = AGENTS[next_idx]
        next_state = f"Previous results: {results}\nYour turn:"
        
        return {
            "current_agent": next_agent,
            "next_observations": {next_agent: next_state},
            "rewards": {agent: 0.0 for agent in AGENTS},
            "done": False,
            "done_agents": [agent_id],
            "shared_info": {"current_idx": next_idx, "results": results},
            "extra_info": extra_info
        }
    else:
        # All agents finished
        reward = evaluate_results(results)
        return {
            "rewards": {agent: reward for agent in AGENTS},
            "scores": {agent: reward for agent in AGENTS},
            "done": True,
            "done_agents": AGENTS,
            "shared_info": {"current_idx": current_idx, "results": results},
            "extra_info": extra_info
        }
```

---

## Type Definitions

### Common Types

```python
from typing import Dict, List, Any

AgentID = str
Observation = str
Action = str
Reward = float

AgentStates = Dict[AgentID, Observation]
AgentRewards = Dict[AgentID, Reward]
SharedInfo = Dict[str, Any]
ExtraInfo = Dict[str, Any]
```

---

## Error Handling

### Common Exceptions

```python
# IndexError: Not enough data
try:
    data = rollout(actor, samples)
except IndexError:
    # Increase PROMPTS_PER_ROLLOUT
    pass

# ConnectionError: Server not ready
try:
    response = sync_request(url, endpoint)
except ConnectionError:
    # Wait for server or restart
    pass

# RuntimeError: Training failed
try:
    trainer.fit()
except RuntimeError as e:
    # Check logs and GPU status
    print(f"Training failed: {e}")
```

---

## Best Practices

1. **Always validate environment outputs**
   - Check all required keys are present
   - Ensure rewards are floats
   - Verify agent_ids match

2. **Use type hints**
   ```python
   async def reset(...) -> Dict[str, Any]:
       ...
   ```

3. **Handle edge cases**
   - Empty actions
   - Invalid agent IDs
   - Timeout scenarios

4. **Log important events**
   ```python
   print(f"[Agent {agent_id}] Action: {action[:50]}...")
   ```

5. **Test incrementally**
   - Start with 1 agent
   - Add agents one by one
   - Test with small data first

---

For more examples, see `docs/multi_agent_guide.md`.
For troubleshooting, see the Troubleshooting section in the guide.
