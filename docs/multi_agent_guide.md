# Multi-Agent Reinforcement Learning Guide

## Overview

RL2 supports multi-agent reinforcement learning where multiple agents collaborate or compete to solve tasks. This guide covers everything you need to know about using multi-agent training.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Creating Custom Environments](#creating-custom-environments)
4. [Configuration](#configuration)
5. [Training Modes](#training-modes)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Running the Countdown Example

The simplest way to get started is with the included countdown example:

```bash
cd /path/to/RL2

# 2-GPU training
export CUDA_VISIBLE_DEVICES=0,1
NPROC_PER_NODE=2 \
MODEL_NAME=Qwen/Qwen2.5-3B-Instruct \
PROMPTS_PER_ROLLOUT=16 \
TOTAL_STEPS=100 \
bash examples/multi_agent_countdown_reinforce.sh

# 4-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC_PER_NODE=4 \
PROMPTS_PER_ROLLOUT=16 \
bash examples/multi_agent_countdown_reinforce.sh

# 8-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8 \
PROMPTS_PER_ROLLOUT=32 \
bash examples/multi_agent_countdown_reinforce.sh
```

**Important:** `PROMPTS_PER_ROLLOUT` should be >= `NPROC_PER_NODE * 4` to ensure each GPU gets enough data.

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Process                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Trainer  │  │ Trainer  │  │ Trainer  │  │ Trainer  │   │
│  │  Rank 0  │  │  Rank 1  │  │  Rank 2  │  │  Rank 3  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │           │
│       └─────────────┴─────────────┴─────────────┘           │
│                         │                                    │
│                    ┌────▼────┐                              │
│                    │ Router  │  (Load Balancer)             │
│                    └────┬────┘                              │
│                         │                                    │
│       ┌─────────────────┼─────────────────┐                │
│       │                 │                 │                 │
│  ┌────▼────┐      ┌────▼────┐      ┌────▼────┐           │
│  │ Server  │      │ Server  │      │ Server  │           │
│  │  GPU 0  │      │  GPU 1  │      │  GPU 2  │  ...      │
│  └─────────┘      └─────────┘      └─────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### Multi-Agent Workflow

```
1. Environment Reset
   ├─> Initialize all agents
   ├─> Set initial observations for each agent
   └─> Define agent execution order

2. Agent Execution Loop
   ├─> Current agent generates action
   ├─> Environment processes action
   ├─> Update observations for next agent(s)
   ├─> Check if episode is done
   └─> Repeat until all agents finish

3. Reward Assignment
   ├─> Calculate rewards based on final state
   ├─> Distribute rewards (team/individual/competitive)
   └─> Log metrics

4. Policy Update
   ├─> Collect trajectories from all agents
   ├─> Compute advantages
   ├─> Update shared or individual policies
   └─> Save checkpoints
```

---

## Creating Custom Environments

### Environment Interface

Every multi-agent environment must implement two async functions:

```python
async def reset(sample, tokenizer, extra_info, **kwargs) -> Dict[str, Any]:
    """
    Initialize the environment for a new episode.
    
    Args:
        sample: Input data sample
        tokenizer: Tokenizer for formatting prompts
        extra_info: Additional metadata
        
    Returns:
        {
            "agent_ids": List[str],           # List of agent identifiers
            "current_agent": str,              # Which agent acts first
            "next_observations": {             # Initial prompts for each agent
                "agent_1": str,
                "agent_2": str,
                ...
            },
            "rewards": {                       # Initial rewards (usually 0)
                "agent_1": float,
                "agent_2": float,
                ...
            },
            "done": bool,                      # Episode finished?
            "done_agents": List[str],          # Agents that finished
            "shared_info": Dict,               # Shared state between agents
            "extra_info": Dict                 # Metadata
        }
    """
    pass

async def step(
    state: str,                    # Current agent's state
    action: str,                   # Current agent's action
    extra_info: Dict[str, Any],
    agent_id: str,                 # Current agent ID
    agent_states: Dict[str, str],  # All agents' states
    shared_info: Dict[str, Any],   # Shared information
    **kwargs
) -> Dict[str, Any]:
    """
    Process one agent's action and transition to next state.
    
    Returns:
        {
            "current_agent": str,              # Next agent to act (if not done)
            "next_observations": {             # Updated observations
                "next_agent": str,
                ...
            },
            "rewards": {                       # Rewards for this step
                "agent_1": float,
                "agent_2": float,
                ...
            },
            "scores": {                        # Optional: separate scores
                "agent_1": float,
                "agent_2": float,
                ...
            },
            "done": bool,                      # Episode finished?
            "done_agents": List[str],          # Newly finished agents
            "shared_info": Dict,               # Updated shared state
            "extra_info": Dict                 # Updated metadata
        }
    """
    pass
```

### Example: Dialogue Environment

```python
# envs/multi_agent_dialogue.py

AGENT_A = "agent_a"
AGENT_B = "agent_b"
MAX_TURNS = 5

async def reset(sample, tokenizer, extra_info, **kwargs):
    topic = sample.sample.get("topic", "general conversation")
    
    return {
        "agent_ids": [AGENT_A, AGENT_B],
        "current_agent": AGENT_A,
        "next_observations": {
            AGENT_A: tokenizer.apply_chat_template(
                [{"role": "user", "content": f"Start a conversation about: {topic}"}],
                add_generation_prompt=True,
                tokenize=False
            ),
            AGENT_B: tokenizer.apply_chat_template(
                [{"role": "user", "content": f"You are agent B. Wait for agent A to speak first about: {topic}"}],
                add_generation_prompt=True,
                tokenize=False
            )
        },
        "rewards": {AGENT_A: 0.0, AGENT_B: 0.0},
        "done": False,
        "done_agents": [],
        "shared_info": {"turn": 0, "dialogue_history": []},
        "extra_info": extra_info
    }

async def step(state, action, extra_info, agent_id, agent_states, shared_info, **kwargs):
    turn = shared_info["turn"] + 1
    dialogue_history = shared_info["dialogue_history"] + [f"{agent_id}: {action}"]
    
    # Check if conversation should end
    if turn >= MAX_TURNS:
        # Calculate reward based on dialogue quality
        reward = _evaluate_dialogue(dialogue_history)
        return {
            "rewards": {AGENT_A: reward, AGENT_B: reward},
            "scores": {AGENT_A: reward, AGENT_B: reward},
            "done": True,
            "done_agents": [AGENT_A, AGENT_B],
            "shared_info": {"turn": turn, "dialogue_history": dialogue_history},
            "extra_info": extra_info
        }
    
    # Continue conversation
    next_agent = AGENT_B if agent_id == AGENT_A else AGENT_A
    next_state = agent_states[next_agent] + f"\n{agent_id} said: {action}\nYour turn:"
    
    return {
        "current_agent": next_agent,
        "next_observations": {next_agent: next_state},
        "rewards": {AGENT_A: 0.0, AGENT_B: 0.0},
        "done": False,
        "done_agents": [agent_id],
        "shared_info": {"turn": turn, "dialogue_history": dialogue_history},
        "extra_info": extra_info
    }

def _evaluate_dialogue(history):
    # Implement your evaluation logic
    # For example: coherence, relevance, engagement
    return 0.5  # Placeholder
```

---

## Configuration

### Key Parameters

```yaml
rollout:
  env_path: envs/multi_agent_countdown.py  # Path to environment file
  
  multi_agent:
    enabled: true                           # Enable multi-agent mode
    shared_policy: true                     # All agents share same model
    reward_mode: team                       # team | individual | competitive
    agent_order: env_driven                 # env_driven | round_robin | random
    
  train:
    prompts_per_rollout: 32                 # Total prompts per rollout
    responses_per_prompt: 2                 # Responses per prompt
    sampling_params:
      max_new_tokens: 256
      temperature: 0.7
      top_p: 0.9
      stop: ['</answer>']                   # Stop tokens

actor:
  model_name: Qwen/Qwen2.5-3B-Instruct
  max_length_per_device: 8192
  
trainer:
  total_steps: 100
  test_freq: 10
  save_freq: 20
  project: MyProject
  experiment_name: my_experiment
```

### Reward Modes

**Team Reward** (`reward_mode: team`)
- All agents receive the same reward
- Encourages collaboration
- Best for cooperative tasks

```python
# All agents get the final reward
rewards = {
    "agent_1": final_reward,
    "agent_2": final_reward,
    ...
}
```

**Individual Reward** (`reward_mode: individual`)
- Each agent receives its own reward
- Agents optimize for personal gain
- Best for independent tasks

```python
# Each agent gets different reward
rewards = {
    "agent_1": agent_1_reward,
    "agent_2": agent_2_reward,
    ...
}
```

**Competitive Reward** (`reward_mode: competitive`)
- Zero-sum game: one agent's gain is another's loss
- Encourages competition
- Best for adversarial tasks

```python
# Winner takes all
rewards = {
    "agent_1": 1.0,
    "agent_2": -1.0,
}
```

### Agent Order

**Environment Driven** (`agent_order: env_driven`)
- Environment controls which agent acts next
- Most flexible, supports complex turn-taking
- Default and recommended

**Round Robin** (`agent_order: round_robin`)
- Agents take turns in fixed order
- Simple and predictable
- Good for symmetric tasks

**Random** (`agent_order: random`)
- Random agent selected each turn
- Adds stochasticity
- Experimental

---

## Training Modes

### Shared Policy (Recommended)

```yaml
multi_agent:
  shared_policy: true
```

**Pros:**
- Faster training (single model)
- Better sample efficiency
- Lower memory usage
- Agents learn from each other

**Cons:**
- Agents behave similarly
- Less diversity in strategies

**Use when:**
- Agents have similar roles
- You want faster training
- Memory is limited

### Individual Policies

```yaml
multi_agent:
  shared_policy: false
```

**Pros:**
- Agents can specialize
- More diverse behaviors
- Better for asymmetric roles

**Cons:**
- Slower training
- Higher memory usage
- Need more data

**Use when:**
- Agents have different roles
- You need diverse strategies
- You have enough resources

---

## Examples

### Example 1: Countdown (Included)

**Task:** Use numbers to reach a target value

**Agents:**
- Planner: Thinks about strategy
- Solver: Executes the solution

**Run:**
```bash
bash examples/multi_agent_countdown_reinforce.sh
```

### Example 2: Debate (See below)

**Task:** Two agents debate a topic

**Agents:**
- Pro: Argues for the proposition
- Con: Argues against the proposition

### Example 3: Code Review (See below)

**Task:** Collaborative code review

**Agents:**
- Reviewer: Finds issues
- Author: Fixes issues

---

## Troubleshooting

### Common Issues

**1. IndexError: list index out of range**

```
Error: IndexError: list index out of range in pack_tensor_dicts
```

**Cause:** Not enough prompts for the number of GPUs

**Solution:** Increase `PROMPTS_PER_ROLLOUT`:
```bash
# Rule of thumb: PROMPTS_PER_ROLLOUT >= NPROC_PER_NODE * 4
PROMPTS_PER_ROLLOUT=32  # For 8 GPUs
```

**2. Triton CPU Tensor Error**

```
Error: ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
```

**Cause:** `enable_memory_saver=True` causes CPU/GPU tensor issues

**Solution:** Already fixed in latest code (disabled by default)

**3. Server Connection Refused**

```
Error: Connection refused to http://x.x.x.x:xxxxx
```

**Cause:** Server not ready or crashed

**Solution:**
- Check if server processes are running: `ps aux | grep sglang`
- Check GPU memory: `nvidia-smi`
- Increase server startup timeout in code

**4. OOM Killed**

```
Error: Process killed by OOM killer
```

**Cause:** Not enough memory

**Solution:**
- Reduce `PROMPTS_PER_ROLLOUT`
- Reduce `MAX_NEW_TOKENS`
- Use fewer GPUs
- Reduce `max_length_per_device`

**5. Training Hangs**

**Cause:** Distributed communication deadlock

**Solution:**
- Check all processes are running: `ps aux | grep python`
- Check network connectivity between nodes
- Restart training with clean environment

### Debug Tips

**Enable verbose logging:**
```bash
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
```

**Monitor GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Check process status:**
```bash
ps aux | grep -E "(sglang|RL2)" | grep -v grep
```

**View logs:**
```bash
tail -f /tmp/training.log
```

---

## Performance Tips

### GPU Utilization

**For 2-4 GPUs:**
```bash
PROMPTS_PER_ROLLOUT=16
RESPONSES_PER_PROMPT=2
MAX_NEW_TOKENS=256
```

**For 8 GPUs:**
```bash
PROMPTS_PER_ROLLOUT=32
RESPONSES_PER_PROMPT=2
MAX_NEW_TOKENS=256
```

### Memory Optimization

**Reduce memory usage:**
```yaml
actor:
  max_length_per_device: 4096  # Instead of 8192
  
rollout:
  train:
    sampling_params:
      max_new_tokens: 128  # Instead of 256
```

### Training Speed

**Faster training:**
- Use shared policy
- Reduce test frequency
- Use smaller models for testing
- Enable gradient checkpointing (if available)

---

## Next Steps

1. Try the countdown example
2. Create your own environment
3. Experiment with different reward modes
4. Scale to more GPUs
5. Share your results!

For more examples, see `examples/` directory.
For API reference, see `docs/api_reference.md`.
For architecture details, see `docs/architecture.md`.
