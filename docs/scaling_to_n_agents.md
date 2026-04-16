# Scaling to N Agents - Complete Guide

## Overview

This guide shows how to easily scale from 2-3 agents to **any number of agents** (5, 10, 20+) using the new `MultiAgentBase` framework.

## 🚀 Quick Start

### Run with Different Agent Counts

```bash
# 3 agents (default)
NUM_AGENTS=3 bash examples/multi_agent_n_collaborative.sh

# 5 agents
NUM_AGENTS=5 bash examples/multi_agent_n_collaborative.sh

# 10 agents
NUM_AGENTS=10 bash examples/multi_agent_n_collaborative.sh

# 20 agents (requires more GPUs and prompts)
NUM_AGENTS=20 NPROC_PER_NODE=8 PROMPTS_PER_ROLLOUT=320 \
bash examples/multi_agent_n_collaborative.sh
```

### Run Different Team Types

```bash
# Software team (4 agents)
TEAM_TYPE=software bash examples/multi_agent_dynamic_team.sh

# Research team with 5 researchers (7 agents total)
TEAM_TYPE=research NUM_RESEARCHERS=5 bash examples/multi_agent_dynamic_team.sh

# Creative team (4 agents)
TEAM_TYPE=creative bash examples/multi_agent_dynamic_team.sh

# Custom team
TEAM_TYPE=custom CUSTOM_ROLES="leader,worker1,worker2,reviewer" \
bash examples/multi_agent_dynamic_team.sh
```

---

## 📚 Architecture

### New Components

```
envs/
├── multi_agent_base.py              # Base class for N-agent environments
├── multi_agent_n_collaborative.py   # N agents collaborate (configurable N)
└── multi_agent_dynamic_team.py      # Teams with custom roles
```

### How It Works

```python
# Old way: Hard-coded 2 agents
PLANNER_ID = "planner"
SOLVER_ID = "solver"

# New way: Dynamic N agents
class MyEnvironment(MultiAgentBase):
    def get_agent_roles(self):
        return [f"agent_{i}" for i in range(self.num_agents)]
```

---

## 🛠️ Creating N-Agent Environments

### Method 1: Use MultiAgentBase (Recommended)

```python
from envs.multi_agent_base import MultiAgentBase, set_environment

class MyNAgentEnv(MultiAgentBase):
    def __init__(self, num_agents=5):
        self.n = num_agents
        super().__init__()
    
    def get_agent_roles(self):
        # Generate N agent IDs
        return [f"agent_{i+1}" for i in range(self.n)]
    
    def get_initial_prompt(self, agent_id, sample):
        # Customize per agent
        return f"You are {agent_id}. Task: {sample['task']}"
    
    def process_action(self, agent_id, action, state):
        # Your logic here
        agent_idx = int(agent_id.split("_")[1])
        
        if agent_idx == self.n:
            # Last agent - calculate reward
            reward = self.evaluate(state)
            rewards = {f"agent_{i+1}": reward for i in range(self.n)}
            return None, rewards, True, state
        
        # Continue to next agent
        next_agent = f"agent_{agent_idx + 1}"
        rewards = {f"agent_{i+1}": 0.0 for i in range(self.n)}
        return next_agent, rewards, False, state

# Initialize with desired number
set_environment(lambda: MyNAgentEnv(num_agents=10))
```

### Method 2: Dynamic Roles

```python
class DynamicRoleEnv(MultiAgentBase):
    def __init__(self, roles):
        self.roles = roles  # ["manager", "worker1", "worker2", "reviewer"]
        super().__init__()
    
    def get_agent_roles(self):
        return self.roles
    
    def get_initial_prompt(self, agent_id, sample):
        role_prompts = {
            "manager": "You are the manager. Assign tasks.",
            "worker1": "You are worker 1. Execute tasks.",
            "worker2": "You are worker 2. Execute tasks.",
            "reviewer": "You are the reviewer. Check quality."
        }
        return role_prompts.get(agent_id, f"You are {agent_id}")
    
    def process_action(self, agent_id, action, state):
        # Custom logic based on roles
        ...

# Use with any roles
set_environment(lambda: DynamicRoleEnv(["role1", "role2", "role3", ...]))
```

---

## ⚙️ Configuration

### Key Parameters

#### PROMPTS_PER_ROLLOUT

**Critical:** Must scale with number of agents!

```bash
# Rule of thumb:
PROMPTS_PER_ROLLOUT >= NPROC_PER_NODE * NUM_AGENTS * 2

# Examples:
# 4 GPUs, 3 agents: 4 * 3 * 2 = 24
# 4 GPUs, 5 agents: 4 * 5 * 2 = 40
# 8 GPUs, 10 agents: 8 * 10 * 2 = 160
```

#### GPU Requirements

| Agents | GPUs | PROMPTS_PER_ROLLOUT | Memory |
|--------|------|---------------------|--------|
| 3      | 2-4  | 24-48               | ~60GB  |
| 5      | 4    | 40-80               | ~120GB |
| 10     | 4-8  | 80-160              | ~240GB |
| 20     | 8    | 320+                | ~480GB |

### Training Script Template

```bash
#!/usr/bin/env bash
set -euo pipefail

# Configuration
NUM_AGENTS="${NUM_AGENTS:-5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# Auto-scale prompts
PROMPTS_PER_ROLLOUT="${PROMPTS_PER_ROLLOUT:-$((NPROC_PER_NODE * NUM_AGENTS * 2))}"

# Export for Python
export NUM_AGENTS

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m RL2.trainer.ppo \
    rollout.env_path=envs/my_n_agent_env.py \
    rollout.train.prompts_per_rollout="${PROMPTS_PER_ROLLOUT}" \
    rollout.multi_agent.enabled=true \
    ...
```

---

## 📊 Examples

### Example 1: N-Agent Collaborative (3-20 agents)

**File:** `envs/multi_agent_n_collaborative.py`

**Features:**
- Configurable number of agents
- Sequential collaboration
- Team reward

**Usage:**
```bash
# 3 agents
NUM_AGENTS=3 bash examples/multi_agent_n_collaborative.sh

# 10 agents
NUM_AGENTS=10 NPROC_PER_NODE=8 PROMPTS_PER_ROLLOUT=160 \
bash examples/multi_agent_n_collaborative.sh
```

**Code:**
```python
# In your environment file
NUM_AGENTS = int(os.getenv("NUM_AGENTS", "3"))
set_environment(lambda: NAgentCollaborative(num_agents=NUM_AGENTS))
```

### Example 2: Dynamic Team (Custom Roles)

**File:** `envs/multi_agent_dynamic_team.py`

**Features:**
- Predefined team types (software, research, creative)
- Custom role definitions
- Flexible team sizes

**Usage:**
```bash
# Software team (4 agents)
TEAM_TYPE=software bash examples/multi_agent_dynamic_team.sh

# Research team with 5 researchers (7 agents)
TEAM_TYPE=research NUM_RESEARCHERS=5 bash examples/multi_agent_dynamic_team.sh

# Custom team
TEAM_TYPE=custom CUSTOM_ROLES="ceo,cto,engineer1,engineer2,qa" \
bash examples/multi_agent_dynamic_team.sh
```

**Code:**
```python
# Predefined teams
create_software_team()  # 4 agents
create_research_team(num_researchers=5)  # 7 agents
create_creative_team()  # 4 agents

# Custom team
create_custom_team(["role1", "role2", "role3"])
```

---

## 🎯 Best Practices

### 1. Start Small, Scale Up

```bash
# Step 1: Test with 3 agents
NUM_AGENTS=3 TOTAL_STEPS=10 bash examples/multi_agent_n_collaborative.sh

# Step 2: Scale to 5 agents
NUM_AGENTS=5 TOTAL_STEPS=10 bash examples/multi_agent_n_collaborative.sh

# Step 3: Scale to 10 agents
NUM_AGENTS=10 NPROC_PER_NODE=8 TOTAL_STEPS=100 \
bash examples/multi_agent_n_collaborative.sh
```

### 2. Monitor Memory Usage

```bash
# Watch GPU memory
watch -n 1 nvidia-smi

# If OOM, reduce:
# - PROMPTS_PER_ROLLOUT
# - MAX_NEW_TOKENS
# - max_length_per_device
```

### 3. Adjust Prompts Per Agent

```python
def get_initial_prompt(self, agent_id, sample):
    agent_idx = int(agent_id.split("_")[1])
    
    if agent_idx == 1:
        return "You are the first agent. Start the task."
    elif agent_idx == self.n:
        return "You are the last agent. Finalize the solution."
    else:
        return f"You are agent {agent_idx}. Build on previous work."
```

### 4. Implement Smart Reward Sharing

```python
def process_action(self, agent_id, action, state):
    if is_last_agent:
        # Evaluate team performance
        team_reward = self.evaluate(state)
        
        # Option 1: Equal sharing (team mode)
        rewards = {agent: team_reward for agent in self.agent_roles}
        
        # Option 2: Contribution-based
        contributions = self.measure_contributions(state)
        rewards = {
            agent: team_reward * contributions[agent]
            for agent in self.agent_roles
        }
        
        return None, rewards, True, state
```

---

## 🔧 Advanced Features

### 1. Parallel Agent Execution

```python
class ParallelAgents(MultiAgentBase):
    def get_agent_order(self):
        return "parallel"  # All agents act simultaneously
    
    async def step(self, ...):
        # Collect actions from all agents
        # Then process together
        ...
```

### 2. Hierarchical Agents

```python
class HierarchicalTeam(MultiAgentBase):
    def __init__(self):
        self.hierarchy = {
            "manager": ["worker1", "worker2"],
            "worker1": [],
            "worker2": []
        }
        super().__init__()
    
    def get_agent_roles(self):
        return ["manager", "worker1", "worker2"]
    
    def process_action(self, agent_id, action, state):
        # Manager delegates to workers
        if agent_id == "manager":
            # Activate workers
            return "worker1", rewards, False, state
        ...
```

### 3. Dynamic Agent Addition

```python
class DynamicAgents(MultiAgentBase):
    def process_action(self, agent_id, action, state):
        # Add new agent based on need
        if need_more_help:
            new_agent = f"helper_{state['helper_count']}"
            state['agent_ids'].append(new_agent)
            state['helper_count'] += 1
        ...
```

---

## 📈 Performance Optimization

### Memory Optimization

```bash
# For large teams (10+ agents)
# 1. Use shared policy
rollout.multi_agent.shared_policy=true

# 2. Reduce sequence length
actor.max_length_per_device=4096

# 3. Reduce max tokens
rollout.train.sampling_params.max_new_tokens=128

# 4. Use gradient checkpointing (if available)
actor.use_gradient_checkpointing=true
```

### Speed Optimization

```bash
# 1. Increase batch size
PROMPTS_PER_ROLLOUT=$((NPROC_PER_NODE * NUM_AGENTS * 4))

# 2. Use more GPUs
NPROC_PER_NODE=8

# 3. Reduce test frequency
TEST_FREQ=20
```

---

## 🐛 Troubleshooting

### Issue 1: IndexError with Many Agents

**Error:** `IndexError: list index out of range`

**Cause:** Not enough prompts for all agents

**Solution:**
```bash
# Increase PROMPTS_PER_ROLLOUT
PROMPTS_PER_ROLLOUT=$((NPROC_PER_NODE * NUM_AGENTS * 3))
```

### Issue 2: OOM with Many Agents

**Error:** Process killed by OOM

**Solution:**
```bash
# Option 1: Use more GPUs
NPROC_PER_NODE=8

# Option 2: Reduce prompts
PROMPTS_PER_ROLLOUT=$((NPROC_PER_NODE * NUM_AGENTS))

# Option 3: Reduce sequence length
MAX_LENGTH_PER_DEVICE=4096
```

### Issue 3: Slow Training

**Cause:** Too many agents, each taking long turns

**Solution:**
```bash
# Reduce max tokens per agent
MAX_NEW_TOKENS=128

# Use parallel execution (if applicable)
# Implement in your environment
```

---

## 📚 API Reference

### MultiAgentBase Class

```python
class MultiAgentBase(ABC):
    @abstractmethod
    def get_agent_roles(self) -> List[str]:
        """Return list of agent IDs"""
        
    @abstractmethod
    def get_initial_prompt(self, agent_id: str, sample: Dict) -> str:
        """Generate initial prompt for agent"""
        
    @abstractmethod
    def process_action(
        self, agent_id: str, action: str, state: Dict
    ) -> Tuple[Optional[str], Dict[str, float], bool, Dict]:
        """
        Process agent action.
        
        Returns:
            (next_agent, rewards, done, new_state)
        """
        
    def build_next_prompt(
        self, next_agent: str, prev_agent: str, 
        prev_action: str, state: Dict, agent_states: Dict
    ) -> str:
        """Build prompt for next agent (optional override)"""
```

### Helper Functions

```python
# Set environment
from envs.multi_agent_base import set_environment
set_environment(lambda: MyEnvironment(num_agents=10))

# Get environment
from envs.multi_agent_base import get_environment
env = get_environment()
```

---

## 🎓 Complete Example

```python
# my_10_agent_env.py
from envs.multi_agent_base import MultiAgentBase, set_environment

class My10AgentEnv(MultiAgentBase):
    def __init__(self):
        self.n = 10
        super().__init__()
    
    def get_agent_roles(self):
        return [f"agent_{i+1}" for i in range(self.n)]
    
    def get_initial_prompt(self, agent_id, sample):
        idx = int(agent_id.split("_")[1])
        return f"Agent {idx}/{self.n}. Task: {sample['task']}"
    
    def process_action(self, agent_id, action, state):
        idx = int(agent_id.split("_")[1])
        
        # Store contribution
        state.setdefault("contributions", {})[agent_id] = action
        
        if idx == self.n:
            # Last agent - evaluate
            reward = len(state["contributions"]) / self.n
            rewards = {f"agent_{i+1}": reward for i in range(self.n)}
            return None, rewards, True, state
        
        # Next agent
        next_agent = f"agent_{idx + 1}"
        rewards = {f"agent_{i+1}": 0.0 for i in range(self.n)}
        return next_agent, rewards, False, state

set_environment(lambda: My10AgentEnv())
```

**Run:**
```bash
NPROC_PER_NODE=8 PROMPTS_PER_ROLLOUT=160 \
bash examples/multi_agent_n_collaborative.sh
```

---

## 🚀 Summary

**Before:** Hard-coded 2-3 agents, difficult to scale

**After:** 
- ✅ Easy to scale to 5, 10, 20+ agents
- ✅ `MultiAgentBase` class handles complexity
- ✅ Simple API: just implement 3 methods
- ✅ Automatic prompt building and state management
- ✅ Flexible team configurations

**Next Steps:**
1. Try `multi_agent_n_collaborative.py` with different N
2. Create custom teams with `multi_agent_dynamic_team.py`
3. Build your own N-agent environment using `MultiAgentBase`

---

**Last Updated:** 2026-04-16
**Status:** Production Ready ✅
