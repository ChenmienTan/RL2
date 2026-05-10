# Multi-Agent Reinforcement Learning Documentation

Complete documentation for RL2's multi-agent training capabilities.

## 📚 Documentation Index

### Getting Started
- **[Multi-Agent Guide](multi_agent_guide.md)** - Complete guide to multi-agent training
  - Quick start examples
  - Architecture overview
  - Creating custom environments
  - Configuration reference
  - Troubleshooting

### Scaling to N Agents ⭐ NEW
- **[Scaling to N Agents](scaling_to_n_agents.md)** - Scale from 2 to 20+ agents easily
  - MultiAgentBase framework
  - Dynamic team configurations
  - N-agent examples (3, 5, 10, 20+ agents)
  - Performance optimization
  - Complete code examples

### API Reference
- **[API Reference](api_reference.md)** - Detailed API documentation
  - Environment API
  - Configuration API
  - Utility functions
  - Training API
  - Code examples

## 🚀 Quick Start

### 1. Run the Countdown Example (2 agents)

```bash
cd /path/to/RL2

# 4-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC_PER_NODE=4 \
PROMPTS_PER_ROLLOUT=16 \
bash examples/multi_agent_countdown_reinforce.sh
```

### 2. Scale to N Agents ⭐ NEW

```bash
# 5 agents
NUM_AGENTS=5 bash examples/multi_agent_n_collaborative.sh

# 10 agents
NUM_AGENTS=10 NPROC_PER_NODE=8 PROMPTS_PER_ROLLOUT=160 \
bash examples/multi_agent_n_collaborative.sh

# Custom team
TEAM_TYPE=software bash examples/multi_agent_dynamic_team.sh
```

### 3. Try Other Examples

**Debate (2 agents):**
```bash
bash examples/multi_agent_debate.sh
```

**Code Review (2 agents):**
```bash
bash examples/multi_agent_code_review.sh
```

**Story Writing (3 agents):**
```bash
bash examples/multi_agent_story.sh
```

## 📖 Available Environments

| Environment | Agents | Scalable | Description | File |
|-------------|--------|----------|-------------|------|
| **Countdown** | 2 | No | Planner + Solver for math problems | `envs/multi_agent_countdown.py` |
| **Debate** | 2 | No | Pro vs Con debate on topics | `envs/multi_agent_debate.py` |
| **Code Review** | 2 | No | Reviewer + Author for code improvement | `envs/multi_agent_code_review.py` |
| **Story Writing** | 3 | No | Planner + Writer + Editor for stories | `envs/multi_agent_story.py` |
| **N-Collaborative** ⭐ | **3-20+** | **Yes** | N agents collaborate on problems | `envs/multi_agent_n_collaborative.py` |
| **Dynamic Team** ⭐ | **4-20+** | **Yes** | Configurable teams with custom roles | `envs/multi_agent_dynamic_team.py` |

## 🏗️ Architecture

```
Training Process
├── Trainer (PPO algorithm)
├── Router (Load balancer)
├── Servers (Inference engines)
└── Environment (Multi-agent logic)

Multi-Agent Workflow
1. Environment.reset() → Initialize agents
2. Agent 1 acts → Environment.step()
3. Agent 2 acts → Environment.step()
4. ... (continue until done)
5. Calculate rewards → Update policy
```

## 🎯 Key Features

### Supported Configurations

- **Agent Count**: 2-3 agents (easily extensible to more)
- **GPU Scale**: 2/4/8 GPUs tested and working
- **Policy Sharing**: Shared or individual policies
- **Reward Modes**: Team, individual, competitive
- **Agent Order**: Environment-driven, round-robin, random

### Training Modes

**Shared Policy (Recommended)**
- All agents use the same model
- Faster training, better sample efficiency
- Lower memory usage

**Individual Policies**
- Each agent has its own model
- More diverse behaviors
- Higher memory usage

## 📝 Creating Custom Environments

### Minimal Template

```python
async def reset(sample, tokenizer, extra_info, **kwargs):
    return {
        "agent_ids": ["agent_1", "agent_2"],
        "current_agent": "agent_1",
        "next_observations": {
            "agent_1": "Your prompt here",
            "agent_2": "Your prompt here"
        },
        "rewards": {"agent_1": 0.0, "agent_2": 0.0},
        "done": False,
        "done_agents": [],
        "shared_info": {},
        "extra_info": extra_info
    }

async def step(state, action, extra_info, agent_id, agent_states, shared_info, **kwargs):
    # Your logic here
    return {
        "rewards": {"agent_1": reward, "agent_2": reward},
        "done": True,
        "done_agents": ["agent_1", "agent_2"],
        "shared_info": shared_info,
        "extra_info": extra_info
    }
```

See [Multi-Agent Guide](multi_agent_guide.md#creating-custom-environments) for detailed examples.

## ⚙️ Configuration

### Basic Configuration

```bash
# Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC_PER_NODE=4

# Model
MODEL_NAME=Qwen/Qwen2.5-3B-Instruct

# Training
PROMPTS_PER_ROLLOUT=16  # Must be >= NPROC_PER_NODE * 4
RESPONSES_PER_PROMPT=2
MAX_NEW_TOKENS=256
TOTAL_STEPS=100

# Multi-Agent
REWARD_MODE=team  # team | individual | competitive
```

### Advanced Configuration

See [API Reference](api_reference.md#configuration-api) for complete configuration options.

## 🐛 Troubleshooting

### Common Issues

**IndexError: list index out of range**
```bash
# Solution: Increase PROMPTS_PER_ROLLOUT
PROMPTS_PER_ROLLOUT=32  # For 8 GPUs
```

**Triton CPU Tensor Error**
- Already fixed in latest code (disabled `enable_memory_saver`)

**Server Connection Refused**
- Check server processes: `ps aux | grep sglang`
- Check GPU memory: `nvidia-smi`

**OOM Killed**
- Reduce `PROMPTS_PER_ROLLOUT`
- Reduce `MAX_NEW_TOKENS`
- Use fewer GPUs

See [Troubleshooting Guide](multi_agent_guide.md#troubleshooting) for more details.

## 📊 Performance Tips

### GPU Utilization

| GPUs | PROMPTS_PER_ROLLOUT | RESPONSES_PER_PROMPT |
|------|---------------------|----------------------|
| 2    | 8-16                | 2                    |
| 4    | 16-32               | 2                    |
| 8    | 32-64               | 2                    |

### Memory Optimization

- Use `shared_policy=true` for lower memory usage
- Reduce `max_length_per_device` if OOM
- Reduce `MAX_NEW_TOKENS` for faster training

## 🔬 Research & Development

### Extending to More Agents

Current examples support 2-3 agents. To add more:

1. Define agent IDs in your environment
2. Initialize observations for all agents in `reset()`
3. Implement turn-taking logic in `step()`
4. Ensure `PROMPTS_PER_ROLLOUT >= NPROC_PER_NODE * num_agents`

### Implementing New Reward Modes

Current modes: `team`, `individual`, `competitive`

To add custom reward logic:
1. Modify reward calculation in `step()`
2. Return different rewards for each agent
3. Test with small dataset first

### Advanced Features

- **Dynamic agent count**: Vary number of agents per episode
- **Hierarchical agents**: Agents with sub-agents
- **Communication protocols**: Structured message passing
- **Partial observability**: Agents see different information

## 📚 Additional Resources

### Papers & References

- [Multi-Agent Reinforcement Learning: A Survey](https://arxiv.org/abs/1911.10635)
- [Emergent Communication in Multi-Agent RL](https://arxiv.org/abs/1605.06676)
- [Cooperative Multi-Agent Learning](https://arxiv.org/abs/1812.09755)

### Related Projects

- [OpenAI Multi-Agent](https://github.com/openai/multiagent-particle-envs)
- [PettingZoo](https://github.com/Farama-Foundation/PettingZoo)
- [SMAC](https://github.com/oxwhirl/smac)

## 🤝 Contributing

To contribute new environments or improvements:

1. Create your environment in `envs/multi_agent_*.py`
2. Add training script in `examples/multi_agent_*.sh`
3. Test with 2/4 GPUs
4. Update documentation
5. Submit PR

## 📞 Support

- **Issues**: Report bugs or request features
- **Discussions**: Ask questions or share ideas
- **Documentation**: Improve or clarify docs

## 📄 License

Same as RL2 main project.

---

**Last Updated**: 2026-04-16

**Version**: 1.0.0

**Status**: Production Ready ✅
