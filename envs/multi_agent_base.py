"""
Generic Multi-Agent Environment Base Class

This base class makes it easy to create environments with any number of agents.
Users only need to define agent roles and implement simple logic.

Example usage:
    class MyEnvironment(MultiAgentBase):
        def get_agent_roles(self):
            return ["agent_1", "agent_2", "agent_3", ...]

        def get_initial_prompt(self, agent_id, sample):
            return f"You are {agent_id}. Task: {sample['task']}"

        def process_action(self, agent_id, action, state):
            # Your logic here
            return next_agent, reward, done
"""

from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod


class MultiAgentBase(ABC):
    """
    Base class for multi-agent environments.

    Subclasses only need to implement:
    - get_agent_roles(): Return list of agent IDs
    - get_initial_prompt(): Return initial prompt for each agent
    - process_action(): Process agent action and return next state
    """

    def __init__(self):
        self.agent_roles = self.get_agent_roles()
        self.num_agents = len(self.agent_roles)

    @abstractmethod
    def get_agent_roles(self) -> List[str]:
        """
        Define agent roles/IDs.

        Returns:
            List of agent IDs, e.g., ["planner", "solver", "reviewer"]
        """
        pass

    @abstractmethod
    def get_initial_prompt(self, agent_id: str, sample: Dict[str, Any]) -> str:
        """
        Generate initial prompt for an agent.

        Args:
            agent_id: Agent identifier
            sample: Input data sample

        Returns:
            Initial prompt string for the agent
        """
        pass

    @abstractmethod
    def process_action(
        self,
        agent_id: str,
        action: str,
        state: Dict[str, Any]
    ) -> Tuple[Optional[str], Dict[str, float], bool, Dict[str, Any]]:
        """
        Process an agent's action.

        Args:
            agent_id: Current agent ID
            action: Agent's generated action
            state: Current environment state (shared_info)

        Returns:
            Tuple of:
            - next_agent: ID of next agent to act (None if episode done)
            - rewards: Dict of rewards for each agent
            - done: Whether episode is finished
            - new_state: Updated environment state
        """
        pass

    def get_reward_mode(self) -> str:
        """
        Override to specify reward mode.

        Returns:
            "team" (default), "individual", or "competitive"
        """
        return "team"

    def get_agent_order(self) -> str:
        """
        Override to specify agent order.

        Returns:
            "sequential" (default), "round_robin", or "custom"
        """
        return "sequential"

    def get_next_agent_sequential(
        self,
        current_agent: str,
        done_agents: List[str]
    ) -> Optional[str]:
        """
        Get next agent in sequential order.

        Args:
            current_agent: Current agent ID
            done_agents: List of agents that have finished

        Returns:
            Next agent ID or None if all done
        """
        current_idx = self.agent_roles.index(current_agent)

        # Find next agent that hasn't finished
        for i in range(current_idx + 1, self.num_agents):
            if self.agent_roles[i] not in done_agents:
                return self.agent_roles[i]

        return None  # All agents done

    async def reset(self, sample, tokenizer, extra_info, **kwargs) -> Dict[str, Any]:
        """
        Initialize environment for a new episode.

        This method is called by RL2 framework.
        """
        # Get initial prompts for all agents
        next_observations = {}
        for agent_id in self.agent_roles:
            prompt = self.get_initial_prompt(agent_id, sample.sample)
            next_observations[agent_id] = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )

        # Initialize rewards
        rewards = {agent_id: 0.0 for agent_id in self.agent_roles}

        return {
            "agent_ids": self.agent_roles,
            "current_agent": self.agent_roles[0],  # Start with first agent
            "next_observations": next_observations,
            "rewards": rewards,
            "done": False,
            "done_agents": [],
            "shared_info": {
                "sample": sample.sample,
                "history": [],
                "turn": 0
            },
            "extra_info": extra_info
        }

    async def step(
        self,
        state: str,
        action: str,
        extra_info: Dict[str, Any],
        agent_id: str,
        agent_states: Dict[str, str],
        shared_info: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process one agent's action.

        This method is called by RL2 framework.
        """
        # Update history
        history = shared_info.get("history", [])
        history.append(f"{agent_id}: {action}")

        turn = shared_info.get("turn", 0) + 1

        # Update state
        new_state = {
            **shared_info,
            "history": history,
            "turn": turn,
            "last_agent": agent_id,
            "last_action": action
        }

        # Process action (user-defined logic)
        next_agent, rewards, done, new_state = self.process_action(
            agent_id, action, new_state
        )

        # Update done_agents
        done_agents = shared_info.get("done_agents", []) + [agent_id]

        if done:
            # Episode finished
            return {
                "rewards": rewards,
                "scores": rewards,  # Use same values for scores
                "done": True,
                "done_agents": self.agent_roles,  # All agents done
                "shared_info": new_state,
                "extra_info": extra_info
            }

        # Determine next agent
        if next_agent is None:
            # Use default sequential order
            next_agent = self.get_next_agent_sequential(agent_id, done_agents)

        if next_agent is None:
            # All agents finished but episode not marked done
            # Calculate final rewards
            final_rewards = rewards if rewards else {
                agent_id: 0.0 for agent_id in self.agent_roles
            }
            return {
                "rewards": final_rewards,
                "scores": final_rewards,
                "done": True,
                "done_agents": self.agent_roles,
                "shared_info": new_state,
                "extra_info": extra_info
            }

        # Continue to next agent
        # Build next observation
        next_prompt = self.build_next_prompt(
            next_agent, agent_id, action, new_state, agent_states
        )

        # Use stored tokenizer if available
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            next_observation = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": next_prompt}],
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            next_observation = next_prompt

        return {
            "current_agent": next_agent,
            "next_observations": {next_agent: next_observation},
            "rewards": {agent_id: 0.0 for agent_id in self.agent_roles},
            "done": False,
            "done_agents": done_agents,
            "shared_info": new_state,
            "extra_info": extra_info
        }

    def build_next_prompt(
        self,
        next_agent: str,
        prev_agent: str,
        prev_action: str,
        state: Dict[str, Any],
        agent_states: Dict[str, str]
    ) -> str:
        """
        Build prompt for next agent.

        Override this method to customize how agents receive information.

        Default: Append previous agent's action to next agent's state.
        """
        current_state = agent_states.get(next_agent, "")
        return f"{current_state}\n\n{prev_agent} said: {prev_action}\n\nYour turn:"

    def set_tokenizer(self, tokenizer):
        """Store tokenizer reference for use in step()"""
        self.tokenizer = tokenizer


# Global instance for RL2 to use
_env_instance = None

def set_environment(env_class):
    """Set the environment class to use"""
    global _env_instance
    _env_instance = env_class()

def get_environment():
    """Get the current environment instance"""
    return _env_instance


# Export reset and step functions for RL2
async def reset(sample, tokenizer, extra_info, **kwargs):
    """Wrapper for RL2 framework"""
    env = get_environment()
    if env is None:
        raise RuntimeError("Environment not set. Call set_environment() first.")
    env.set_tokenizer(tokenizer)
    return await env.reset(sample, tokenizer, extra_info, **kwargs)

async def step(state, action, extra_info, agent_id, agent_states, shared_info, **kwargs):
    """Wrapper for RL2 framework"""
    env = get_environment()
    if env is None:
        raise RuntimeError("Environment not set. Call set_environment() first.")
    return await env.step(state, action, extra_info, agent_id, agent_states, shared_info, **kwargs)
