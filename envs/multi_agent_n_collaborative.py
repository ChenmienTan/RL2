"""
N-Agent Collaborative Problem Solving

This example shows how to easily create an environment with N agents
using the MultiAgentBase class.

Each agent contributes to solving a problem sequentially.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from multi_agent_base import MultiAgentBase, set_environment
from typing import Dict, Any, List, Tuple, Optional


class NAgentCollaborative(MultiAgentBase):
    """
    N agents collaborate to solve a problem.

    Each agent:
    1. Receives the problem and previous agents' contributions
    2. Adds their own contribution
    3. Passes to next agent

    Final agent synthesizes all contributions into a solution.
    """

    def __init__(self, num_agents: int = 3):
        """
        Initialize with specified number of agents.

        Args:
            num_agents: Number of agents (default: 3)
        """
        self.n = num_agents
        super().__init__()

    def get_agent_roles(self) -> List[str]:
        """
        Generate agent IDs dynamically.

        Returns:
            ["agent_1", "agent_2", ..., "agent_N"]
        """
        return [f"agent_{i+1}" for i in range(self.n)]

    def get_initial_prompt(self, agent_id: str, sample: Dict[str, Any]) -> str:
        """
        Generate initial prompt for each agent.

        First agent gets the problem.
        Other agents wait for their turn.
        """
        problem = self._get_problem(sample)
        agent_idx = int(agent_id.split("_")[1])

        if agent_idx == 1:
            # First agent
            return (
                f"You are Agent 1 in a team of {self.n} agents.\n"
                f"Problem: {problem}\n\n"
                f"Your role: Analyze the problem and provide initial insights.\n"
                f"Think step by step and share your analysis."
            )
        elif agent_idx == self.n:
            # Last agent
            return (
                f"You are Agent {self.n} (final agent) in a team of {self.n} agents.\n"
                f"Problem: {problem}\n\n"
                f"Your role: Synthesize all previous contributions and provide the final solution.\n"
                f"Wait for other agents' input."
            )
        else:
            # Middle agents
            return (
                f"You are Agent {agent_idx} in a team of {self.n} agents.\n"
                f"Problem: {problem}\n\n"
                f"Your role: Build on previous agents' work and add your insights.\n"
                f"Wait for Agent {agent_idx-1}'s contribution."
            )

    def process_action(
        self,
        agent_id: str,
        action: str,
        state: Dict[str, Any]
    ) -> Tuple[Optional[str], Dict[str, float], bool, Dict[str, Any]]:
        """
        Process agent's action.

        Returns:
            (next_agent, rewards, done, new_state)
        """
        agent_idx = int(agent_id.split("_")[1])

        # Store this agent's contribution
        contributions = state.get("contributions", {})
        contributions[agent_id] = action
        state["contributions"] = contributions

        # Check if this is the last agent
        if agent_idx == self.n:
            # Last agent finished - evaluate solution
            reward = self._evaluate_solution(state)

            # All agents get same reward (team mode)
            rewards = {f"agent_{i+1}": reward for i in range(self.n)}

            return None, rewards, True, state

        # Continue to next agent
        next_agent = f"agent_{agent_idx + 1}"
        rewards = {f"agent_{i+1}": 0.0 for i in range(self.n)}

        return next_agent, rewards, False, state

    def build_next_prompt(
        self,
        next_agent: str,
        prev_agent: str,
        prev_action: str,
        state: Dict[str, Any],
        agent_states: Dict[str, str]
    ) -> str:
        """
        Build prompt for next agent with all previous contributions.
        """
        contributions = state.get("contributions", {})
        next_idx = int(next_agent.split("_")[1])

        # Build summary of previous contributions
        summary = []
        for i in range(1, next_idx):
            agent = f"agent_{i}"
            if agent in contributions:
                summary.append(f"Agent {i}: {contributions[agent]}")

        summary_text = "\n\n".join(summary)

        if next_idx == self.n:
            # Last agent - synthesize
            return (
                f"Previous agents' contributions:\n\n{summary_text}\n\n"
                f"Now provide the final solution based on all contributions above. "
                f"Synthesize the insights and give a clear answer."
            )
        else:
            # Middle agent - build on previous work
            return (
                f"Previous agents' contributions:\n\n{summary_text}\n\n"
                f"Now add your insights. Build on what previous agents said and "
                f"contribute your unique perspective."
            )

    def _get_problem(self, sample: Dict[str, Any]) -> str:
        """Extract problem from sample."""
        if "problem" in sample:
            return sample["problem"]
        if "question" in sample:
            return sample["question"]
        if "prompt" in sample:
            return sample["prompt"]
        return str(sample)

    def _evaluate_solution(self, state: Dict[str, Any]) -> float:
        """
        Evaluate the quality of the collaborative solution.

        Simple heuristic:
        - Check if all agents contributed
        - Check solution length
        - Check for key reasoning words
        """
        contributions = state.get("contributions", {})

        # All agents contributed?
        if len(contributions) != self.n:
            return 0.1

        # Get final solution
        final_solution = contributions.get(f"agent_{self.n}", "")

        # Check length (reasonable solution)
        length_score = min(1.0, len(final_solution.split()) / 50)

        # Check for reasoning keywords
        reasoning_keywords = [
            "because", "therefore", "thus", "hence", "so",
            "first", "second", "finally", "in conclusion"
        ]
        reasoning_score = sum(
            1 for keyword in reasoning_keywords
            if keyword.lower() in final_solution.lower()
        ) / len(reasoning_keywords)

        # Check if solution references previous contributions
        reference_score = sum(
            1 for i in range(1, self.n)
            if f"agent {i}" in final_solution.lower() or
               f"agent_{i}" in final_solution.lower() or
               any(word in final_solution.lower()
                   for word in ["previous", "earlier", "mentioned", "said"])
        ) / max(1, self.n - 1)

        # Combine scores
        final_score = (
            length_score * 0.3 +
            reasoning_score * 0.4 +
            reference_score * 0.3
        )

        return min(1.0, max(0.0, final_score))


# Initialize environment with desired number of agents
# You can change this number easily!
NUM_AGENTS = 3  # Try 3, 4, 5, or more!

set_environment(lambda: NAgentCollaborative(num_agents=NUM_AGENTS))


# Example: Create environments with different agent counts
def create_3_agent_env():
    set_environment(lambda: NAgentCollaborative(num_agents=3))

def create_5_agent_env():
    set_environment(lambda: NAgentCollaborative(num_agents=5))

def create_10_agent_env():
    set_environment(lambda: NAgentCollaborative(num_agents=10))


# For command-line configuration
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        print(f"Creating environment with {n} agents")
        set_environment(lambda: NAgentCollaborative(num_agents=n))
    else:
        print(f"Using default {NUM_AGENTS} agents")
        print("Usage: python multi_agent_n_collaborative.py <num_agents>")
