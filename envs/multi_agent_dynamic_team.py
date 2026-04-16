"""
Dynamic Team Environment

This example shows how to create teams with different roles and sizes.
Users can easily configure:
- Number of agents
- Agent roles
- Collaboration patterns
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from multi_agent_base import MultiAgentBase, set_environment
from typing import Dict, Any, List, Tuple, Optional


class DynamicTeam(MultiAgentBase):
    """
    Flexible team environment with configurable roles.

    Example configurations:
    - Software team: ["product_manager", "developer", "tester", "reviewer"]
    - Research team: ["researcher_1", "researcher_2", "reviewer", "editor"]
    - Creative team: ["ideator", "writer", "editor", "critic"]
    """

    def __init__(self, roles: List[str], role_descriptions: Dict[str, str] = None):
        """
        Initialize with custom roles.

        Args:
            roles: List of role names, e.g., ["manager", "worker_1", "worker_2"]
            role_descriptions: Optional descriptions for each role
        """
        self.roles = roles
        self.role_descriptions = role_descriptions or {}
        super().__init__()

    def get_agent_roles(self) -> List[str]:
        """Return configured roles."""
        return self.roles

    def get_initial_prompt(self, agent_id: str, sample: Dict[str, Any]) -> str:
        """Generate role-specific initial prompt."""
        task = self._get_task(sample)
        role_desc = self.role_descriptions.get(
            agent_id,
            f"You are {agent_id}"
        )

        agent_idx = self.roles.index(agent_id)
        total_agents = len(self.roles)

        if agent_idx == 0:
            # First agent
            return (
                f"{role_desc}\n\n"
                f"Task: {task}\n\n"
                f"You are the first in a team of {total_agents}. "
                f"Start the work and set the direction for the team."
            )
        elif agent_idx == total_agents - 1:
            # Last agent
            return (
                f"{role_desc}\n\n"
                f"Task: {task}\n\n"
                f"You are the final member of the team. "
                f"Review all previous work and provide the final output."
            )
        else:
            # Middle agents
            return (
                f"{role_desc}\n\n"
                f"Task: {task}\n\n"
                f"You are member {agent_idx + 1} of {total_agents}. "
                f"Build on previous work and contribute your expertise."
            )

    def process_action(
        self,
        agent_id: str,
        action: str,
        state: Dict[str, Any]
    ) -> Tuple[Optional[str], Dict[str, float], bool, Dict[str, Any]]:
        """Process action and determine next state."""
        agent_idx = self.roles.index(agent_id)

        # Store contribution
        contributions = state.get("contributions", {})
        contributions[agent_id] = action
        state["contributions"] = contributions

        # Check if last agent
        if agent_idx == len(self.roles) - 1:
            # Evaluate team performance
            reward = self._evaluate_team_work(state)
            rewards = {role: reward for role in self.roles}
            return None, rewards, True, state

        # Continue to next agent
        next_agent = self.roles[agent_idx + 1]
        rewards = {role: 0.0 for role in self.roles}
        return next_agent, rewards, False, state

    def build_next_prompt(
        self,
        next_agent: str,
        prev_agent: str,
        prev_action: str,
        state: Dict[str, Any],
        agent_states: Dict[str, str]
    ) -> str:
        """Build context-aware prompt for next agent."""
        contributions = state.get("contributions", {})
        next_idx = self.roles.index(next_agent)

        # Summarize previous work
        summary = []
        for i in range(next_idx):
            role = self.roles[i]
            if role in contributions:
                summary.append(f"{role}: {contributions[role]}")

        summary_text = "\n\n".join(summary)

        return (
            f"Previous team members' work:\n\n{summary_text}\n\n"
            f"Now it's your turn as {next_agent}. "
            f"Contribute your part based on the work above."
        )

    def _get_task(self, sample: Dict[str, Any]) -> str:
        """Extract task from sample."""
        for key in ["task", "problem", "question", "prompt"]:
            if key in sample:
                return sample[key]
        return str(sample)

    def _evaluate_team_work(self, state: Dict[str, Any]) -> float:
        """Evaluate overall team performance."""
        contributions = state.get("contributions", {})

        # Check all members contributed
        if len(contributions) != len(self.roles):
            return 0.1

        # Evaluate based on contribution quality
        total_length = sum(len(c.split()) for c in contributions.values())
        length_score = min(1.0, total_length / (len(self.roles) * 30))

        # Check for collaboration (references to other members)
        collab_score = 0
        for role, contribution in contributions.items():
            # Check if this member references others
            references = sum(
                1 for other_role in self.roles
                if other_role != role and other_role.lower() in contribution.lower()
            )
            collab_score += min(1.0, references / max(1, len(self.roles) - 1))

        collab_score /= len(self.roles)

        # Final score
        return (length_score * 0.5 + collab_score * 0.5)


# Predefined team configurations

def create_software_team():
    """Create a software development team."""
    roles = ["product_manager", "developer", "tester", "reviewer"]
    descriptions = {
        "product_manager": "You are the Product Manager. Define requirements and priorities.",
        "developer": "You are the Developer. Implement the solution based on requirements.",
        "tester": "You are the Tester. Test the implementation and find issues.",
        "reviewer": "You are the Code Reviewer. Review everything and provide final assessment."
    }
    set_environment(lambda: DynamicTeam(roles, descriptions))
    return len(roles)


def create_research_team(num_researchers: int = 2):
    """Create a research team with N researchers."""
    roles = [f"researcher_{i+1}" for i in range(num_researchers)] + ["reviewer", "editor"]
    descriptions = {
        **{f"researcher_{i+1}": f"You are Researcher {i+1}. Conduct research and share findings."
           for i in range(num_researchers)},
        "reviewer": "You are the Reviewer. Evaluate all research findings critically.",
        "editor": "You are the Editor. Synthesize all research and write the final report."
    }
    set_environment(lambda: DynamicTeam(roles, descriptions))
    return len(roles)


def create_creative_team():
    """Create a creative writing team."""
    roles = ["brainstormer", "writer", "editor", "critic"]
    descriptions = {
        "brainstormer": "You are the Brainstormer. Generate creative ideas and concepts.",
        "writer": "You are the Writer. Turn ideas into a compelling narrative.",
        "editor": "You are the Editor. Refine the writing and improve clarity.",
        "critic": "You are the Critic. Provide constructive feedback and final assessment."
    }
    set_environment(lambda: DynamicTeam(roles, descriptions))
    return len(roles)


def create_custom_team(roles: List[str], descriptions: Dict[str, str] = None):
    """
    Create a custom team with any roles.

    Args:
        roles: List of role names
        descriptions: Optional role descriptions

    Returns:
        Number of agents in the team
    """
    set_environment(lambda: DynamicTeam(roles, descriptions))
    return len(roles)


# Default: 4-agent software team
NUM_AGENTS = create_software_team()


# Command-line interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        team_type = sys.argv[1]

        if team_type == "software":
            n = create_software_team()
            print(f"Created software team with {n} agents")

        elif team_type == "research":
            num_researchers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
            n = create_research_team(num_researchers)
            print(f"Created research team with {n} agents ({num_researchers} researchers)")

        elif team_type == "creative":
            n = create_creative_team()
            print(f"Created creative team with {n} agents")

        elif team_type == "custom":
            # Custom roles from command line
            roles = sys.argv[2].split(",")
            n = create_custom_team(roles)
            print(f"Created custom team with {n} agents: {roles}")

        else:
            print(f"Unknown team type: {team_type}")
            print("Available types: software, research, creative, custom")
    else:
        print(f"Using default software team with {NUM_AGENTS} agents")
        print("Usage:")
        print("  python multi_agent_dynamic_team.py software")
        print("  python multi_agent_dynamic_team.py research 3")
        print("  python multi_agent_dynamic_team.py creative")
        print("  python multi_agent_dynamic_team.py custom role1,role2,role3")
