"""
Multi-Agent Debate Environment

Two agents debate a given topic:
- Pro: Argues for the proposition
- Con: Argues against the proposition

After MAX_TURNS rounds, a judge evaluates the debate quality.
"""

from typing import Dict, Any
import re

PRO_AGENT = "pro"
CON_AGENT = "con"
MAX_TURNS = 3  # Each agent speaks 3 times


def _get_topic(sample: Dict[str, Any]) -> str:
    """Extract debate topic from sample."""
    if "topic" in sample:
        return sample["topic"]
    if "prompt" in sample:
        return sample["prompt"]
    raise KeyError("Debate environment expects 'topic' or 'prompt' in sample")


def _evaluate_debate(debate_history: list, topic: str) -> float:
    """
    Evaluate debate quality based on:
    - Argument coherence
    - Evidence usage
    - Rebuttal quality
    - Overall persuasiveness

    Returns reward in [0, 1]
    """
    # Simple heuristic: longer, more structured arguments = better
    total_length = sum(len(turn) for turn in debate_history)

    # Check for argument structure keywords
    structure_keywords = ["because", "therefore", "however", "evidence", "studies show", "for example"]
    structure_score = sum(
        1 for turn in debate_history
        for keyword in structure_keywords
        if keyword.lower() in turn.lower()
    ) / (len(debate_history) * len(structure_keywords))

    # Check for rebuttals (mentioning opponent's points)
    rebuttal_keywords = ["you said", "your argument", "however", "but", "on the contrary"]
    rebuttal_score = sum(
        1 for turn in debate_history[2:]  # Skip first two turns
        for keyword in rebuttal_keywords
        if keyword.lower() in turn.lower()
    ) / max(1, len(debate_history) - 2)

    # Combine scores
    length_score = min(1.0, total_length / 1000)  # Normalize by expected length
    final_score = (length_score * 0.3 + structure_score * 0.4 + rebuttal_score * 0.3)

    return min(1.0, max(0.0, final_score))


async def reset(sample, tokenizer, extra_info, **kwargs) -> Dict[str, Any]:
    """
    Initialize debate environment.

    Returns:
        Initial state with both agents ready to debate
    """
    topic = _get_topic(sample.sample)

    pro_prompt = (
        f"You are debating the topic: '{topic}'\n"
        f"You are arguing FOR this proposition.\n"
        f"Present your opening argument with clear reasoning and evidence.\n"
        f"Be persuasive and logical."
    )

    con_prompt = (
        f"You are debating the topic: '{topic}'\n"
        f"You are arguing AGAINST this proposition.\n"
        f"Wait for the Pro side to present their opening argument, "
        f"then you will present your counter-argument."
    )

    return {
        "agent_ids": [PRO_AGENT, CON_AGENT],
        "current_agent": PRO_AGENT,
        "next_observations": {
            PRO_AGENT: tokenizer.apply_chat_template(
                [{"role": "user", "content": pro_prompt}],
                add_generation_prompt=True,
                tokenize=False
            ),
            CON_AGENT: tokenizer.apply_chat_template(
                [{"role": "user", "content": con_prompt}],
                add_generation_prompt=True,
                tokenize=False
            )
        },
        "rewards": {PRO_AGENT: 0.0, CON_AGENT: 0.0},
        "done": False,
        "done_agents": [],
        "shared_info": {
            "turn": 0,
            "debate_history": [],
            "topic": topic
        },
        "extra_info": extra_info
    }


async def step(
    state: str,
    action: str,
    extra_info: Dict[str, Any],
    agent_id: str,
    agent_states: Dict[str, str],
    shared_info: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Process one agent's argument and transition to next state.

    Args:
        state: Current agent's observation
        action: Agent's argument
        agent_id: Current agent (PRO_AGENT or CON_AGENT)
        agent_states: All agents' current states
        shared_info: Shared debate state

    Returns:
        Next state or final rewards
    """
    turn = shared_info["turn"] + 1
    debate_history = shared_info["debate_history"] + [f"{agent_id.upper()}: {action}"]
    topic = shared_info["topic"]

    # Check if debate should end
    if turn >= MAX_TURNS * 2:  # Each agent speaks MAX_TURNS times
        # Evaluate debate quality
        reward = _evaluate_debate(debate_history, topic)

        return {
            "rewards": {PRO_AGENT: reward, CON_AGENT: reward},
            "scores": {PRO_AGENT: reward, CON_AGENT: reward},
            "done": True,
            "done_agents": [PRO_AGENT, CON_AGENT],
            "shared_info": {
                "turn": turn,
                "debate_history": debate_history,
                "topic": topic,
                "final_reward": reward
            },
            "extra_info": extra_info
        }

    # Continue debate - switch to other agent
    next_agent = CON_AGENT if agent_id == PRO_AGENT else PRO_AGENT

    # Build context for next agent
    debate_context = "\n\n".join(debate_history)

    if next_agent == PRO_AGENT:
        next_prompt = (
            f"Debate so far:\n{debate_context}\n\n"
            f"The Con side has presented their argument. "
            f"Now present your rebuttal and strengthen your position FOR the proposition."
        )
    else:
        next_prompt = (
            f"Debate so far:\n{debate_context}\n\n"
            f"The Pro side has presented their argument. "
            f"Now present your counter-argument AGAINST the proposition."
        )

    next_state = tokenizer.apply_chat_template(
        [{"role": "user", "content": next_prompt}],
        add_generation_prompt=True,
        tokenize=False
    )

    return {
        "current_agent": next_agent,
        "next_observations": {next_agent: next_state},
        "rewards": {PRO_AGENT: 0.0, CON_AGENT: 0.0},
        "done": False,
        "done_agents": [agent_id],
        "shared_info": {
            "turn": turn,
            "debate_history": debate_history,
            "topic": topic
        },
        "extra_info": extra_info
    }


# For compatibility with tokenizer parameter
def apply_chat_template(messages, add_generation_prompt=True, tokenize=False):
    """Fallback if tokenizer not available in step()"""
    content = messages[0]["content"]
    if add_generation_prompt:
        content += "\n\nAssistant:"
    return content


# Store tokenizer reference
_tokenizer = None

def set_tokenizer(tokenizer):
    global _tokenizer
    _tokenizer = tokenizer

# Use in step if needed
def get_tokenizer():
    return _tokenizer or type('obj', (object,), {'apply_chat_template': apply_chat_template})()
