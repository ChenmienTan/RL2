"""
Multi-Agent Code Review Environment

Collaborative code review with two agents:
- Reviewer: Identifies issues and suggests improvements
- Author: Responds to feedback and fixes issues

The goal is to improve code quality through iterative review.
"""

from typing import Dict, Any
import re

REVIEWER_ID = "reviewer"
AUTHOR_ID = "author"
MAX_ROUNDS = 2  # Number of review-fix cycles


def _get_code(sample: Dict[str, Any]) -> str:
    """Extract code from sample."""
    if "code" in sample:
        return sample["code"]
    if "prompt" in sample:
        return sample["prompt"]
    raise KeyError("Code review environment expects 'code' or 'prompt' in sample")


def _evaluate_code_quality(original_code: str, final_code: str, review_history: list) -> float:
    """
    Evaluate code improvement based on:
    - Number of issues identified
    - Quality of fixes
    - Code structure improvements

    Returns reward in [0, 1]
    """
    # Check if code was actually modified
    if original_code.strip() == final_code.strip():
        return 0.1  # Minimal reward for no changes

    # Count review comments (more thorough review = better)
    review_keywords = ["bug", "issue", "improve", "suggest", "consider", "should", "could"]
    review_quality = sum(
        1 for turn in review_history
        if any(keyword in turn.lower() for keyword in review_keywords)
    ) / max(1, len(review_history))

    # Check if fixes were applied (look for code blocks in author responses)
    code_blocks = sum(1 for turn in review_history if "```" in turn)
    fix_quality = min(1.0, code_blocks / (MAX_ROUNDS * 2))

    # Check for common improvements
    improvements = 0
    improvement_patterns = [
        (r"def\s+\w+\(.*\):", "function definitions"),
        (r"#.*", "comments"),
        (r"try:", "error handling"),
        (r"assert\s+", "assertions"),
        (r"if\s+.*:", "conditionals")
    ]

    for pattern, _ in improvement_patterns:
        original_count = len(re.findall(pattern, original_code))
        final_count = len(re.findall(pattern, final_code))
        if final_count > original_count:
            improvements += 1

    improvement_score = improvements / len(improvement_patterns)

    # Combine scores
    final_score = (review_quality * 0.3 + fix_quality * 0.4 + improvement_score * 0.3)

    return min(1.0, max(0.0, final_score))


async def reset(sample, tokenizer, extra_info, **kwargs) -> Dict[str, Any]:
    """
    Initialize code review environment.

    Returns:
        Initial state with reviewer ready to review code
    """
    code = _get_code(sample.sample)

    reviewer_prompt = (
        f"You are a code reviewer. Review the following code and identify issues, "
        f"bugs, or areas for improvement. Be specific and constructive.\n\n"
        f"Code to review:\n```python\n{code}\n```\n\n"
        f"Provide your review with specific suggestions:"
    )

    author_prompt = (
        f"You are the code author. Your code is being reviewed:\n```python\n{code}\n```\n\n"
        f"Wait for the reviewer's feedback, then address their concerns and provide improved code."
    )

    return {
        "agent_ids": [REVIEWER_ID, AUTHOR_ID],
        "current_agent": REVIEWER_ID,
        "next_observations": {
            REVIEWER_ID: tokenizer.apply_chat_template(
                [{"role": "user", "content": reviewer_prompt}],
                add_generation_prompt=True,
                tokenize=False
            ),
            AUTHOR_ID: tokenizer.apply_chat_template(
                [{"role": "user", "content": author_prompt}],
                add_generation_prompt=True,
                tokenize=False
            )
        },
        "rewards": {REVIEWER_ID: 0.0, AUTHOR_ID: 0.0},
        "done": False,
        "done_agents": [],
        "shared_info": {
            "round": 0,
            "original_code": code,
            "current_code": code,
            "review_history": []
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
    Process one agent's action in the code review cycle.

    Args:
        state: Current agent's observation
        action: Agent's response (review or fixed code)
        agent_id: Current agent (REVIEWER_ID or AUTHOR_ID)
        agent_states: All agents' current states
        shared_info: Shared review state

    Returns:
        Next state or final rewards
    """
    round_num = shared_info["round"]
    review_history = shared_info["review_history"] + [f"{agent_id.upper()}: {action}"]
    original_code = shared_info["original_code"]
    current_code = shared_info["current_code"]

    if agent_id == REVIEWER_ID:
        # Reviewer finished, author's turn to fix

        # Check if this is the final round
        if round_num >= MAX_ROUNDS:
            # Final review, calculate reward
            reward = _evaluate_code_quality(original_code, current_code, review_history)

            return {
                "rewards": {REVIEWER_ID: reward, AUTHOR_ID: reward},
                "scores": {REVIEWER_ID: reward, AUTHOR_ID: reward},
                "done": True,
                "done_agents": [REVIEWER_ID, AUTHOR_ID],
                "shared_info": {
                    "round": round_num,
                    "original_code": original_code,
                    "current_code": current_code,
                    "review_history": review_history,
                    "final_reward": reward
                },
                "extra_info": extra_info
            }

        # Continue to author's fix
        author_prompt = (
            f"Review feedback:\n{action}\n\n"
            f"Current code:\n```python\n{current_code}\n```\n\n"
            f"Address the reviewer's concerns and provide the improved code. "
            f"Wrap your code in ```python ... ``` blocks."
        )

        author_state = tokenizer.apply_chat_template(
            [{"role": "user", "content": author_prompt}],
            add_generation_prompt=True,
            tokenize=False
        )

        return {
            "current_agent": AUTHOR_ID,
            "next_observations": {AUTHOR_ID: author_state},
            "rewards": {REVIEWER_ID: 0.0, AUTHOR_ID: 0.0},
            "done": False,
            "done_agents": [REVIEWER_ID],
            "shared_info": {
                "round": round_num,
                "original_code": original_code,
                "current_code": current_code,
                "review_history": review_history
            },
            "extra_info": extra_info
        }

    else:  # AUTHOR_ID
        # Author finished fixing, extract new code
        code_match = re.search(r"```python\n(.*?)\n```", action, re.DOTALL)
        if code_match:
            new_code = code_match.group(1)
        else:
            # If no code block, use the whole response
            new_code = action

        # Next round of review
        next_round = round_num + 1

        reviewer_prompt = (
            f"The author has updated the code based on your feedback.\n\n"
            f"Updated code:\n```python\n{new_code}\n```\n\n"
            f"Review the changes. Are there any remaining issues? "
            f"Provide your final assessment."
        )

        reviewer_state = tokenizer.apply_chat_template(
            [{"role": "user", "content": reviewer_prompt}],
            add_generation_prompt=True,
            tokenize=False
        )

        return {
            "current_agent": REVIEWER_ID,
            "next_observations": {REVIEWER_ID: reviewer_state},
            "rewards": {REVIEWER_ID: 0.0, AUTHOR_ID: 0.0},
            "done": False,
            "done_agents": [AUTHOR_ID],
            "shared_info": {
                "round": next_round,
                "original_code": original_code,
                "current_code": new_code,
                "review_history": review_history
            },
            "extra_info": extra_info
        }


# Tokenizer fallback (same as debate environment)
def apply_chat_template(messages, add_generation_prompt=True, tokenize=False):
    content = messages[0]["content"]
    if add_generation_prompt:
        content += "\n\nAssistant:"
    return content

_tokenizer = None

def set_tokenizer(tokenizer):
    global _tokenizer
    _tokenizer = tokenizer

def get_tokenizer():
    return _tokenizer or type('obj', (object,), {'apply_chat_template': apply_chat_template})()
