"""
Multi-Agent Story Writing Environment

Three agents collaborate to write a story:
- Planner: Creates story outline and plot points
- Writer: Writes the actual story content
- Editor: Reviews and suggests improvements

This demonstrates a 3-agent sequential workflow.
"""

from typing import Dict, Any
import re

PLANNER_ID = "planner"
WRITER_ID = "writer"
EDITOR_ID = "editor"


def _get_prompt(sample: Dict[str, Any]) -> str:
    """Extract story prompt from sample."""
    if "prompt" in sample:
        return sample["prompt"]
    if "topic" in sample:
        return sample["topic"]
    if "theme" in sample:
        return f"Write a story about: {sample['theme']}"
    raise KeyError("Story environment expects 'prompt', 'topic', or 'theme' in sample")


def _evaluate_story(plan: str, story: str, edits: str) -> float:
    """
    Evaluate story quality based on:
    - Plan completeness
    - Story length and structure
    - Editor feedback quality

    Returns reward in [0, 1]
    """
    # Check plan quality (should have structure)
    plan_keywords = ["beginning", "middle", "end", "character", "conflict", "resolution", "plot"]
    plan_score = sum(1 for keyword in plan_keywords if keyword.lower() in plan.lower()) / len(plan_keywords)

    # Check story length (reasonable length)
    story_length = len(story.split())
    length_score = min(1.0, story_length / 200)  # Target ~200 words

    # Check story structure (paragraphs, dialogue, etc.)
    has_paragraphs = story.count('\n\n') >= 2
    has_dialogue = '"' in story or "'" in story
    structure_score = (0.5 if has_paragraphs else 0.0) + (0.5 if has_dialogue else 0.0)

    # Check editor feedback quality
    editor_keywords = ["good", "improve", "suggest", "consider", "well-written", "could", "should"]
    editor_score = sum(1 for keyword in editor_keywords if keyword.lower() in edits.lower()) / len(editor_keywords)

    # Combine scores
    final_score = (plan_score * 0.2 + length_score * 0.3 + structure_score * 0.3 + editor_score * 0.2)

    return min(1.0, max(0.0, final_score))


async def reset(sample, tokenizer, extra_info, **kwargs) -> Dict[str, Any]:
    """
    Initialize story writing environment.

    Returns:
        Initial state with planner ready to create outline
    """
    prompt = _get_prompt(sample.sample)

    planner_prompt = (
        f"You are a story planner. Create a detailed outline for a story based on this prompt:\n"
        f"'{prompt}'\n\n"
        f"Include:\n"
        f"- Main characters\n"
        f"- Setting\n"
        f"- Beginning, middle, and end\n"
        f"- Key plot points\n"
        f"- Conflict and resolution\n\n"
        f"Provide a clear outline:"
    )

    writer_prompt = (
        f"You are a story writer. You will receive a story outline and write the actual story.\n"
        f"Wait for the planner's outline."
    )

    editor_prompt = (
        f"You are a story editor. You will review the written story and provide feedback.\n"
        f"Wait for the writer to finish the story."
    )

    return {
        "agent_ids": [PLANNER_ID, WRITER_ID, EDITOR_ID],
        "current_agent": PLANNER_ID,
        "next_observations": {
            PLANNER_ID: tokenizer.apply_chat_template(
                [{"role": "user", "content": planner_prompt}],
                add_generation_prompt=True,
                tokenize=False
            ),
            WRITER_ID: tokenizer.apply_chat_template(
                [{"role": "user", "content": writer_prompt}],
                add_generation_prompt=True,
                tokenize=False
            ),
            EDITOR_ID: tokenizer.apply_chat_template(
                [{"role": "user", "content": editor_prompt}],
                add_generation_prompt=True,
                tokenize=False
            )
        },
        "rewards": {PLANNER_ID: 0.0, WRITER_ID: 0.0, EDITOR_ID: 0.0},
        "done": False,
        "done_agents": [],
        "shared_info": {
            "prompt": prompt,
            "plan": None,
            "story": None,
            "edits": None
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
    Process one agent's contribution to the story.

    Workflow:
    1. Planner creates outline
    2. Writer writes story based on outline
    3. Editor reviews and provides feedback
    4. Calculate final reward

    Args:
        state: Current agent's observation
        action: Agent's output
        agent_id: Current agent
        agent_states: All agents' states
        shared_info: Shared story state

    Returns:
        Next state or final rewards
    """
    if agent_id == PLANNER_ID:
        # Planner finished, pass outline to writer
        writer_prompt = (
            f"Here is the story outline:\n\n{action}\n\n"
            f"Now write a complete story based on this outline. "
            f"Make it engaging and well-structured with clear paragraphs."
        )

        writer_state = tokenizer.apply_chat_template(
            [{"role": "user", "content": writer_prompt}],
            add_generation_prompt=True,
            tokenize=False
        )

        return {
            "current_agent": WRITER_ID,
            "next_observations": {WRITER_ID: writer_state},
            "rewards": {PLANNER_ID: 0.0, WRITER_ID: 0.0, EDITOR_ID: 0.0},
            "done": False,
            "done_agents": [PLANNER_ID],
            "shared_info": {
                **shared_info,
                "plan": action
            },
            "extra_info": extra_info
        }

    elif agent_id == WRITER_ID:
        # Writer finished, pass story to editor
        editor_prompt = (
            f"Review this story and provide constructive feedback:\n\n{action}\n\n"
            f"Consider:\n"
            f"- Story structure and flow\n"
            f"- Character development\n"
            f"- Engagement and readability\n"
            f"- Grammar and style\n\n"
            f"Provide your editorial feedback:"
        )

        editor_state = tokenizer.apply_chat_template(
            [{"role": "user", "content": editor_prompt}],
            add_generation_prompt=True,
            tokenize=False
        )

        return {
            "current_agent": EDITOR_ID,
            "next_observations": {EDITOR_ID: editor_state},
            "rewards": {PLANNER_ID: 0.0, WRITER_ID: 0.0, EDITOR_ID: 0.0},
            "done": False,
            "done_agents": [PLANNER_ID, WRITER_ID],
            "shared_info": {
                **shared_info,
                "story": action
            },
            "extra_info": extra_info
        }

    else:  # EDITOR_ID
        # Editor finished, calculate final reward
        plan = shared_info["plan"]
        story = shared_info["story"]
        edits = action

        reward = _evaluate_story(plan, story, edits)

        return {
            "rewards": {PLANNER_ID: reward, WRITER_ID: reward, EDITOR_ID: reward},
            "scores": {PLANNER_ID: reward, WRITER_ID: reward, EDITOR_ID: reward},
            "done": True,
            "done_agents": [PLANNER_ID, WRITER_ID, EDITOR_ID],
            "shared_info": {
                **shared_info,
                "edits": edits,
                "final_reward": reward
            },
            "extra_info": extra_info
        }


# Tokenizer fallback
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
