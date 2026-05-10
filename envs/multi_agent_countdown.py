from typing import Dict, Any
import re

PLANNER_ID = "planner"
SOLVER_ID = "solver"


def _get_question(sample: Dict[str, Any]) -> str:
    if "prompt" in sample:
        return sample["prompt"]
    if "numbers" in sample and "target" in sample:
        numbers = ", ".join(map(str, sample["numbers"]))
        return (
            f"Use the numbers [{numbers}] exactly once to make {sample['target']}. "
            "Return a valid arithmetic expression."
        )
    raise KeyError(
        "multi_agent_countdown expects either `prompt` or `numbers` + `target`."
    )


def _planner_prompt(question: str) -> str:
    return (
        f"Question: {question}\n"
        "You are the planner agent. Think about the solution strategy and "
        "share short notes for the solver. Do not output the final answer."
    )


def _solver_prompt(question: str) -> str:
    return (
        f"Question: {question}\n"
        "You are the solver agent. Wait for the planner's notes, then give the "
        "final arithmetic expression inside <answer></answer>."
    )


def _compute_countdown_reward(sample: Dict[str, Any], action: str) -> float:
    match = re.search(r"<answer>(.*?)</answer>", action, re.DOTALL)
    equation = match.group(1).strip() if match is not None else action.strip()

    if "target" not in sample or "numbers" not in sample:
        answer = str(
            sample.get("extra_info", {}).get(
                "answer",
                sample.get("answer", "")
            )
        ).strip()
        return float(equation == answer)

    reward = 0.1
    try:
        numbers = [int(n) for n in re.findall(r"\d+", equation)]
        assert sorted(numbers) == sorted(sample["numbers"])
    except Exception:
        return 0.0

    try:
        assert re.match(r"^[\d+\-*/().\s]+$", equation)
        result = eval(equation, {"__builtins__": None}, {})
        assert abs(result - sample["target"]) < 1e-5
        reward = 1.0
    except Exception:
        pass
    return reward


async def reset(sample, tokenizer, extra_info, **kwargs) -> Dict[str, Any]:
    question = _get_question(sample.sample)
    return {
        "agent_ids": [PLANNER_ID, SOLVER_ID],
        "current_agent": PLANNER_ID,
        "next_observations": {
            PLANNER_ID: tokenizer.apply_chat_template(
                [{"role": "user", "content": _planner_prompt(question)}],
                add_generation_prompt=True,
                tokenize=False
            ),
            SOLVER_ID: tokenizer.apply_chat_template(
                [{"role": "user", "content": _solver_prompt(question)}],
                add_generation_prompt=True,
                tokenize=False
            )
        },
        "rewards": {PLANNER_ID: 0.0, SOLVER_ID: 0.0},
        "done": False,
        "done_agents": [],
        "shared_info": {},
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
    if agent_id == PLANNER_ID:
        solver_state = (
            agent_states[SOLVER_ID] +
            f"\nPlanner notes:\n{action}\n"
            "Now provide the final arithmetic expression inside <answer></answer>."
        )
        return {
            "current_agent": SOLVER_ID,
            "next_observations": {SOLVER_ID: solver_state},
            "rewards": {PLANNER_ID: 0.0, SOLVER_ID: 0.0},
            "done": False,
            "done_agents": [PLANNER_ID],
            "shared_info": shared_info,
            "extra_info": extra_info
        }

    sample_obj = kwargs.get("sample")
    raw_sample = sample_obj.sample if sample_obj is not None else {}
    reward = _compute_countdown_reward(
        {
            **raw_sample,
            "extra_info": extra_info
        },
        action
    )
    return {
        "rewards": {
            PLANNER_ID: reward,
            SOLVER_ID: reward
        },
        "scores": {
            PLANNER_ID: reward,
            SOLVER_ID: reward
        },
        "done": True,
        "done_agents": [PLANNER_ID, SOLVER_ID],
        "shared_info": {
            **shared_info,
            "global_reward": reward
        },
        "extra_info": extra_info
    }
