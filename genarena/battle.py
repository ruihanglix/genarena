# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Battle execution module with position debiasing."""

from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Optional, Union

from genarena.data import DataSample
from genarena.vlm import VLMJudge


logger = logging.getLogger(__name__)

def _progress_update(progress: Any, n: int) -> None:
    """
    Update progress for each API call.

    Supports:
    - queue-like objects with .put(int) (recommended for multi-thread/process)
    - tqdm-like objects with .update(int)
    """
    if progress is None or n <= 0:
        return
    if hasattr(progress, "put"):
        try:
            progress.put(n)
            return
        except Exception:
            return
    if hasattr(progress, "update"):
        try:
            progress.update(n)
        except Exception:
            return


@dataclass
class CallResult:
    """Result from a single VLM call."""

    raw_response: str
    parsed_result: Optional[dict[str, Any]]
    parse_success: bool
    parse_error: Optional[str] = None
    winner: Optional[str] = None  # "A" or "B" (position in that call)


@dataclass
class BattleResult:
    """Result from a complete battle (two VLM calls with position swap)."""

    # Final result
    final_winner: str  # "model_a", "model_b", or "tie"
    is_consistent: bool  # Whether both calls agreed

    # Individual call results
    original_call: CallResult
    swapped_call: CallResult

    # Metadata
    model_a: str = ""
    model_b: str = ""
    sample_index: int = 0

    # Converted winners (in terms of model_a/model_b, not A/B position)
    original_model_winner: Optional[str] = None  # model_a, model_b, or tie
    swapped_model_winner: Optional[str] = None   # model_a, model_b, or tie


def _call_vlm_judge(
    vlm: VLMJudge,
    prompt_module: ModuleType,
    instruction: str,
    input_images: list[Union[str, bytes]],
    image_a: Union[str, bytes],
    image_b: Union[str, bytes]
) -> CallResult:
    """
    Execute a single VLM judge call.

    Args:
        vlm: VLMJudge instance
        prompt_module: Prompt module with build_prompt and parse_response
        instruction: The editing instruction
        input_images: List of input images
        image_a: Output image for position A
        image_b: Output image for position B

    Returns:
        CallResult with raw response and parsed result
    """
    # Build prompt
    messages = prompt_module.build_prompt(
        instruction=instruction,
        input_images=input_images,
        output_image_a=image_a,
        output_image_b=image_b
    )

    # Call VLM
    try:
        raw_response = vlm.call(messages)
    except Exception as e:
        return CallResult(
            raw_response="",
            parsed_result=None,
            parse_success=False,
            parse_error=f"VLM call failed: {e}",
            winner=None
        )

    # Parse response
    try:
        parsed_result = prompt_module.parse_response(raw_response)
        winner = parsed_result.get("winner")
        return CallResult(
            raw_response=raw_response,
            parsed_result=parsed_result,
            parse_success=True,
            parse_error=None,
            winner=winner
        )
    except Exception as e:
        return CallResult(
            raw_response=raw_response,
            parsed_result=None,
            parse_success=False,
            parse_error=f"Parse failed: {e}",
            winner=None
        )


def _convert_position_winner_to_model(
    position_winner: Optional[str],
    is_swapped: bool
) -> Optional[str]:
    """
    Convert position-based winner (A/B) to model-based winner (model_a/model_b).

    In original order: A -> model_a, B -> model_b
    In swapped order: A -> model_b, B -> model_a

    Args:
        position_winner: "A", "B", "tie", or None
        is_swapped: Whether this was from the swapped call

    Returns:
        "model_a", "model_b", "tie", or None
    """
    if position_winner is None:
        return None

    winner_upper = position_winner.upper()

    if winner_upper == "TIE":
        return "tie"

    if not is_swapped:
        # Original order: A = model_a, B = model_b
        if winner_upper == "A":
            return "model_a"
        elif winner_upper == "B":
            return "model_b"
    else:
        # Swapped order: A = model_b, B = model_a
        if winner_upper == "A":
            return "model_b"
        elif winner_upper == "B":
            return "model_a"

    return None


def _combine_votes(
    original_model_winner: Optional[str],
    swapped_model_winner: Optional[str],
    allow_tie: bool
) -> tuple[str, bool]:
    """
    Combine two voting results to determine final winner.

    Position debiasing logic:
    - If both calls agree on a winner -> that model wins (consistent)
    - If calls disagree or either is tie/None -> tie (inconsistent)

    Args:
        original_model_winner: Winner from original call ("model_a", "model_b", "tie", None)
        swapped_model_winner: Winner from swapped call ("model_a", "model_b", "tie", None)
        allow_tie: Whether single-round ties are allowed by the prompt

    Returns:
        Tuple of (final_winner, is_consistent)
        - final_winner: "model_a", "model_b", or "tie"
        - is_consistent: True if both calls agreed
    """
    # Handle None cases (parse failures)
    if original_model_winner is None or swapped_model_winner is None:
        return "tie", False

    # Both are valid results
    if original_model_winner == swapped_model_winner:
        # Both agree
        if original_model_winner in ("model_a", "model_b"):
            return original_model_winner, True
        else:
            # Both returned tie (only possible if ALLOW_TIE=True)
            return "tie", True
    else:
        # Disagreement -> tie
        return "tie", False


def execute_battle(
    vlm: VLMJudge,
    prompt_module: ModuleType,
    sample: DataSample,
    model_a_output: Union[str, bytes],
    model_b_output: Union[str, bytes],
    model_a: str = "",
    model_b: str = "",
    parallel_swap_calls: bool = False,
    progress: Any = None,
) -> BattleResult:
    """
    Execute a complete battle with position debiasing.

    Makes two VLM calls:
    1. Original order: position A = model_a output, position B = model_b output
    2. Swapped order: position A = model_b output, position B = model_a output

    Then combines results according to ALLOW_TIE setting in prompt module.

    Args:
        vlm: VLMJudge instance
        prompt_module: Prompt module with build_prompt, parse_response, ALLOW_TIE
        sample: DataSample with instruction and input_images
        model_a_output: Output image path/bytes from model A
        model_b_output: Output image path/bytes from model B
        model_a: Model A name (for logging)
        model_b: Model B name (for logging)

    Returns:
        BattleResult with final winner and both call details
    """
    allow_tie = getattr(prompt_module, "ALLOW_TIE", False)

    if not parallel_swap_calls:
        # Call 1: Original order (A = model_a, B = model_b)
        logger.debug(f"Battle {model_a} vs {model_b}: executing original call")
        _progress_update(progress, 1)
        original_call = _call_vlm_judge(
            vlm=vlm,
            prompt_module=prompt_module,
            instruction=sample.instruction,
            input_images=sample.input_images,
            image_a=model_a_output,
            image_b=model_b_output
        )

        # Call 2: Swapped order (A = model_b, B = model_a)
        logger.debug(f"Battle {model_a} vs {model_b}: executing swapped call")
        _progress_update(progress, 1)
        swapped_call = _call_vlm_judge(
            vlm=vlm,
            prompt_module=prompt_module,
            instruction=sample.instruction,
            input_images=sample.input_images,
            image_a=model_b_output,
            image_b=model_a_output
        )
    else:
        # Execute original + swapped calls in parallel to reduce per-battle latency.
        # Note: This doubles the instantaneous request concurrency per battle.
        logger.debug(f"Battle {model_a} vs {model_b}: executing original+swapped calls in parallel")
        with ThreadPoolExecutor(max_workers=2) as executor:
            _progress_update(progress, 1)
            fut_original = executor.submit(
                _call_vlm_judge,
                vlm,
                prompt_module,
                sample.instruction,
                sample.input_images,
                model_a_output,
                model_b_output,
            )
            _progress_update(progress, 1)
            fut_swapped = executor.submit(
                _call_vlm_judge,
                vlm,
                prompt_module,
                sample.instruction,
                sample.input_images,
                model_b_output,
                model_a_output,
            )
            # Preserve error behavior: propagate exceptions if any occur.
            original_call = fut_original.result()
            swapped_call = fut_swapped.result()

    # Convert position winners to model winners
    original_model_winner = _convert_position_winner_to_model(
        original_call.winner, is_swapped=False
    )
    swapped_model_winner = _convert_position_winner_to_model(
        swapped_call.winner, is_swapped=True
    )

    # Combine votes
    final_winner, is_consistent = _combine_votes(
        original_model_winner,
        swapped_model_winner,
        allow_tie
    )

    logger.debug(
        f"Battle {model_a} vs {model_b}: "
        f"original={original_call.winner}->{original_model_winner}, "
        f"swapped={swapped_call.winner}->{swapped_model_winner}, "
        f"final={final_winner}, consistent={is_consistent}"
    )

    return BattleResult(
        final_winner=final_winner,
        is_consistent=is_consistent,
        original_call=original_call,
        swapped_call=swapped_call,
        model_a=model_a,
        model_b=model_b,
        sample_index=sample.index,
        original_model_winner=original_model_winner,
        swapped_model_winner=swapped_model_winner
    )
