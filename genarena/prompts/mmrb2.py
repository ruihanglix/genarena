# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""MMRB2 prompt implementation for image editing evaluation.

This module implements the MMRB2 evaluation prompt for pairwise comparison
of image editing results. It uses a 1-6 scoring scale and does not allow
ties in single rounds.

Reference: MMRB2 evaluation framework
"""

import base64
import io
import re
from typing import Any, Union

import json_repair
from PIL import Image as PILImage


# Whether single-round ties are allowed (mmrb2 requires a winner)
ALLOW_TIE = False


# The full evaluation prompt text from get_image_edit_prompt()
PROMPT_TEXT = """You are an expert in image editing quality analysis and AI evaluation. Your role is to act as an objective judge for comparing two AI-generated image editing responses to the same prompt. You will evaluate which response is better based on a comprehensive rubric specifically designed for image editing tasks.

**Important Guidelines:**
- Be completely impartial and avoid any position biases
- Ensure that the order in which the responses were presented does not influence your decision
- Do not allow the length of the responses to influence your evaluation
- Do not favor certain model names or types
- Be as objective as possible in your assessment
- Focus on image editing specific factors: faithfulness to editing instructions, preservation of input image elements, and overall editing quality

**Understanding the Content Structure:**
- **[ORIGINAL PROMPT TO MODEL:]**: This is the image editing instruction given to both AI models
- **[INPUT IMAGE FROM PROMPT:]**: This is the source image provided to both models for editing
- **[RESPONSE A:]**: The first model's edited image response
- **[RESPONSE B:]**: The second model's edited image response

Your evaluation must be based on a fine-grained rubric that covers the following criteria. For each criterion, you must provide detailed step-by-step reasoning comparing both responses. You will use a 1-6 scoring scale.

**Evaluation Criteria:**
1. **text_faithfulness:** Which response better adheres to the text editing instruction? Consider how well each response follows the specific editing instructions (e.g., adding objects, changing colors, modifying scenes).

2. **image_faithfulness:** Which response better respects and incorporates the key elements of the input image? Consider how well each response preserves important aspects of the original image (composition, lighting, style, background elements) while making the requested changes.

3. **overall_image_quality:** Which response has better general technical and aesthetic quality, with fewer visual artifacts, distortions, or inconsistencies introduced during the editing process?

4. **text_rendering:** If either response contains rendered text, which one has better text quality (spelling, legibility, integration with the image)? If no text is rendered, state "Not Applicable."

**Scoring Rubric:**
- Score 6 (A is significantly better): Response A is significantly superior across most criteria
- Score 5 (A is marginally better): Response A is noticeably better across several criteria
- Score 4 (Unsure or A is negligibly better): Response A is slightly better or roughly equivalent
- Score 3 (Unsure or B is negligibly better): Response B is slightly better or roughly equivalent
- Score 2 (B is marginally better): Response B is noticeably better across several criteria
- Score 1 (B is significantly better): Response B is significantly superior across most criteria

**Confidence Assessment:**
After your evaluation, assess your confidence in this judgment on a scale of 0.0 to 1.0:

**CRITICAL**: Be EXTREMELY conservative with confidence scores. Most comparisons should be in the 0.2-0.5 range.

- **Very High Confidence (0.8-1.0)**: ONLY for absolutely obvious cases where one response is dramatically better across ALL criteria with zero ambiguity. Use this extremely rarely (less than 10% of cases).
- **High Confidence (0.6-0.7)**: Clear differences but some uncertainty remains. Use sparingly (less than 20% of cases).
- **Medium Confidence (0.4-0.5)**: Noticeable differences but significant uncertainty. This should be your DEFAULT range.
- **Low Confidence (0.2-0.3)**: Very close comparison, difficult to distinguish. Responses are roughly equivalent or have conflicting strengths.
- **Very Low Confidence (0.0-0.1)**: Essentially indistinguishable responses or major conflicting strengths.

**IMPORTANT GUIDELINES**:
- DEFAULT to 0.3-0.5 range for most comparisons
- Only use 0.6+ when you are absolutely certain
- Consider: Could reasonable people disagree on this comparison?
- Consider: Are there any strengths in the "worse" response?
- Consider: How obvious would this be to a human evaluator?
- Remember: Quality assessment is inherently subjective

After your reasoning, you will provide a final numerical score, indicate which response is better, and assess your confidence. You must always output your response in the following structured JSON format:

{
    "reasoning": {
        "text_faithfulness": "YOUR REASONING HERE",
        "image_faithfulness": "YOUR REASONING HERE",
        "overall_image_quality": "YOUR REASONING HERE",
        "text_rendering": "YOUR REASONING HERE",
        "comparison_summary": "YOUR OVERALL COMPARISON SUMMARY HERE"
    },
    "score": <int 1-6>,
    "better_response": "A" or "B",
    "confidence": <float 0.0-1.0>,
    "confidence_rationale": "YOUR CONFIDENCE ASSESSMENT REASONING HERE"
}"""


def _encode_image_to_base64(image_source: Union[str, bytes, PILImage.Image, io.BytesIO, dict[str, Any]]) -> str:
    """
    Encode an image to base64.

    Args:
        image_source: Either a file path (str), raw bytes, PIL.Image object, or BytesIO

    Returns:
        Base64 encoded string

    Raises:
        TypeError: If image_source type is not supported
        ValueError: If image_source cannot be converted to bytes
    """
    image_bytes: bytes

    if isinstance(image_source, str):
        # It's a file path
        with open(image_source, "rb") as f:
            image_bytes = f.read()
    elif isinstance(image_source, io.BytesIO):
        # It's a BytesIO object
        image_source.seek(0)
        image_bytes = image_source.read()
    elif isinstance(image_source, PILImage.Image):
        # It's a PIL Image object (e.g., from HuggingFace datasets)
        buffer = io.BytesIO()
        image_source.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
    elif isinstance(image_source, dict):
        # It's a dict (e.g., from HuggingFace datasets Image() type)
        if "bytes" in image_source:
            raw = image_source["bytes"]
            if isinstance(raw, bytes):
                image_bytes = raw
            elif isinstance(raw, io.BytesIO):
                raw.seek(0)
                image_bytes = raw.read()
            else:
                # Recurse to handle nested types
                return _encode_image_to_base64(raw)
        elif "path" in image_source and image_source["path"]:
            with open(image_source["path"], "rb") as f:
                image_bytes = f.read()
        else:
            raise ValueError(f"Cannot extract image from dict: {image_source.keys()}")
    elif isinstance(image_source, bytes):
        # It's already bytes - MUST check after more specific types
        image_bytes = image_source
    else:
        # Unknown type - raise error with helpful message
        raise TypeError(
            f"Unsupported image type: {type(image_source).__name__}. "
            f"Expected str (path), bytes, PIL.Image, io.BytesIO, or dict. "
            f"Got: {repr(image_source)[:200]}"
        )

    # Verify we have valid bytes before encoding
    if not isinstance(image_bytes, bytes):
        raise ValueError(
            f"Failed to convert image to bytes. "
            f"Got {type(image_bytes).__name__} instead. "
            f"Original input was {type(image_source).__name__}"
        )

    return base64.b64encode(image_bytes).decode("utf-8")


def _get_image_media_type(image_source: Union[str, bytes, PILImage.Image]) -> str:
    """
    Determine the media type of an image.

    Args:
        image_source: Either a file path (str), raw bytes, or PIL.Image object

    Returns:
        Media type string (e.g., 'image/png')
    """
    if isinstance(image_source, str):
        ext = image_source.lower().split('.')[-1]
        media_types = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'webp': 'image/webp',
            'gif': 'image/gif',
        }
        return media_types.get(ext, 'image/png')
    elif isinstance(image_source, PILImage.Image):
        # For PIL.Image, we convert to PNG
        return 'image/png'
    else:
        # Try to detect from bytes magic
        if image_source[:8] == b'\x89PNG\r\n\x1a\n':
            return 'image/png'
        elif image_source[:2] == b'\xff\xd8':
            return 'image/jpeg'
        elif image_source[:4] == b'RIFF' and image_source[8:12] == b'WEBP':
            return 'image/webp'
        else:
            return 'image/png'


def _create_image_content(image_source: Union[str, bytes]) -> dict[str, Any]:
    """
    Create an image content block for OpenAI API.

    Args:
        image_source: Either a file path (str) or raw bytes

    Returns:
        Image content dict for OpenAI API
    """
    base64_data = _encode_image_to_base64(image_source)
    media_type = _get_image_media_type(image_source)

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{media_type};base64,{base64_data}"
        }
    }


def build_prompt(
    instruction: str,
    input_images: list[Union[str, bytes]],
    output_image_a: Union[str, bytes],
    output_image_b: Union[str, bytes]
) -> list[dict[str, Any]]:
    """
    Build the VLM prompt messages for pairwise evaluation.

    Constructs messages in the format:
    [EVALUATION PROMPT TEXT]
    [ORIGINAL PROMPT TO MODEL:]
    {instruction and input_images}
    [RESPONSE A:]
    {output_image_a}
    [RESPONSE B:]
    {output_image_b}

    Args:
        instruction: The editing instruction given to models
        input_images: List of input images (file paths or bytes)
        output_image_a: Output from model A (file path or bytes)
        output_image_b: Output from model B (file path or bytes)

    Returns:
        List of message dicts for OpenAI Chat Completion API
    """
    # Build content list
    content = []

    # 1. Evaluation prompt
    content.append({
        "type": "text",
        "text": PROMPT_TEXT
    })

    # 2. Original prompt to model section
    content.append({
        "type": "text",
        "text": "[ORIGINAL PROMPT TO MODEL:]"
    })

    # Add instruction text
    content.append({
        "type": "text",
        "text": instruction
    })

    # Add input images if any
    if input_images:
        content.append({
            "type": "text",
            "text": "[INPUT IMAGE FROM PROMPT:]"
        })
        for img in input_images:
            content.append(_create_image_content(img))

    # 3. Response A
    content.append({
        "type": "text",
        "text": "[RESPONSE A:]"
    })
    content.append(_create_image_content(output_image_a))

    # 4. Response B
    content.append({
        "type": "text",
        "text": "[RESPONSE B:]"
    })
    content.append(_create_image_content(output_image_b))

    # Return as OpenAI API format
    return [
        {
            "role": "user",
            "content": content
        }
    ]


def parse_response(response: str) -> dict[str, Any]:
    """
    Parse the VLM judge response.

    Extracts structured information from VLM's JSON response,
    handling markdown code blocks and minor JSON errors.

    Args:
        response: Raw response text from VLM

    Returns:
        Dict containing:
        - winner: "A" or "B" (from better_response field)
        - score: int 1-6
        - confidence: float 0.0-1.0
        - reasoning: dict with evaluation criteria
        - raw_response: the original parsed JSON

    Raises:
        ValueError: If response cannot be parsed
    """
    # Remove markdown code block formatting
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    # Try to parse JSON with json_repair for fault tolerance
    try:
        parsed = json_repair.loads(text)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON response: {e}\nResponse was:\n{response}")

    # Extract fields
    better_response = parsed.get("better_response", "")

    # Normalize winner to uppercase
    if isinstance(better_response, str):
        winner = better_response.upper().strip()
        if winner not in ("A", "B"):
            # Try to extract from text
            if "A" in winner:
                winner = "A"
            elif "B" in winner:
                winner = "B"
            else:
                raise ValueError(f"Invalid better_response value: {better_response}")
    else:
        raise ValueError(f"better_response must be a string, got: {type(better_response)}")

    # Extract score (1-6)
    score = parsed.get("score", 4)
    if isinstance(score, str):
        score = int(score)
    score = max(1, min(6, score))

    # Extract confidence (0.0-1.0)
    confidence = parsed.get("confidence", 0.5)
    if isinstance(confidence, str):
        confidence = float(confidence)
    confidence = max(0.0, min(1.0, confidence))

    # Extract reasoning
    reasoning = parsed.get("reasoning", {})

    return {
        "winner": winner,
        "score": score,
        "confidence": confidence,
        "reasoning": reasoning,
        "raw_response": parsed
    }
