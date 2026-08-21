"""
Core LLM-calling logic for the Workout Plan Generator.

generate_workout_plan() is the single function the UI depends on: it takes
structured inputs, builds the prompt, calls the Groq API in strict JSON mode,
parses the response into a WorkoutPlan, and always returns a
GenerationResult — it never raises.
"""

import json
import os
from typing import Optional

import groq

from prompts import SYSTEM_PROMPT, WORKOUT_PLAN_SCHEMA, build_user_prompt
from workout_models import Exercise, GenerationResult, WorkoutDay, WorkoutPlan, WorkoutRequest

DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
MODEL_NAME = os.environ.get("GROQ_MODEL", DEFAULT_MODEL_NAME)
MAX_TOKENS = 2000
TEMPERATURE = 0.7

VALID_GOALS = {"Build muscle", "Lose fat", "General fitness", "Improve endurance"}
VALID_EXPERIENCE = {"Beginner", "Intermediate", "Advanced"}
VALID_EQUIPMENT = {"No equipment", "Home dumbbells", "Full gym"}


def validate_request(request: WorkoutRequest) -> Optional[str]:
    """
    Validate structured inputs before spending an API call on them.

    Returns a friendly error message if something is missing/invalid,
    or None if the request is good to send.
    """
    if request.goal not in VALID_GOALS:
        return "Please select a valid fitness goal."

    if request.experience not in VALID_EXPERIENCE:
        return "Please select a valid experience level."

    if not isinstance(request.days_per_week, int) or not (1 <= request.days_per_week <= 7):
        return "Please choose a number of training days between 1 and 7."

    if not request.equipment:
        return "Please select at least one equipment option."
    if any(item not in VALID_EQUIPMENT for item in request.equipment):
        return "Please select a valid equipment option."

    if request.limitations and len(request.limitations) > 500:
        return "Injuries/limitations text is too long — please keep it under 500 characters."

    return None


def _get_client() -> groq.Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return groq.Groq(api_key=api_key)


def _parse_plan(raw_json: str) -> Optional[WorkoutPlan]:
    """
    Parse the model's JSON response into a WorkoutPlan.

    Returns None (rather than raising) if the JSON is malformed or doesn't
    contain a usable plan, so the caller can fall back to a friendly message.
    """
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return None

    try:
        days = [
            WorkoutDay(
                day_number=int(day["day_number"]),
                focus=str(day["focus"]),
                warm_up=str(day.get("warm_up", "")),
                cool_down=str(day.get("cool_down", "")),
                exercises=[
                    Exercise(
                        name=str(ex["name"]),
                        sets=int(ex["sets"]),
                        reps=str(ex["reps"]),
                        notes=str(ex.get("notes") or "") or None,
                    )
                    for ex in day.get("exercises", [])
                ],
            )
            for day in data.get("days", [])
        ]
    except (KeyError, TypeError, ValueError):
        return None

    if not days or not all(day.exercises for day in days):
        # A plan with no days, or a day with no exercises, isn't usable.
        return None

    return WorkoutPlan(
        summary=str(data.get("summary", "")),
        days=days,
        disclaimer=str(data.get("disclaimer") or "") or None,
        rest_recovery_note=str(data.get("rest_recovery_note") or "") or None,
    )


def _attempt_generation(
    client: groq.Groq, request: WorkoutRequest, regenerate: bool
) -> "tuple[GenerationResult, bool]":
    """
    Make a single Groq API call and return (result, retryable).

    `retryable` is True for failures that are likely a one-off generation
    glitch (empty/malformed JSON) rather than a systemic problem (bad key,
    rate limit, network, deprecated model) — those aren't worth retrying
    automatically.
    """
    user_prompt = build_user_prompt(request, regenerate=regenerate)
    temperature = min(TEMPERATURE + 0.15, 1.0) if regenerate else TEMPERATURE

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_schema", "json_schema": WORKOUT_PLAN_SCHEMA},
        )
    except groq.AuthenticationError:
        return GenerationResult(
            success=False,
            error="The Groq API key was rejected. Please check that GROQ_API_KEY is correct.",
        ), False
    except groq.RateLimitError:
        return GenerationResult(
            success=False,
            error="Groq's rate limit was hit. Please wait a moment and try again.",
        ), False
    except groq.APITimeoutError:
        return GenerationResult(
            success=False,
            error="The request to Groq timed out. Please try again.",
        ), False
    except groq.APIConnectionError:
        return GenerationResult(
            success=False,
            error="Couldn't reach the Groq API — please check your internet connection.",
        ), False
    except groq.APIStatusError as exc:
        if exc.status_code == 404:
            error_message = (
                f"The model '{MODEL_NAME}' wasn't found on Groq — it may have been "
                "deprecated. Check https://console.groq.com/docs/models for a current "
                "model ID and update MODEL_NAME in workout_generator.py."
            )
            return GenerationResult(success=False, error=error_message), False
        if exc.status_code == 400:
            # Groq validates the model's JSON against our schema server-side and
            # returns 400 (code "json_validate_failed") when the generation
            # itself doesn't comply — an occasional model glitch, worth retrying.
            return GenerationResult(
                success=False,
                error="The model returned a malformed plan. Please try generating again.",
            ), True
        error_message = f"Groq API returned an error (status {exc.status_code}). Please try again later."
        return GenerationResult(success=False, error=error_message), False
    except Exception:
        return GenerationResult(
            success=False,
            error="Something unexpected went wrong while generating your plan. Please try again.",
        ), False

    try:
        raw_content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError):
        raw_content = None

    if not raw_content:
        return GenerationResult(
            success=False,
            error="The model returned an empty response. Please try generating again.",
        ), True

    plan = _parse_plan(raw_content)
    if plan is None:
        return GenerationResult(
            success=False,
            error="The model returned an unexpected response format. Please try generating again.",
        ), True

    days_returned = len(plan.days)
    if days_returned != request.days_per_week:
        # Not fatal — still show the plan — but this is worth knowing about
        # while iterating on the prompt (see README's prompt design notes).
        plan.summary = (
            f"{plan.summary} (Note: {days_returned} day(s) were generated "
            f"instead of the requested {request.days_per_week}.)"
        ).strip()

    return GenerationResult(success=True, plan=plan), False


def generate_workout_plan(request: WorkoutRequest, regenerate: bool = False) -> GenerationResult:
    """
    Build a prompt from `request`, call the Groq API, and return the plan.

    Always returns a GenerationResult — API/network failures and malformed
    responses are captured as friendly error messages rather than raised.
    A single automatic retry is made if the model's response fails JSON
    validation, since that tends to be a one-off generation glitch.

    `regenerate=True` asks the model for a different variation of the plan
    (used by the app's "Regenerate" button) and nudges the sampling
    temperature up slightly so the result actually differs.
    """
    validation_error = validate_request(request)
    if validation_error:
        return GenerationResult(success=False, error=validation_error)

    try:
        client = _get_client()
    except RuntimeError:
        return GenerationResult(
            success=False,
            error="The app isn't configured with a Groq API key. "
            "Set GROQ_API_KEY in your environment and restart the app.",
        )

    result, retryable = _attempt_generation(client, request, regenerate)
    if not result.success and retryable:
        result, _ = _attempt_generation(client, request, regenerate)

    return result
