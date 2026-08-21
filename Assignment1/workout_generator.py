"""
Core LLM-calling logic for the Workout Plan Generator.

generate_workout_plan() is the main entry point the UI depends on: it takes
structured inputs, builds the prompt, calls the Groq API in strict JSON mode,
parses the response into a WorkoutPlan, and always returns a
GenerationResult — it never raises.

swap_exercise() is a smaller sibling used by the "Swap this exercise"
feature: it asks for a single replacement exercise rather than regenerating
the whole plan.
"""

import json
import os
from typing import Optional, Tuple

import groq

from prompts import (
    EXERCISE_SWAP_SCHEMA,
    SWAP_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    WORKOUT_PLAN_SCHEMA,
    build_swap_user_prompt,
    build_user_prompt,
)
from workout_models import (
    Exercise,
    ExerciseSwapResult,
    GenerationResult,
    WorkoutDay,
    WorkoutPlan,
    WorkoutRequest,
)

DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
MODEL_NAME = os.environ.get("GROQ_MODEL", DEFAULT_MODEL_NAME)
MAX_TOKENS = 2000
TEMPERATURE = 0.7

SWAP_MAX_TOKENS = 1000  # openai/gpt-oss-120b spends tokens on internal reasoning before
# emitting the JSON itself, so even a single-exercise response needs real headroom —
# a smaller budget (tested down to 300) reliably hit "max tokens reached" 400s.
SWAP_TEMPERATURE = 0.9  # a bit more variety is desirable for "give me something different"

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


def _exercise_from_dict(data: dict) -> Exercise:
    """Build an Exercise from a parsed JSON object. Raises on bad shape."""
    return Exercise(
        name=str(data["name"]),
        sets=int(data["sets"]),
        reps=str(data["reps"]),
        notes=str(data.get("notes") or "") or None,
    )


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
                exercises=[_exercise_from_dict(ex) for ex in day.get("exercises", [])],
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


def _parse_exercise(raw_json: str) -> Optional[Exercise]:
    """Parse the model's JSON response for a single exercise. Returns None on failure."""
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return None

    try:
        return _exercise_from_dict(data)
    except (KeyError, TypeError, ValueError):
        return None


def _call_groq_structured(
    client: groq.Groq,
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    temperature: float,
    max_tokens: int,
) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Make one Groq structured-output call.

    Returns (raw_json_content, error_message, retryable) — exactly one of
    raw_json_content / error_message is non-None. `retryable` is True only
    for failures that look like a one-off generation glitch (empty response,
    or a schema-validation 400) rather than a systemic problem (bad key,
    rate limit, network, deprecated model).
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_schema", "json_schema": schema},
        )
    except groq.AuthenticationError:
        return None, "The Groq API key was rejected. Please check that GROQ_API_KEY is correct.", False
    except groq.RateLimitError:
        return None, "Groq's rate limit was hit. Please wait a moment and try again.", False
    except groq.APITimeoutError:
        return None, "The request to Groq timed out. Please try again.", False
    except groq.APIConnectionError:
        return None, "Couldn't reach the Groq API — please check your internet connection.", False
    except groq.APIStatusError as exc:
        if exc.status_code == 404:
            return None, (
                f"The model '{MODEL_NAME}' wasn't found on Groq — it may have been "
                "deprecated. Check https://console.groq.com/docs/models for a current "
                "model ID and update GROQ_MODEL accordingly."
            ), False
        if exc.status_code == 400:
            # Groq validates the model's JSON against our schema server-side and
            # returns 400 (code "json_validate_failed") when the generation
            # itself doesn't comply — an occasional model glitch, worth retrying.
            return None, "The model returned a malformed response. Please try again.", True
        return None, f"Groq API returned an error (status {exc.status_code}). Please try again later.", False
    except Exception:
        return None, "Something unexpected went wrong. Please try again.", False

    try:
        raw_content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError):
        raw_content = None

    if not raw_content:
        return None, "The model returned an empty response. Please try again.", True

    return raw_content, None, False


def _attempt_generation(
    client: groq.Groq, request: WorkoutRequest, regenerate: bool
) -> Tuple[GenerationResult, bool]:
    """Make a single attempt at generating the full plan. Returns (result, retryable)."""
    user_prompt = build_user_prompt(request, regenerate=regenerate)
    temperature = min(TEMPERATURE + 0.15, 1.0) if regenerate else TEMPERATURE

    raw_content, error, retryable = _call_groq_structured(
        client, SYSTEM_PROMPT, user_prompt, WORKOUT_PLAN_SCHEMA, temperature, MAX_TOKENS
    )
    if error:
        return GenerationResult(success=False, error=error), retryable

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


def _attempt_swap(
    client: groq.Groq, request: WorkoutRequest, day_focus: str, current_exercise: Exercise
) -> Tuple[ExerciseSwapResult, bool]:
    """Make a single attempt at swapping one exercise. Returns (result, retryable)."""
    user_prompt = build_swap_user_prompt(request, day_focus, current_exercise)

    raw_content, error, retryable = _call_groq_structured(
        client, SWAP_SYSTEM_PROMPT, user_prompt, EXERCISE_SWAP_SCHEMA, SWAP_TEMPERATURE, SWAP_MAX_TOKENS
    )
    if error:
        return ExerciseSwapResult(success=False, error=error), retryable

    exercise = _parse_exercise(raw_content)
    if exercise is None:
        return ExerciseSwapResult(
            success=False,
            error="The model returned an unexpected response format. Please try swapping again.",
        ), True

    return ExerciseSwapResult(success=True, exercise=exercise), False


def swap_exercise(request: WorkoutRequest, day_focus: str, current_exercise: Exercise) -> ExerciseSwapResult:
    """
    Ask the LLM for a single alternative to `current_exercise` within a day
    of the given focus, respecting the same equipment/experience/limitation
    constraints as the original plan.

    Always returns an ExerciseSwapResult — never raises. Retries once on a
    malformed/empty response, same policy as generate_workout_plan.
    """
    validation_error = validate_request(request)
    if validation_error:
        return ExerciseSwapResult(success=False, error=validation_error)

    try:
        client = _get_client()
    except RuntimeError:
        return ExerciseSwapResult(
            success=False,
            error="The app isn't configured with a Groq API key. "
            "Set GROQ_API_KEY in your environment and restart the app.",
        )

    result, retryable = _attempt_swap(client, request, day_focus, current_exercise)
    if not result.success and retryable:
        result, _ = _attempt_swap(client, request, day_focus, current_exercise)

    return result
