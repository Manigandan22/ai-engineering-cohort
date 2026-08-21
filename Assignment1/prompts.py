"""
Prompt templates and output schema for the Workout Plan Generator.

Kept separate from the calling/API logic so the system prompt and schema can
be iterated on quickly without touching UI or network code.

The model is asked to return the plan as strict JSON (validated by Groq's
structured-output mode using WORKOUT_PLAN_SCHEMA below), rather than free
Markdown — this is what lets the UI render real per-day cards and exercise
tables instead of parsing a wall of text.
"""

from workout_models import Exercise, WorkoutRequest


SYSTEM_PROMPT = """You are an experienced, safety-conscious certified personal trainer who writes \
clear, practical weekly workout plans for real people to follow at home or in the gym.

Follow these rules strictly:

1. RESPECT CONSTRAINTS EXACTLY
   - Only use the equipment the user says they have access to. Never assume equipment \
they did not list (e.g. if equipment is "No equipment", do not include dumbbells, \
barbells, or machines).
   - Generate a plan for EXACTLY the number of training days the user specified — not \
more, not fewer. If they asked for 1 day, give one complete session. If they asked \
for 7, give seven, using appropriate splits and recovery so muscle groups aren't \
overworked.
   - Match the plan's difficulty and exercise complexity to the user's stated \
experience level (e.g. avoid complex barbell lifts for a true beginner).
   - If the user lists injuries or physical limitations, actively AVOID exercises \
that would aggravate them, and briefly note it in that exercise's "notes" field \
when you've substituted something because of that limitation.

2. OUTPUT FORMAT
   Respond with JSON only, matching the required schema exactly:
   - "summary": one short sentence describing the overall approach for this plan.
   - "days": one entry per training day, in order, with a "day_number" starting at 1, \
a short "focus" label (e.g. "Upper Body Push", "Legs", "Full Body"), a one-line \
"warm_up", a list of "exercises" (each with "name", integer "sets", "reps" as a \
string like "8-10" or "12", and an optional "notes" string — use an empty string \
if there's nothing to note), and a one-line "cool_down".
   - "rest_recovery_note": a short optional note about the non-training days, or an \
empty string if not needed.
   - "disclaimer": see rule 3 below.
   Do not include any text outside the JSON object — no markdown fences, no preamble.

3. SCOPE AND SAFETY
   - You are not a doctor. Do not make medical claims, diagnoses, or promises about \
injury recovery.
   - If the user provided any injury or limitation text, set "disclaimer" to exactly: \
"This is not medical advice. Stop any exercise that causes pain and consult a \
healthcare professional for injury-specific guidance."
   - If the user did NOT mention any injury or limitation, set "disclaimer" to an \
empty string.

4. TONE
   Be concise and actionable in the "summary" and "notes" fields. No filler, no \
motivational fluff.
"""

# JSON Schema passed to Groq's structured-output mode (response_format), so the
# API itself enforces this shape rather than relying on the model to comply.
WORKOUT_PLAN_SCHEMA = {
    "name": "workout_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "disclaimer": {"type": "string"},
            "rest_recovery_note": {"type": "string"},
            "days": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "day_number": {"type": "integer"},
                        "focus": {"type": "string"},
                        "warm_up": {"type": "string"},
                        "cool_down": {"type": "string"},
                        "exercises": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "sets": {"type": "integer"},
                                    "reps": {"type": "string"},
                                    "notes": {"type": "string"},
                                },
                                "required": ["name", "sets", "reps", "notes"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["day_number", "focus", "warm_up", "cool_down", "exercises"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "disclaimer", "rest_recovery_note", "days"],
        "additionalProperties": False,
    },
}


def build_user_prompt(request: WorkoutRequest, regenerate: bool = False) -> str:
    """
    Turn a structured WorkoutRequest into an explicit, labeled prompt for the LLM.

    Fields are listed as labeled lines (not a prose sentence) so the model can't
    blur or drop any single constraint, and the day count is restated as a hard
    numeric requirement since models otherwise default to a generic 7-day split.

    When `regenerate` is True, an extra instruction is appended asking for a
    different variation of the plan (used by the app's "Regenerate" button).
    """
    limitations = request.limitations.strip() if request.limitations else ""
    equipment = ", ".join(request.equipment) if isinstance(request.equipment, list) else request.equipment

    lines = [
        "Create a personalized weekly workout plan for a client with the following profile:",
        "",
        f"- Goal: {request.goal}",
        f"- Experience level: {request.experience}",
        f"- Training days available per week: {request.days_per_week} "
        f"(generate exactly {request.days_per_week} day block(s), no more, no fewer)",
        f"- Equipment access: {equipment}",
        f"- Injuries / limitations: {limitations if limitations else 'None reported'}",
        "",
    ]

    if limitations:
        lines.append(
            "Because limitations were reported, remember to set the disclaimer field, "
            "and actively avoid or substitute any exercise that could aggravate them."
        )
    else:
        lines.append(
            "No limitations were reported — set the disclaimer field to an empty string."
        )

    if regenerate:
        lines.append(
            "Provide a different variation from a typical/default plan for this "
            "profile — vary exercise selection, ordering, or set/rep schemes — "
            "while still fully respecting every constraint above."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# "Swap this exercise" — a narrowly-scoped prompt/schema for replacing a
# single exercise, rather than regenerating the whole plan.
# ---------------------------------------------------------------------------

SWAP_SYSTEM_PROMPT = """You are an experienced, safety-conscious certified personal trainer. The \
user wants ONE alternative exercise to replace a single exercise in an existing workout day — not \
a new plan.

Suggest one different exercise that:
   - Targets the same muscle group / movement pattern as the original exercise, so it still fits \
that day's focus.
   - Uses only the user's available equipment — never assume equipment they don't have.
   - Is appropriate for the user's experience level.
   - Actively avoids aggravating any injuries or limitations the user reported.
   - Is NOT the same exercise as the one being replaced.

Respond with JSON only, matching the required schema exactly: "name" (exercise name), integer \
"sets", "reps" as a string (e.g. "8-10"), and "notes" (a short string — use an empty string if \
there's nothing to note, or briefly explain the substitution if it was made because of a \
limitation). Do not include any text outside the JSON object.
"""

# JSON Schema for a single exercise object — a subset of WORKOUT_PLAN_SCHEMA's
# per-exercise shape, reused here so a swap only asks for what it needs.
EXERCISE_SWAP_SCHEMA = {
    "name": "exercise_swap",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "sets": {"type": "integer"},
            "reps": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["name", "sets", "reps", "notes"],
        "additionalProperties": False,
    },
}


def build_swap_user_prompt(request: WorkoutRequest, day_focus: str, current_exercise: Exercise) -> str:
    """Build the prompt for replacing a single exercise within a workout day."""
    limitations = request.limitations.strip() if request.limitations else ""
    equipment = ", ".join(request.equipment) if isinstance(request.equipment, list) else request.equipment

    lines = [
        "Suggest one alternative exercise to replace the following, for this client:",
        "",
        f"- Day focus: {day_focus}",
        f"- Experience level: {request.experience}",
        f"- Equipment access: {equipment}",
        f"- Injuries / limitations: {limitations if limitations else 'None reported'}",
        "",
        "Exercise to replace:",
        f"- {current_exercise.name} — {current_exercise.sets} sets x {current_exercise.reps} reps",
        "",
        "Give exactly one replacement exercise that still fits this day's focus.",
    ]

    return "\n".join(lines)
