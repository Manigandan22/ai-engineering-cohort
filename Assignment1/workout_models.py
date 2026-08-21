"""
Typed data structures shared between the UI layer and the generation logic.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WorkoutRequest:
    """Structured user input collected from the Streamlit form."""

    goal: str
    experience: str
    days_per_week: int
    equipment: List[str]
    limitations: Optional[str] = None


@dataclass
class Exercise:
    """A single exercise line within a workout day."""

    name: str
    sets: int
    reps: str
    notes: Optional[str] = None


@dataclass
class WorkoutDay:
    """One training day within the plan."""

    day_number: int
    focus: str
    warm_up: str
    exercises: List[Exercise] = field(default_factory=list)
    cool_down: str = ""


@dataclass
class WorkoutPlan:
    """
    The full structured plan returned by the LLM, parsed from its JSON
    response so the UI can render real tables/cards instead of raw text.
    """

    summary: str
    days: List[WorkoutDay] = field(default_factory=list)
    disclaimer: Optional[str] = None
    rest_recovery_note: Optional[str] = None

    def to_markdown(self) -> str:
        """Render the structured plan back to Markdown, e.g. for file download."""
        lines: List[str] = []

        if self.disclaimer:
            lines.append(f"_{self.disclaimer}_")
            lines.append("")

        lines.append(self.summary)
        lines.append("")

        for day in self.days:
            lines.append(f"## Day {day.day_number}: {day.focus}")
            if day.warm_up:
                lines.append(f"- **Warm-up:** {day.warm_up}")
            for ex in day.exercises:
                notes = f" ({ex.notes})" if ex.notes else ""
                lines.append(f"- {ex.name} — {ex.sets} sets x {ex.reps} reps{notes}")
            if day.cool_down:
                lines.append(f"- **Cooldown:** {day.cool_down}")
            lines.append("")

        if self.rest_recovery_note:
            lines.append(f"**Rest & Recovery:** {self.rest_recovery_note}")

        return "\n".join(lines)


@dataclass
class GenerationResult:
    """
    Outcome of a call to generate_workout_plan.

    Using a typed result object (instead of raising exceptions across the
    UI/logic boundary) keeps app.py free of try/except blocks — every failure
    path is represented explicitly and rendered the same way.
    """

    success: bool
    plan: Optional[WorkoutPlan] = None
    error: Optional[str] = None
