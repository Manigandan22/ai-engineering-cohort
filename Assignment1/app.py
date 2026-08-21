"""
Streamlit UI for the Workout Plan Generator.

This module is intentionally kept free of prompt-building or API-calling
logic — that lives in prompts.py / workout_generator.py — so the UI layer
only deals with collecting input, validating it, and rendering output.
"""

from datetime import datetime

from dotenv import load_dotenv

# Must run before importing workout_generator: MODEL_NAME is read from the
# environment at that module's import time, so .env has to be loaded first.
load_dotenv()

import streamlit as st

from workout_generator import generate_workout_plan, swap_exercise
from workout_models import WorkoutDay, WorkoutPlan, WorkoutRequest

GOAL_OPTIONS = ["Build muscle", "Lose fat", "General fitness", "Improve endurance"]
EXPERIENCE_OPTIONS = ["Beginner", "Intermediate", "Advanced"]
EQUIPMENT_OPTIONS = ["No equipment", "Home dumbbells", "Full gym"]

st.set_page_config(page_title="Workout Plan Generator", page_icon="🏋️", layout="centered")


def _init_session_state() -> None:
    st.session_state.setdefault("plan", None)
    st.session_state.setdefault("last_request", None)
    st.session_state.setdefault("error", None)
    # Which day's expander should stay open across a rerun — reset to Day 1
    # on a fresh generate/regenerate, pinned to the relevant day after a swap.
    st.session_state.setdefault("expanded_day", 1)


def _render_inputs() -> WorkoutRequest:
    st.subheader("Tell us about yourself")

    goal = st.selectbox("Fitness goal", GOAL_OPTIONS)
    experience = st.selectbox("Experience level", EXPERIENCE_OPTIONS)
    days_per_week = st.slider("Days available per week", min_value=1, max_value=7, value=3)
    equipment = st.selectbox("Equipment access", EQUIPMENT_OPTIONS)
    limitations = st.text_area(
        "Injuries or limitations (optional)",
        placeholder='e.g. "bad knees", "no overhead pressing", "lower back pain"',
    )

    return WorkoutRequest(
        goal=goal,
        experience=experience,
        days_per_week=days_per_week,
        equipment=[equipment],
        limitations=limitations.strip() if limitations else None,
    )


def _run_generation(request: WorkoutRequest, regenerate: bool = False) -> None:
    with st.spinner("Generating your plan..."):
        result = generate_workout_plan(request, regenerate=regenerate)

    if result.success:
        st.session_state["plan"] = result.plan
        st.session_state["last_request"] = request
        st.session_state["error"] = None
        st.session_state["expanded_day"] = 1
    else:
        st.session_state["error"] = result.error
        # Keep any previously successful plan visible rather than wiping it out.


def _run_swap(day: WorkoutDay, exercise_index: int) -> None:
    request: WorkoutRequest = st.session_state["last_request"]
    current_exercise = day.exercises[exercise_index]

    with st.spinner(f"Finding an alternative to {current_exercise.name}..."):
        result = swap_exercise(request, day.focus, current_exercise)

    if result.success:
        day.exercises[exercise_index] = result.exercise
        st.session_state["error"] = None
    else:
        st.session_state["error"] = result.error

    # Keep this day's card open so the swap (or the error) is visible.
    st.session_state["expanded_day"] = day.day_number


def _render_summary_bar(request: WorkoutRequest) -> None:
    cols = st.columns(4)
    cols[0].metric("Goal", request.goal)
    cols[1].metric("Level", request.experience)
    cols[2].metric("Days/week", request.days_per_week)
    cols[3].metric("Equipment", ", ".join(request.equipment))


def _render_exercise_row(day: WorkoutDay, ex_index: int) -> None:
    ex = day.exercises[ex_index]
    cols = st.columns([3, 1, 1, 3, 1])
    cols[0].write(ex.name)
    cols[1].write(ex.sets)
    cols[2].write(ex.reps)
    cols[3].write(ex.notes or "")
    if cols[4].button("🔄", key=f"swap_{day.day_number}_{ex_index}", help="Swap this exercise"):
        _run_swap(day, ex_index)
        st.rerun()


def _render_plan(plan: WorkoutPlan) -> None:
    st.subheader("Your Weekly Plan")

    if plan.disclaimer:
        st.warning(f"⚠️ {plan.disclaimer}")

    if plan.summary:
        st.caption(plan.summary)

    for day in plan.days:
        expanded = day.day_number == st.session_state["expanded_day"]
        with st.expander(f"📅 Day {day.day_number}: {day.focus}", expanded=expanded):
            if day.warm_up:
                st.markdown(f"**🔥 Warm-up:** {day.warm_up}")

            header = st.columns([3, 1, 1, 3, 1])
            for col, label in zip(header, ["**Exercise**", "**Sets**", "**Reps**", "**Notes**", ""]):
                col.markdown(label)

            for ex_index in range(len(day.exercises)):
                _render_exercise_row(day, ex_index)

            if day.cool_down:
                st.markdown(f"**🧊 Cooldown:** {day.cool_down}")

    if plan.rest_recovery_note:
        st.info(f"💤 **Rest & Recovery:** {plan.rest_recovery_note}")


def _render_output() -> None:
    if st.session_state["error"]:
        st.error(st.session_state["error"])

    plan: WorkoutPlan = st.session_state["plan"]
    if plan:
        _render_plan(plan)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔁 Regenerate (different variation)"):
                _run_generation(st.session_state["last_request"], regenerate=True)
                st.rerun()
        with col2:
            filename = f"workout_plan_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
            st.download_button(
                "⬇️ Download plan (.md)",
                data=plan.to_markdown(),
                file_name=filename,
                mime="text/markdown",
            )


def main() -> None:
    _init_session_state()

    st.title("🏋️ Workout Plan Generator")
    st.caption(
        "Answer a few questions and get a personalized weekly workout plan, "
        "generated by an LLM via the Groq API."
    )

    request = _render_inputs()

    if st.button("Generate Plan", type="primary"):
        _run_generation(request, regenerate=False)

    if st.session_state["plan"]:
        st.divider()
        _render_summary_bar(st.session_state["last_request"])

    _render_output()


if __name__ == "__main__":
    main()
