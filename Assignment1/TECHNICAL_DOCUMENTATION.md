# Technical Documentation — Workout Plan Generator

This document covers the architecture, data flow, module reference, and —
most importantly — how to tweak the LLM's behavior (prompt, schema, model,
sampling) without breaking the rest of the app. For install/run steps, see
[README.md](README.md).

---

## 1. Architecture overview

```
┌─────────────┐      WorkoutRequest       ┌──────────────────────┐
│   app.py    │ ────────────────────────► │ workout_generator.py │
│ (Streamlit  │                            │  validate_request()  │
│    UI)      │ ◄──────────────────────── │  generate_workout_   │
└─────────────┘      GenerationResult      │       plan()          │
       │                                    └──────────┬───────────┘
       │                                               │ build_user_prompt()
       │                                               ▼
       │                                    ┌──────────────────────┐
       │                                    │      prompts.py       │
       │                                    │  SYSTEM_PROMPT         │
       │                                    │  WORKOUT_PLAN_SCHEMA    │
       │                                    └──────────┬───────────┘
       │                                               │ messages + response_format
       │                                               ▼
       │                                    ┌──────────────────────┐
       │                                    │     Groq API           │
       │                                    │ (openai/gpt-oss-120b)   │
       │                                    └──────────┬───────────┘
       │                                               │ JSON string
       │                                               ▼
       │                                    ┌──────────────────────┐
       │                                    │  _parse_plan() in       │
       │                                    │  workout_generator.py   │
       │                                    └──────────┬───────────┘
       │                                               │ WorkoutPlan
       ▼                                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       workout_models.py                          │
│   WorkoutRequest · Exercise · WorkoutDay · WorkoutPlan ·          │
│   GenerationResult                                                │
└─────────────────────────────────────────────────────────────────┘
```

**Design principle:** `app.py` never talks to Groq directly and never sees a
raw exception — it only ever gets back a `GenerationResult(success, plan,
error)`. This keeps all LLM-specific logic (prompt text, schema, retries,
exception handling) in one place (`workout_generator.py` + `prompts.py`),
and keeps the UI layer trivial to read.

---

## 2. File-by-file reference

### `workout_models.py` — typed data structures

| Type | Fields | Purpose |
|---|---|---|
| `WorkoutRequest` | `goal, experience, days_per_week, equipment, limitations` | The form input, built once per generate/regenerate click |
| `Exercise` | `name, sets, reps, notes` | One row in a day's exercise table |
| `WorkoutDay` | `day_number, focus, warm_up, exercises, cool_down` | One `st.expander` card |
| `WorkoutPlan` | `summary, days, disclaimer, rest_recovery_note` | The full parsed plan; has `.to_markdown()` for file export |
| `GenerationResult` | `success, plan, error` | What `generate_workout_plan()` always returns |

`WorkoutPlan.to_markdown()` reconstructs a Markdown document from the typed
data — this is what powers the **Download plan (.md)** button. It is *not*
the same string the LLM produced; the LLM never produces Markdown anymore
(see §3).

### `prompts.py` — prompt templates and output schema

- `SYSTEM_PROMPT` — fixed instructions (role, constraints, output contract, safety scope, tone). Loaded once per call as the `system` message.
- `WORKOUT_PLAN_SCHEMA` — a JSON Schema dict passed to Groq as `response_format`. This is what actually enforces structure — see §3.
- `build_user_prompt(request, regenerate=False) -> str` — turns a `WorkoutRequest` into labeled key/value lines (not a prose sentence) plus explicit restatement of the day count and disclaimer requirement. Appends a "give a different variation" instruction when `regenerate=True`.

### `workout_generator.py` — validation + Groq call + parsing

- `validate_request(request) -> Optional[str]` — pure function, no network call. Returns an error string or `None`.
- `_get_client() -> groq.Groq` — reads `GROQ_API_KEY` from the environment; raises `RuntimeError` if absent (caught by the caller).
- `_parse_plan(raw_json) -> Optional[WorkoutPlan]` — `json.loads` + defensive field extraction; returns `None` on any parse/shape problem rather than raising.
- `_attempt_generation(client, request, regenerate) -> (GenerationResult, retryable: bool)` — makes exactly one Groq call and classifies the outcome. `retryable=True` only for empty/malformed responses (see §5).
- `generate_workout_plan(request, regenerate=False) -> GenerationResult` — the public entry point. Validates → gets client → calls `_attempt_generation` → retries once if `retryable` → returns.

Module-level constants you'll tweak most often:
```python
MODEL_NAME = "openai/gpt-oss-120b"
MAX_TOKENS = 2000
TEMPERATURE = 0.7
```

### `app.py` — Streamlit UI

- `_render_inputs() -> WorkoutRequest` — the form (dropdowns, slider, text area).
- `_run_generation(request, regenerate)` — calls `generate_workout_plan`, writes the result into `st.session_state`.
- `_render_summary_bar(request)` — the 4-column metrics strip (Goal / Level / Days / Equipment).
- `_render_plan(plan)` — renders the disclaimer as `st.warning`, each day as an `st.expander` with an `st.table` of exercises, and the rest/recovery note as `st.info`.
- `_render_output()` — renders any error (`st.error`) plus the plan (if present), and the Regenerate/Download buttons.

`st.session_state` keys: `plan` (`WorkoutPlan | None`), `last_request` (`WorkoutRequest | None`), `error` (`str | None`). A failed generation does **not** clear a previously successful `plan` — the old plan stays visible with the new error shown above it.

---

## 3. Why JSON schema instead of parsed Markdown

Earlier versions of this app asked the LLM to return Markdown (`## Day 1: ...`)
and rendered it as-is with `st.markdown()`. That's fragile: nothing stops the
model from changing heading levels, forgetting a day, or writing prose instead
of a list.

Instead, the app uses Groq's **structured output mode**:

```python
response_format={"type": "json_schema", "json_schema": WORKOUT_PLAN_SCHEMA}
```

`WORKOUT_PLAN_SCHEMA` (defined in `prompts.py`) declares `"strict": True` and
`"additionalProperties": False` at every object level. Groq validates the
model's generation against this schema **server-side** before returning it.
If the generation doesn't comply, the API itself returns an HTTP 400 with
`code: "json_validate_failed"` instead of handing back bad JSON — this is
what `_attempt_generation` treats as retryable (see §5).

This only works because `openai/gpt-oss-120b` supports Groq's strict
JSON-schema mode. If you change `MODEL_NAME` to a model that doesn't support
it, either switch `response_format` to best-effort JSON object mode
(`{"type": "json_object"}`, less reliable — validate app-side) or check
Groq's docs for that model's structured-output support before relying on it.

---

## 4. How to tweak the LLM

All of the following live in `prompts.py` and `workout_generator.py` — no UI
changes needed for any of these.

### 4.1 Change the model

The model is read from the `GROQ_MODEL` environment variable, falling back
to a hardcoded default if unset:
```python
# workout_generator.py
DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
MODEL_NAME = os.environ.get("GROQ_MODEL", DEFAULT_MODEL_NAME)
```
To switch models without touching code, set `GROQ_MODEL` in `.env` (see
`.env.example`). Check current model IDs at
https://console.groq.com/docs/models — Groq periodically deprecates models
(this happened once already during this project; see the 404 handling in
§5). If you switch models, re-verify JSON schema support, since not all
Groq-hosted models support `strict: true`.

⚠️ **Import-order caveat:** `MODEL_NAME` is resolved once, at the moment
`workout_generator` is first imported — not lazily per-request like
`GROQ_API_KEY` is. `app.py` therefore calls `load_dotenv()` **before**
importing `workout_generator`, so `.env` values are already in
`os.environ` by the time `MODEL_NAME` is read. If you ever reorder those
imports, `GROQ_MODEL` from `.env` will silently stop being picked up.

### 4.2 Adjust creativity / length

```python
MAX_TOKENS = 2000     # raise if plans get cut off for 6-7 day requests
TEMPERATURE = 0.7      # lower = more consistent/conservative, higher = more varied
```
`_attempt_generation` already bumps temperature by `+0.15` (capped at `1.0`)
automatically when `regenerate=True`, so the "different variation" button has
something to work with beyond the prompt instruction alone.

### 4.3 Change what the model is told to do — `SYSTEM_PROMPT`

This is the primary lever for output quality. It's organized into four
numbered rule blocks in `prompts.py`:

1. **Respect constraints exactly** — equipment, exact day count, experience-appropriate difficulty, injury avoidance.
2. **Output format** — the field-by-field description of the JSON shape (kept in sync with `WORKOUT_PLAN_SCHEMA`).
3. **Scope and safety** — no medical claims, exact disclaimer wording rule.
4. **Tone** — concise, no filler.

To change behavior, edit the relevant numbered block directly rather than
appending unrelated instructions at the end — the model tends to weight
rules by clarity/proximity to related rules, not just presence.

**Example — make workouts more time-boxed:**
```python
# Add to rule block 1 in SYSTEM_PROMPT:
"   - Assume each session should take 45-60 minutes; keep exercise count per day realistic for that window."
```

**Example — add a progression scheme:**
This requires both a prompt change *and* a schema change (§4.4), since
"which week" isn't currently a field the model can express.

### 4.4 Change the output shape — `WORKOUT_PLAN_SCHEMA` + parsing + models

If you want new structured fields (e.g. an `equipment_used` tag per
exercise, a `difficulty` field, a 4-week progression), you must update
**three places together** or parsing will silently drop the new data:

1. **`prompts.py` → `WORKOUT_PLAN_SCHEMA`** — add the field to the relevant
   object's `"properties"` and to its `"required"` list (strict mode
   requires every declared property to be listed as required — Groq's
   strict mode does not support true-optional properties; use an empty
   string/sentinel value convention instead, exactly as `notes` and
   `rest_recovery_note` already do).
2. **`prompts.py` → `SYSTEM_PROMPT`** rule block 2 — describe the new field
   in plain English so the model knows what to put in it.
3. **`workout_models.py`** — add the field to `Exercise` / `WorkoutDay` /
   `WorkoutPlan` as appropriate, and update `WorkoutPlan.to_markdown()` if it
   should appear in the downloaded file.
4. **`workout_generator.py` → `_parse_plan()`** — extract the new field from
   the parsed JSON (`day.get(...)` / `ex.get(...)`) into the dataclass
   constructor call.

Then update `app.py`'s `_render_plan()` if the new field should be visible
in the UI (e.g. a `st.caption` under each exercise row for `difficulty`).

### 4.5 Change the user-facing prompt content — `build_user_prompt()`

This function controls exactly what gets sent per-request. Two patterns
already used here that are worth keeping if you extend it:

- **Labeled lines, not prose.** `f"- Goal: {request.goal}"` rather than
  blending fields into a sentence — this is what makes the constraints hard
  for the model to blur together or drop.
- **Restate hard numeric constraints explicitly.** The day count is stated
  twice conceptually (as a fact, and as an instruction — *"generate exactly
  N day block(s), no more, no fewer"*) because models default to a generic
  7-day split otherwise.

If you add a new structured input field to the Streamlit form, add its
corresponding labeled line here, and mention it in `SYSTEM_PROMPT` rule
block 1 if it represents a constraint the model must respect.

### 4.6 Iterating and testing prompt changes

There's no automated eval suite here (out of scope for this assignment) —
iteration is manual. Re-run this input matrix after any `SYSTEM_PROMPT` or
schema change and read the actual output, not just check `success=True`:

| Scenario | What to check |
|---|---|
| 1 day, Full gym, Advanced | Does it avoid cramming an unrealistic number of exercises into one day? |
| 7 days, No equipment | Does it avoid overloading the same muscle group across days? Is it genuinely bodyweight-only? |
| Beginner + "bad knees" | Disclaimer present? Are knee-heavy movements (lunges, jump squats, deep goblet squats) actually avoided/substituted, or just mentioned in passing? |
| No equipment + Build muscle | Does it correctly pivot to progressive-overload bodyweight techniques instead of hallucinating dumbbells? |
| No limitations text | Disclaimer correctly absent — no boilerplate leaking in |

A convenient way to run this matrix without clicking through the UI each
time:
```python
from dotenv import load_dotenv; load_dotenv()
from workout_models import WorkoutRequest
from workout_generator import generate_workout_plan

r = WorkoutRequest(goal="Build muscle", experience="Beginner",
                    days_per_week=7, equipment=["No equipment"], limitations=None)
result = generate_workout_plan(r)
print(result.plan.to_markdown() if result.success else result.error)
```

---

## 5. Error handling reference

| Failure | Where it's caught | Retried automatically? | User sees |
|---|---|---|---|
| Invalid/missing form input (e.g. 0 days) | `validate_request()`, before any API call | No — no call made | Friendly `st.error`, e.g. "Please choose a number of training days between 1 and 7." |
| `GROQ_API_KEY` not set | `_get_client()` → `RuntimeError` | No | "The app isn't configured with a Groq API key..." |
| Bad/revoked API key | `groq.AuthenticationError` | No | "The Groq API key was rejected..." |
| Rate limit hit | `groq.RateLimitError` | No | "Groq's rate limit was hit..." |
| Request timeout | `groq.APITimeoutError` | No | "The request to Groq timed out..." |
| Network/connection failure | `groq.APIConnectionError` | No | "Couldn't reach the Groq API..." |
| Model deprecated / wrong model ID (404) | `groq.APIStatusError`, `status_code == 404` | No | Points to `console.groq.com/docs/models` and `MODEL_NAME` |
| Schema-invalid generation (400, `json_validate_failed`) | `groq.APIStatusError`, `status_code == 400` | **Yes, once** | Only shown if the retry also fails |
| Empty response content | after the call, before parsing | **Yes, once** | Only shown if the retry also fails |
| Unparseable/malformed JSON | `_parse_plan()` returns `None` | **Yes, once** | Only shown if the retry also fails |
| Any other unexpected exception | bare `except Exception` in `_attempt_generation` | No | "Something unexpected went wrong..." |

Key invariant: **no raw exception or stack trace ever reaches the Streamlit
UI.** Every path above terminates in a `GenerationResult(success=False,
error=<string>)`, and `app.py` only ever calls `st.error(...)` on that string.

The 400/empty/malformed cases retry because they were observed live during
development to be transient model-generation glitches (see git history /
conversation — Groq occasionally emits a stray `""` in place of an exercise
object even under strict schema mode); everything else (auth, rate limit,
network, wrong model) is a systemic condition a retry won't fix.

---

## 6. Extending the app (stretch-goal notes)

Already implemented: session-state persistence, Regenerate, Markdown download.

**"Swap this exercise" (not yet implemented)** — sketch of the approach if
you build it: add a small "🔄 Swap" button next to each exercise row inside
the `st.table`/loop in `_render_plan()`. On click, call a new
`workout_generator.regenerate_exercise(day, exercise, request) -> Exercise`
that sends a narrowly-scoped prompt ("suggest one alternative exercise for
X, same muscle group, same equipment/limitation constraints") and returns a
single `Exercise`, not a full plan — this avoids the cost and risk of
regenerating the whole week to change one line. You'd need a schema for a
single exercise object (a small subset of `WORKOUT_PLAN_SCHEMA`) and to
mutate the specific `Exercise` in `st.session_state["plan"]` in place before
`st.rerun()`.

---

## 7. Known limitations

- `_parse_plan()` requires every day to have at least one exercise; a plan
  where the model legitimately wants a pure-rest "day" would currently be
  rejected. Not observed in testing since the schema/prompt already treat
  rest days via `rest_recovery_note`, not as `WorkoutDay` entries.
- If Groq returns exactly `request.days_per_week` days is not schema-enforced
  (JSON Schema doesn't easily express "array length must equal a value from
  another field"), so `generate_workout_plan` accepts a mismatched day count
  and appends a note to the summary rather than treating it as an error —
  a design choice to avoid failing a plan that's otherwise usable.
- No automated test suite / CI — verification so far has been manual
  scripted smoke tests (see §4.6) run against the live Groq API.
