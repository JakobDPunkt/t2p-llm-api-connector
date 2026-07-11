# Direct-PNML prompts — few-shot example curation

This note is for maintainers of `01_pnml_json_system_prompt.txt` (the JSON
direct-PNML path). It is **not** part of any prompt and is never sent to a model.

The examples are chosen so that together they teach every control-flow pattern
the serializer depends on, with as little overlap as possible. Each example has
**one** primary job. If you add, remove, or rewrite one, keep the coverage below
intact — a gap here shows up directly as a class of malformed nets, because the
model falls back to improvising the pattern it was never shown.

## Why examples over rules

Rules are stated once in the prompt and in `_pnml_semantics.txt`; weak models
(e.g. the gpt-5.4-nano tier) demonstrably ignore prose rules under load. A
concrete worked example is the more reliable teacher, so a control-flow pattern
that keeps going wrong earns an example, not just a sentence.

## Coverage (one row per example, in prompt order)

1. **"A customer reports an issue…"** (English)
   Primary: **repetition / back-edge** — a loop rejoins at the place its repeated
   part starts from. Secondary: exclusive split at a place; nameless intermediate
   places.

2. **"Ein Kunde bestellt eine Ware…"** (German)
   Primary: an **exclusive choice that rejoins at a shared place and then
   continues with steps modelled once**. This is the pattern a weak model
   otherwise fakes by duplicating the whole tail after the choice — the same
   activity emitted twice under different ids (same name), each copy ending in
   its own end place. Added after exactly that failure was observed on a long
   process (ATM). Secondary: the description is German, so this is also the
   example that keeps non-English labels honest.

3. **"A support agent receives a ticket…"** (English)
   Primary: an **optional step (do-it-or-skip)** modelled as an exclusive choice
   where one branch is the step and the other is a **nameless routing transition**
   that skips it, both rejoining **at a place** before the flow continues. Added
   after models modelled an optional step by pointing both the do-branch and the
   skip straight into the next activity's transition, which makes an AND-join and
   **deadlocks** the net. Also reinforces the silent routing transition.

4. **"Simultaneously, the goods are packed…"** (English)
   Primary: **parallel split and join at transitions**, plus a **nameless routing
   transition** (`t_split`) for parallel work that starts with no preceding
   activity.

## Deliberately NOT given their own example

- **Several distinct final transitions sharing one end place** (e.g. approve →
  end, reject → end). Covered by the single-end discipline in example 2 and by
  the rule text ("…share the one end place"). Removed with the old Urlaubsantrag
  example, whose parallel content already lived in example 3. Add a dedicated
  example only if separate end places for distinct outcomes start reappearing.
- **A parallel split on an activity transition** (rather than a routing `t_split`).
  The semantics text covers both; example 3 shows the routing case, which is the
  one models get wrong more often.

## Guardrail

`tests/test_pnml_json.py::TestSharedPromptSemantics::test_the_json_examples_serialize_to_valid_nets`
parses every JSON object out of this prompt and asserts it serializes to a net the
validator accepts. Adding or removing an example means updating the expected
object count there (currently 5 = the target format object plus four examples).
