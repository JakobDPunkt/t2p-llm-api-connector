import json
import logging
import re
import time
from collections import namedtuple

import google.generativeai as genai
from flask import current_app
from openai import OpenAI

from app.services import model_registry
from app.services.model_validator import ModelValidator
from app.services.pnml_validator import PnmlValidator
from app.utils.prompt_builder import PromptBuilder, STRICT_JSON_REMINDER

#: One pass of the direct-PNML correction loop, after normalization.
PnmlAttempt = namedtuple("PnmlAttempt", "pnml issues stripped counts")


class PnmlGeneration(namedtuple("PnmlGeneration", "attempts best_index")):
    """The full history of a direct-PNML generation.

    ``attempts[0]`` is the model's unaided first shot, every further entry one
    correction pass. Deriving the summary numbers from the raw history keeps
    them consistent and costs no extra bookkeeping: the number of correction
    passes is ``len(attempts) - 1``, the model's unaided quality is
    ``attempts[0].issues``, and a correction that traded net structure for
    validity shows up as shrinking ``counts``.
    """

    @property
    def best(self):
        """The delivered attempt: the one with the fewest remaining issues."""
        return self.attempts[self.best_index]


_PNML_BLOCK_RE = re.compile(r"<pnml\b.*?</pnml>", re.DOTALL | re.IGNORECASE)

# Correction budget for the direct-PNML path: one generation plus up to
# three correction passes (all at temperature 0; the prompt changes between
# passes because it carries the previous document and its issues).
_PNML_MAX_CORRECTIONS = 3

# Output budget for the direct-PNML path only. The /generate defaults are too
# small here: a full PNML document is far longer than the standard path's BPMN
# JSON, and on GPT-5 variants reasoning tokens draw from the same budget (a
# 4096 cap was fully consumed by reasoning, yielding an empty reply). Gemini
# is capped at 8192, the hard output limit of gemini-2.0-flash.
_PNML_OPENAI_MAX_COMPLETION_TOKENS = 16384
_PNML_GEMINI_MAX_OUTPUT_TOKENS = 8192

logger = logging.getLogger(__name__)


class EmptyResponseError(ValueError):
    """Raised when the provider returns no usable completion text.

    ``raw_reply`` carries the provider's actual answer when it was non-empty
    but held no PNML document, so callers can surface what the model wrote
    instead of a bare "no PNML" message.
    """

    def __init__(self, *args, raw_reply=None):
        super().__init__(*args)
        self.raw_reply = raw_reply


class TruncatedResponseError(ValueError):
    """Raised when the provider stopped at its output token limit.

    The reply is cut off and cannot hold a complete PNML document, so it is
    reported as truncation rather than as a generic "no PNML" failure.
    ``raw_reply`` carries the partial answer for diagnostics.
    """

    def __init__(self, *args, raw_reply=None):
        super().__init__(*args)
        self.raw_reply = raw_reply


def _is_truncation_reason(finish_reason):
    """Whether a provider finish reason signals an output-token-limit cutoff.

    Covers OpenAI's ``length`` and Gemini's ``MAX_TOKENS``, tolerating both
    plain strings and enum values (compared by name).
    """
    name = getattr(finish_reason, "name", None) or str(finish_reason or "")
    return name.upper() in {"LENGTH", "MAX_TOKENS"}


class LLMService:
    """Service class for handling LLM API calls"""

    def __init__(self):
        self.prompt_builder = PromptBuilder()
        self.model_validator = ModelValidator()
        self.pnml_validator = PnmlValidator()

    @staticmethod
    def _config_value(name, default=None):
        try:
            return current_app.config.get(name, default)
        except RuntimeError:
            return default

    @staticmethod
    def _extract_json_object(text):
        """Extract and parse a JSON object from model output text."""
        content = (text or "").strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model output does not contain a JSON object.")
        return json.loads(content[start:end + 1])

    @staticmethod
    def _extract_pnml_document(text):
        """Extract the PNML XML document from model output text.

        Counterpart of ``_extract_json_object`` for the direct-PNML path:
        replies may wrap the XML in markdown fences or surrounding prose
        despite the system prompt forbidding it. Returns the
        ``<pnml>...</pnml>`` block with an XML declaration, or ``None`` if
        the reply contains none.
        """
        if not text:
            return None
        match = _PNML_BLOCK_RE.search(text)
        if match is None:
            return None
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{match.group(0)}'

    @staticmethod
    def _merge_known_elements(partials):
        """Merge step outputs into known elements map by preserving first-seen IDs."""
        merged = {"events": [], "tasks": [], "gateways": []}
        seen = {"events": set(), "tasks": set(), "gateways": set()}

        for part in partials:
            if not isinstance(part, dict):
                continue
            for key in ("events", "tasks", "gateways"):
                for item in part.get(key, []):
                    if not isinstance(item, dict):
                        continue
                    item_id = item.get("id")
                    if not item_id or item_id in seen[key]:
                        continue
                    seen[key].add(item_id)
                    merged[key].append(item)
        return merged

    def _build_repair_prompt(self, user_text, model_json, issues):
        """Create a repair prompt from validation findings."""
        issues_block = "\n".join(f"- {issue}" for issue in issues)
        model_block = json.dumps(model_json, ensure_ascii=False, indent=2)
        return (
            "You are repairing a BPMN JSON model to satisfy strict structural rules.\n"
            f"{STRICT_JSON_REMINDER}\n\n"
            "Preserve IDs where possible, but fix broken structure.\n\n"
            "Process text:\n"
            f"{user_text}\n\n"
            "Validation issues to fix:\n"
            f"{issues_block}\n\n"
            "Current model:\n"
            f"{model_block}\n"
        )

    def _build_pnml_repair_prompt(self, user_text, pnml_xml, issues):
        """Create a correction prompt from PNML validation findings.

        Counterpart of ``_build_repair_prompt`` for the direct-PNML path.
        The prompt anchors the correction in the original process text (so
        the model does not drift away from the described process just to
        satisfy the validator) and demands a minimal change (so correct
        parts survive the correction).
        """
        issues_block = "\n".join(f"- {issue}" for issue in issues)
        return (
            "Your previous PNML document for the process description below "
            "violates structural rules.\n"
            "Produce a corrected version of THIS document. Change as little "
            "as possible: fix exactly the listed issues and keep all correct "
            "parts (ids, names, structure) unchanged. The corrected document "
            "must still model the described process.\n"
            "Return exactly one well-formed PNML XML document and nothing "
            "else.\n\n"
            "Process description:\n"
            f"{user_text}\n\n"
            "Validation issues to fix:\n"
            f"{issues_block}\n\n"
            "Your previous PNML:\n"
            f"{pnml_xml}\n"
        )

    @staticmethod
    def _extract_openai_message_text(message):
        """Normalize OpenAI message content into a plain string.

        Some SDK/model combinations return ``message.content`` as a list of
        typed content parts rather than a single string.
        """
        if message is None:
            return ""

        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    continue

                if getattr(part, "type", None) == "text":
                    text_parts.append(getattr(part, "text", ""))

            return "".join(text_parts).strip()

        return ""

    def _has_json_object(self, text):
        """Return True when text contains a parseable JSON object."""
        try:
            self._extract_json_object(text)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _run_few_shot_orchestration(self, user_text, generate_once):
        """Run real multi-call few-shot extraction and merge with validation."""
        pack = self.prompt_builder.few_shot_prompt_pack
        shared = pack.get("00_shared_rules.txt", "")

        required_files = [
            "01_start_event_prompt.txt",
            "02_tasks_prompt.txt",
            "03_gateways_prompt.txt",
            "04_flows_prompt.txt",
            "05_end_event_prompt.txt",
            "06_merge_and_validate_prompt.txt",
        ]
        for file_name in required_files:
            if not pack.get(file_name):
                raise ValueError(f"Missing few-shot prompt file: {file_name}")

        def compose_prompt(step_text, known_elements=None, partial_outputs=None):
            body = step_text.replace(
                "{{PROCESS_TEXT}}",
                "Summarized process context (derived from full input):\n"
                f"{process_context}",
            )
            if "{{KNOWN_ELEMENTS_JSON}}" in body:
                body = body.replace(
                    "{{KNOWN_ELEMENTS_JSON}}",
                    json.dumps(known_elements or {}, ensure_ascii=False, indent=2),
                )
            if "{{PARTIAL_OUTPUTS_JSON}}" in body:
                body = body.replace(
                    "{{PARTIAL_OUTPUTS_JSON}}",
                    json.dumps(partial_outputs or {}, ensure_ascii=False, indent=2),
                )
            return (
                f"{shared}\n\n{body}\n\n"
                f"{STRICT_JSON_REMINDER}\n"
                "Return only JSON matching the step schema."
            ).strip()

        def run_json_step(step_name, prompt):
            response_text = generate_once(prompt)
            try:
                return self._extract_json_object(response_text)
            except ValueError as first_error:
                logger.warning(
                    "few-shot step '%s' returned non-JSON output; retrying with strict JSON reminder",
                    step_name,
                )
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"{STRICT_JSON_REMINDER} "
                    "Respond again with exactly one JSON object matching the requested schema."
                )
                retry_response_text = generate_once(retry_prompt)
                try:
                    return self._extract_json_object(retry_response_text)
                except ValueError:
                    raise ValueError(
                        f"Few-shot step '{step_name}' did not return a JSON object after retry."
                    ) from first_error

        # Build a compact context once from the raw process text so follow-up
        # steps do not need the full source text again.
        context_summary_prompt = (
            "You are preparing context for a multi-step BPMN extraction pipeline.\n"
            f"{shared}\n\n"
            f"{STRICT_JSON_REMINDER}\n"
            "Return exactly one JSON object with this schema:\n"
            '{"process_context": "..."}\n\n'
            "The value of process_context must be a concise but complete summary "
            "of the process text (actors, activities, decisions, loops, end states). "
            "Preserve factual order and key constraints.\n\n"
            "PROCESS TEXT:\n"
            f"{user_text}"
        )

        context_obj = run_json_step("context_summary", context_summary_prompt)
        process_context = (context_obj.get("process_context") or "").strip()
        if not process_context:
            raise ValueError("few-shot context_summary returned empty process_context")

        partials = {}

        try:
            start_obj = run_json_step(
                "start_event", compose_prompt(pack["01_start_event_prompt.txt"])
            )
        except ValueError as start_error:
            logger.warning(
                "few-shot start_event step failed (%s); falling back to default start event",
                start_error,
            )
            start_obj = {
                "events": [
                    {
                        "id": "startEvent1",
                        "type": "startEvent",
                        "name": "Start",
                    }
                ]
            }
        partials["start"] = start_obj

        try:
            tasks_obj = run_json_step(
                "tasks", compose_prompt(pack["02_tasks_prompt.txt"])
            )
        except ValueError as tasks_error:
            logger.warning(
                "few-shot tasks step failed (%s); falling back to empty tasks list",
                tasks_error,
            )
            tasks_obj = {"tasks": []}
        partials["tasks"] = tasks_obj

        try:
            gateways_obj = run_json_step(
                "gateways", compose_prompt(pack["03_gateways_prompt.txt"])
            )
        except ValueError as gateways_error:
            logger.warning(
                "few-shot gateways step failed (%s); falling back to empty gateways list",
                gateways_error,
            )
            gateways_obj = {"gateways": []}
        partials["gateways"] = gateways_obj

        try:
            end_obj = run_json_step(
                "end_event", compose_prompt(pack["05_end_event_prompt.txt"])
            )
        except ValueError as end_error:
            logger.warning(
                "few-shot end_event step failed (%s); falling back to default end event",
                end_error,
            )
            end_obj = {
                "events": [
                    {
                        "id": "endEvent1",
                        "type": "endEvent",
                        "name": "End",
                    }
                ]
            }
        partials["end"] = end_obj

        known_elements = self._merge_known_elements(
            [start_obj, tasks_obj, gateways_obj, end_obj]
        )
        try:
            flows_obj = run_json_step(
                "flows",
                compose_prompt(
                    pack["04_flows_prompt.txt"], known_elements=known_elements
                ),
            )
        except ValueError as flows_error:
            logger.warning(
                "few-shot flows step failed (%s); falling back to empty flows list",
                flows_error,
            )
            flows_obj = {"flows": []}
        partials["flows"] = flows_obj

        merge_input = {
            "events": known_elements.get("events", []),
            "tasks": known_elements.get("tasks", []),
            "gateways": known_elements.get("gateways", []),
            "flows": flows_obj.get("flows", []),
            "partials": partials,
        }
        try:
            merged_obj = run_json_step(
                "merge_and_validate",
                compose_prompt(
                    pack["06_merge_and_validate_prompt.txt"],
                    partial_outputs=merge_input,
                ),
            )
        except ValueError as merge_error:
            logger.warning(
                "few-shot merge step failed (%s); falling back to deterministic local merge",
                merge_error,
            )
            merged_obj = {
                "events": list(known_elements.get("events", [])),
                "tasks": list(known_elements.get("tasks", [])),
                "gateways": list(known_elements.get("gateways", [])),
                "flows": list(flows_obj.get("flows", [])),
            }

        sanitized = self.model_validator.sanitize_model(merged_obj)
        issues = self.model_validator.validate_model(sanitized, user_text)

        if issues:
            logger.warning(
                "few-shot validation found %d issue(s), running repair pass",
                len(issues),
            )
            repair_prompt = self._build_repair_prompt(user_text, sanitized, issues)
            try:
                repaired_obj = run_json_step("repair", repair_prompt)
                sanitized = self.model_validator.sanitize_model(repaired_obj)
                remaining_issues = self.model_validator.validate_model(
                    sanitized, user_text
                )
                if remaining_issues:
                    raise ValueError(
                        "Few-shot repair produced invalid model: "
                        + "; ".join(remaining_issues)
                    )
            except ValueError as repair_error:
                logger.warning(
                    "few-shot repair step failed (%s); returning pre-repair sanitized model",
                    repair_error,
                )

        return json.dumps(sanitized, ensure_ascii=False)

    @staticmethod
    def _openai_generate_once(
        client,
        system_prompt,
        model,
        prompt,
        max_completion_tokens=4096,
        reasoning_effort=None,
    ):
        model_name = (model or "").lower()
        request_kwargs = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "model": model,
            "max_completion_tokens": max_completion_tokens,
        }
        if reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = reasoning_effort
        # GPT-5 variants can reject explicit temperature values and only accept
        # provider defaults. Avoid first-attempt 400s by omitting it up front.
        if not model_name.startswith("gpt-5"):
            request_kwargs["temperature"] = 0
        try:
            chat_completion = client.chat.completions.create(**request_kwargs)
        except Exception as e:
            # Some OpenAI models (for example GPT-5 variants) only accept the
            # default temperature and reject an explicit value.
            error_text = str(e).lower()
            if "temperature" in error_text and "unsupported" in error_text:
                logger.info(
                    "Retrying OpenAI call without explicit temperature (model=%s)",
                    model,
                )
                request_kwargs.pop("temperature", None)
                chat_completion = client.chat.completions.create(**request_kwargs)
            else:
                raise

        first_choice = chat_completion.choices[0] if chat_completion.choices else None
        message = getattr(first_choice, "message", None)
        content = LLMService._extract_openai_message_text(message)

        finish_reason = getattr(first_choice, "finish_reason", None)
        refusal = getattr(message, "refusal", None)
        usage = getattr(chat_completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)

        logger.debug(
            "OpenAI completion metadata (model=%s, finish_reason=%s, prompt_tokens=%s, completion_tokens=%s, content_len=%d)",
            getattr(chat_completion, "model", model),
            finish_reason,
            prompt_tokens,
            completion_tokens,
            len(content),
        )
        if not content:
            logger.warning(
                "OpenAI returned empty message content (model=%s, finish_reason=%s, refusal=%r)",
                getattr(chat_completion, "model", model),
                finish_reason,
                refusal,
            )
            raise EmptyResponseError("OpenAI returned empty message content.")
        if _is_truncation_reason(finish_reason):
            logger.warning(
                "OpenAI truncated the reply at the token limit "
                "(model=%s, completion_tokens=%s)",
                getattr(chat_completion, "model", model),
                completion_tokens,
            )
            raise TruncatedResponseError(
                "OpenAI stopped at the output token limit.", raw_reply=content
            )
        return content

    @staticmethod
    def _gemini_generate_once(gen_model, prompt, max_output_tokens=2048):
        response = gen_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                max_output_tokens=max_output_tokens,
            ),
        )
        text = ((response.text or "") if hasattr(response, "text") else "").strip()
        if not text:
            raise EmptyResponseError("Gemini returned empty response text.")
        candidates = getattr(response, "candidates", None) or []
        finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        if _is_truncation_reason(finish_reason):
            logger.warning("Gemini truncated the reply at the token limit.")
            raise TruncatedResponseError(
                "Gemini stopped at the output token limit.", raw_reply=text
            )
        return text

    def call_openai(
        self, api_key, system_prompt, user_text, prompting_strategy, model="gpt-4o"
    ):
        """Call OpenAI GPT model.

        ``model`` defaults to ``gpt-4o`` so existing (v1) callers and tests keep
        working; the v2 ``/generate`` flow passes the model selected from the
        registry.
        """
        if not user_text:
            logger.warning("call_openai: empty user_text provided")
        start_time = time.time()
        prompt = self.prompt_builder.build_prompt(prompting_strategy, user_text)
        logger.debug(
            "call_openai: strategy=%s, model=%s, user_text_len=%d, prompt_len=%d",
            prompting_strategy,
            model,
            len(user_text or ""),
            len(prompt or ""),
        )

        openai_base_url = self._config_value("OPENAI_BASE_URL")
        client_kwargs = {"api_key": api_key}
        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url
            logger.info("Using configured OpenAI base URL")
        client = OpenAI(**client_kwargs)

        try:
            if prompting_strategy == "few_shot":
                try:
                    logger.info("Running OpenAI few-shot multi-call orchestration")
                    return self._run_few_shot_orchestration(
                        user_text,
                        lambda prompt: self._openai_generate_once(
                            client, system_prompt, model, prompt
                        ),
                    )
                except Exception as orchestration_error:
                    logger.warning(
                        "Few-shot orchestration failed: %s", orchestration_error
                    )
                    raise

            logger.info("Calling OpenAI chat.completions (model=%s)", model)
            if prompting_strategy == "zero_shot":
                prompt_preview = (prompt or "")[:240].replace("\n", "\\n")
                logger.debug(
                    "OpenAI zero-shot prompt preview (len=%d): %s",
                    len(prompt or ""),
                    prompt_preview,
                )
            content = self._openai_generate_once(client, system_prompt, model, prompt)
            duration = time.time() - start_time
            logger.info(
                "OpenAI response received in %.3fs (len=%d)",
                duration,
                len(content),
            )
            model_name = (model or "").lower()
            if (
                prompting_strategy == "zero_shot"
                and model_name.startswith("gpt-5")
                and not self._has_json_object(content)
            ):
                preview = (content or "")[:120].replace("\n", "\\n")
                logger.warning(
                    "OpenAI zero-shot produced non-JSON output for GPT-5 (len=%d, preview=%r); retrying with strict JSON reminder",
                    len(content),
                    preview,
                )
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"{STRICT_JSON_REMINDER}\n"
                    "Return exactly one JSON object."
                )
                content = self._openai_generate_once(
                    client,
                    system_prompt,
                    model,
                    retry_prompt,
                )
                logger.info(
                    "OpenAI zero-shot retry received (len=%d, has_json=%s)",
                    len(content),
                    self._has_json_object(content),
                )
            if prompting_strategy == "zero_shot" and not content:
                logger.warning(
                    "OpenAI zero-shot returned empty content (model=%s, user_text_len=%d)",
                    model,
                    len(user_text or ""),
                )
                raise EmptyResponseError("OpenAI zero-shot returned empty content.")
            return content.strip()
        except Exception as e:
            logger.exception("OpenAI call failed: %s", e)
            raise

    def call_gemini(
        self,
        api_key,
        system_prompt,
        user_text,
        prompting_strategy,
        model=None,
    ):
        """Call Google Gemini model.

        ``model`` is required; callers must pass the model from the request body
        or the registry default.  None is rejected immediately.
        """
        if not model:
            raise ValueError("call_gemini: model must be specified")
        if not user_text:
            logger.warning("call_gemini: empty user_text provided")
        start_time = time.time()
        prompt = self.prompt_builder.build_prompt(prompting_strategy, user_text)
        logger.debug(
            "call_gemini: strategy=%s, model=%s, user_text_len=%d, prompt_len=%d",
            prompting_strategy,
            model,
            len(user_text or ""),
            len(prompt or ""),
        )

        gemini_api_endpoint = self._config_value("GEMINI_API_ENDPOINT")
        genai_kwargs = {"api_key": api_key}
        if gemini_api_endpoint:
            genai_kwargs["client_options"] = {"api_endpoint": gemini_api_endpoint}
            logger.info("Using configured Gemini API endpoint")
        genai.configure(**genai_kwargs)

        gen_model = genai.GenerativeModel(
            model_name=model, system_instruction=system_prompt
        )

        try:
            if prompting_strategy == "few_shot":
                try:
                    logger.info("Running Gemini few-shot multi-call orchestration")
                    return self._run_few_shot_orchestration(
                        user_text,
                        lambda step_prompt: self._gemini_generate_once(
                            gen_model, step_prompt
                        ),
                    )
                except Exception as orchestration_error:
                    logger.warning(
                        "Few-shot orchestration failed: %s", orchestration_error
                    )
                    raise

            logger.info("Calling Gemini generate_content (model=%s)", model)
            text = self._gemini_generate_once(gen_model, prompt)
            duration = time.time() - start_time
            logger.info(
                "Gemini response received in %.3fs (len=%d)",
                duration,
                len(text),
            )
            return text.strip()
        except Exception as e:
            logger.exception("Gemini call failed: %s", e)
            raise

    def generate(
        self,
        api_key,
        provider,
        model,
        user_text,
        system_prompt,
        prompting_strategy="zero_shot",
    ):
        """Provider-agnostic entry point used by the v2 ``/generate`` route.

        Looks up the dispatch method for ``provider`` in the registry and calls
        it with the registry-selected ``model``. Raises ``ValueError`` if the
        provider has no dispatch mapping (the route validates the pair against
        the registry first, so this is a defensive guard).
        """
        method_name = model_registry.dispatch_method(provider)
        if method_name is None:
            raise ValueError(f"Unsupported provider: {provider}")
        method = getattr(self, method_name)
        return method(
            api_key=api_key,
            system_prompt=system_prompt,
            user_text=user_text,
            prompting_strategy=prompting_strategy,
            model=model,
        )

    def generate_pnml(self, api_key, provider, model, user_text, system_prompt):
        """Experimental direct text-to-PNML entry point for ``/generate_pnml_direct``.

        Parallel to ``generate``: same provider dispatch, but one bare provider
        call with the PNML system prompt and the raw user text — no
        PromptBuilder, no few-shot orchestration, no JSON handling.

        Returns a :class:`PnmlGeneration`: the full attempt history plus the
        index of the delivered one. Per contract the document is delivered even
        with remaining issues; the route exposes them in the
        ``X-Validation-Issues`` response header.
        """
        method_name = model_registry.dispatch_method(provider)
        if method_name == "call_openai":
            client_kwargs = {"api_key": api_key}
            openai_base_url = self._config_value("OPENAI_BASE_URL")
            if openai_base_url:
                client_kwargs["base_url"] = openai_base_url
            client = OpenAI(**client_kwargs)
            # reasoning_effort is a GPT-5-only knob; other models reject the
            # parameter (same gating as the temperature special case).
            reasoning_effort = (
                "low" if (model or "").lower().startswith("gpt-5") else None
            )

            def generate_once(prompt):
                return self._openai_generate_once(
                    client,
                    system_prompt,
                    model,
                    prompt,
                    max_completion_tokens=_PNML_OPENAI_MAX_COMPLETION_TOKENS,
                    reasoning_effort=reasoning_effort,
                )

        elif method_name == "call_gemini":
            genai_kwargs = {"api_key": api_key}
            gemini_api_endpoint = self._config_value("GEMINI_API_ENDPOINT")
            if gemini_api_endpoint:
                genai_kwargs["client_options"] = {"api_endpoint": gemini_api_endpoint}
            genai.configure(**genai_kwargs)
            gen_model = genai.GenerativeModel(
                model_name=model, system_instruction=system_prompt
            )

            def generate_once(prompt):
                return self._gemini_generate_once(
                    gen_model,
                    prompt,
                    max_output_tokens=_PNML_GEMINI_MAX_OUTPUT_TOKENS,
                )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        def attempt(document):
            """Normalize and validate one provider reply into a record."""
            result = self.pnml_validator.check(document)
            for entry in result.stripped:
                logger.info("PNML normalization: %s", entry)
            return PnmlAttempt(
                pnml=result.pnml,
                issues=result.issues,
                stripped=result.stripped,
                counts=result.counts,
            )

        reply = generate_once(user_text)
        raw = self._extract_pnml_document(reply)
        if raw is None:
            raise EmptyResponseError(
                "Provider reply contained no PNML document.", raw_reply=reply
            )

        # Correction loop: up to _PNML_MAX_CORRECTIONS passes, each carrying
        # the previous document plus the combined issue list. Stops early
        # when a pass leaves the issues absolutely identical (no progress).
        # Cosmetic deviations never reach this loop -- the validator strips
        # them deterministically, so no correction pass is spent on them.
        # Every pass is recorded: the raw history is the only place where the
        # model's unaided first shot, and any shrinkage along the way, survive.
        attempts = [attempt(raw)]
        best_index = 0
        previous_issues = None
        while attempts[-1].issues and len(attempts) <= _PNML_MAX_CORRECTIONS:
            current = attempts[-1]
            if current.issues == previous_issues:
                logger.warning(
                    "PNML correction made no progress (identical issues); "
                    "stopping after %d correction(s)",
                    len(attempts) - 1,
                )
                break
            previous_issues = current.issues

            logger.info(
                "PNML correction attempt %d/%d for %d issue(s)",
                len(attempts),
                _PNML_MAX_CORRECTIONS,
                len(current.issues),
            )
            repair_prompt = self._build_pnml_repair_prompt(
                user_text, current.pnml, current.issues
            )
            try:
                corrected = self._extract_pnml_document(generate_once(repair_prompt))
            except TruncatedResponseError:
                # A truncated repair has nothing to add; keep the best attempt
                # so far rather than discarding it. Truncation on the initial
                # generation still raises -- there is nothing to fall back to.
                logger.warning(
                    "PNML correction truncated at the token limit; "
                    "keeping the previous attempt"
                )
                break
            if corrected is None:
                logger.warning(
                    "PNML correction reply contained no PNML document; "
                    "keeping the previous attempt"
                )
                break

            attempts.append(attempt(corrected))
            if len(attempts[-1].issues) < len(attempts[best_index].issues):
                best_index = len(attempts) - 1

        return self._finish_pnml_generation(attempts, best_index)

    @staticmethod
    def _finish_pnml_generation(attempts, best_index):
        """Log what the history reveals and wrap it into the result."""
        first, best = attempts[0], attempts[best_index]

        if best.issues:
            logger.warning(
                "PNML validation found %d remaining issue(s) after %d "
                "correction(s): %s",
                len(best.issues),
                len(attempts) - 1,
                "; ".join(best.issues),
            )

        # Structural validity is bought with nodes: the correction loop selects
        # the attempt with the fewest issues, and deleting a node is a cheap way
        # to satisfy the validator. A net that shrank is the one failure mode
        # the structural checks cannot see, so it is called out explicitly.
        if best_index and (
            best.counts.transitions < first.counts.transitions
            or best.counts.places < first.counts.places
        ):
            logger.warning(
                "PNML correction shrank the net: %s -> %s (places, transitions, "
                "arcs); the delivered net may have lost part of the process",
                tuple(first.counts),
                tuple(best.counts),
            )

        return PnmlGeneration(attempts=attempts, best_index=best_index)
