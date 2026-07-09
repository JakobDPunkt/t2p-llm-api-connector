"""Experimental direct text-to-PNML endpoint, parallel to ``/generate``.

The standard path returns BPMN JSON that t2p-2.0 feeds through the
model-transformer; this endpoint lets the LLM emit PNML directly. Auth,
payload validation, error mapping and metrics reuse the ``/generate``
machinery unchanged.

Response contract: the body is one pure, geometry-free PNML document
(application/xml), structurally equivalent to the model-transformer's
output, so downstream post-processing (coordinate assignment in t2p-2.0)
keeps working. Validation issues remaining after the correction loop do
not block delivery (best effort); they are reported in the CORS-exposed
``X-Validation-Issues`` response header.
"""

import logging
import time

from flasgger import swag_from
from flask import Response, current_app, jsonify, request
from flask_cors import cross_origin

from app.api import bp
from app.api.routes import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    _extract_bearer_key,
    _is_quota_error,
    _llm_service,
    _v2_error,
    _validate_generate_payload,
)
from app.services import model_registry
from app.services.llm_service import EmptyResponseError, TruncatedResponseError

logger = logging.getLogger(__name__)

#: How much of a non-PNML provider reply to surface in the debug view.
_REPLY_EXCERPT_LIMIT = 500

def _debug_requested():
    """Whether the caller asked for the attempt-history JSON (demo only)."""
    return request.args.get("debug") == "1"


def _reply_excerpt(reply):
    """A single-line, length-capped excerpt of a raw provider reply."""
    if not reply:
        return ""
    collapsed = " ".join(str(reply).split())
    if len(collapsed) <= _REPLY_EXCERPT_LIMIT:
        return collapsed
    return collapsed[:_REPLY_EXCERPT_LIMIT] + " […]"


def _tokens(usage):
    """Serialize a ``TokenUsage``, or None when the provider reported none."""
    return usage._asdict() if usage else None


def _generation_debug_payload(generation):
    """Serialize the full generation history for the demo's debug view.

    ``attempts[0]`` is the model's unaided first shot; every further entry is
    one correction pass. ``delivered_index`` marks which attempt was returned.
    The top-level ``tokens`` cover every provider call, so they can exceed the
    sum of the per-attempt ones.
    """
    return {
        "pnml": generation.best.pnml,
        "delivered_index": generation.best_index,
        "tokens": _tokens(generation.tokens),
        "attempts": [
            {
                "issues": list(a.issues),
                "counts": a.counts._asdict(),
                "tokens": _tokens(generation.usage_for(i)),
            }
            for i, a in enumerate(generation.attempts)
        ],
    }


@bp.route("/demo")
def pnml_demo():
    """Serve the comparison demo page for the direct-PNML experiment.

    Demo tooling analogous to the Swagger UI at ``/docs``: one static page
    that calls ``/generate_pnml_direct`` (this connector) and the live deployment's
    ``/v2/generate/pnml`` side by side.
    """
    return current_app.send_static_file("pnml_demo.html")


@bp.route("/generate_pnml_direct", methods=["POST"])
# Browser demo clients call this endpoint directly; expose the issues header
# so cross-origin JavaScript may read it.
@cross_origin(expose_headers=["X-Validation-Issues"])
@swag_from(
    {
        "tags": ["pnml-direct-experiment"],
        "summary": "Generate PNML directly (experimental)",
        "description": (
            "Generate a PNML Petri net directly from process text via a "
            "PNML-enriched prompt, bypassing the model-transformer."
        ),
        "security": [{"bearerAuth": []}],
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["user_text", "provider", "model"],
                        "properties": {
                            "user_text": {"type": "string"},
                            "provider": {"type": "string"},
                            "model": {"type": "string"},
                        },
                    }
                }
            },
        },
        "responses": {
            "200": {
                "description": "PNML document",
                "content": {"application/xml": {"schema": {"type": "string"}}},
            },
            "400": {"description": "Invalid request or unusable provider reply"},
            "401": {"description": "Missing or malformed Authorization header"},
            "429": {"description": "Provider quota or rate limit exceeded"},
            "500": {"description": "Upstream provider failure"},
        },
    }
)
def generate_pnml():
    """Generate a PNML document directly from a process description."""
    start_time = time.time()
    status = "200"
    try:
        api_key = _extract_bearer_key()
        data = request.get_json(silent=True)
        validation_error = _validate_generate_payload(api_key, data)
        if validation_error is not None:
            status = str(validation_error[1])
            return validation_error

        provider = data["provider"]
        model = data["model"]
        try:
            model_registry.refresh_model_cache(provider=provider, api_key=api_key)
        except Exception as refresh_err:
            logger.warning(
                "Model cache refresh failed for provider %s: %s",
                provider,
                refresh_err,
            )

        logger.info(
            "Invoking LLMService.generate_pnml (provider=%s, model=%s)",
            provider,
            model,
        )
        generation = _llm_service.generate_pnml(
            api_key=api_key,
            provider=provider,
            model=model,
            user_text=data["user_text"],
            system_prompt=current_app.config["PNML_SYSTEM_PROMPT"],
        )
        best = generation.best

        # Debug view (demo only): the full attempt history as JSON, so the page
        # can show the model's unaided first shot and what each correction pass
        # changed. The default contract is unchanged: pure application/xml.
        if _debug_requested():
            return jsonify(_generation_debug_payload(generation)), 200

        response = Response(best.pnml, status=200, mimetype="application/xml")
        if best.issues:
            # Best-effort delivery per contract: the body stays pure PNML,
            # remaining validation issues travel in the header.
            response.headers["X-Validation-Issues"] = "; ".join(best.issues)
        return response

    except Exception as e:
        if isinstance(e, TruncatedResponseError):
            status = "400"
            excerpt = _reply_excerpt(getattr(e, "raw_reply", None))
            logger.warning(
                "/generate_pnml_direct truncated at token limit: %s (reply: %s)",
                e,
                excerpt or "<empty>",
            )
            return _v2_error(
                400,
                "response_truncated",
                (
                    "The model reached its output token limit before finishing "
                    "the net. The process may be too long for this model. Try a "
                    "shorter description or a larger model."
                ),
                details=[f"Partial reply: {excerpt}"] if excerpt else None,
            )

        if isinstance(e, EmptyResponseError):
            status = "400"
            excerpt = _reply_excerpt(getattr(e, "raw_reply", None))
            logger.warning(
                "/generate_pnml_direct rejected provider response: %s (reply: %s)",
                e,
                excerpt or "<empty>",
            )
            return _v2_error(
                400,
                "invalid_request",
                "The LLM provider returned no usable PNML document.",
                details=[f"Provider reply: {excerpt}"] if excerpt else None,
            )

        if _is_quota_error(e):
            status = "429"
            logger.warning("/generate_pnml_direct provider quota exceeded: %s", e)
            return _v2_error(
                429,
                "rate_limited",
                (
                    "Provider quota or rate limit exceeded. "
                    "Try again later or use another model."
                ),
            )

        status = "500"
        logger.exception("/generate_pnml_direct failed: %s", e)
        return _v2_error(500, "upstream_error", "The LLM provider call failed.")
    finally:
        REQUEST_COUNT.labels(
            method="POST", endpoint="/generate_pnml_direct", status=status
        ).inc()
        REQUEST_LATENCY.labels(method="POST", endpoint="/generate_pnml_direct").observe(
            time.time() - start_time
        )
