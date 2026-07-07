"""Experimental direct text-to-PNML endpoint, parallel to ``/generate``.

The standard path returns BPMN JSON that t2p-2.0 feeds through the
model-transformer; this endpoint lets the LLM emit PNML directly and passes
it through. Auth, payload validation, error mapping and metrics reuse the
``/generate`` machinery unchanged.

TODO(pnml-demo): the response contract is provisional (raw PNML as
application/xml). The final contract must stay compatible with what the
standard pipeline does after the model-transformer (coordinate assignment
in t2p-2.0 etc.).
"""

import logging
import time

from flasgger import swag_from
from flask import Response, current_app, request
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
from app.services.llm_service import EmptyResponseError

logger = logging.getLogger(__name__)


@bp.route("/generate_pnml", methods=["POST"])
@cross_origin()  # browser demo clients call this endpoint directly
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
        pnml = _llm_service.generate_pnml(
            api_key=api_key,
            provider=provider,
            model=model,
            user_text=data["user_text"],
            system_prompt=current_app.config["PNML_SYSTEM_PROMPT"],
        )
        return Response(pnml, status=200, mimetype="application/xml")

    except Exception as e:
        if isinstance(e, EmptyResponseError):
            status = "400"
            logger.warning("/generate_pnml rejected provider response: %s", e)
            return _v2_error(
                400,
                "invalid_request",
                "The LLM provider returned no usable PNML document.",
            )

        if _is_quota_error(e):
            status = "429"
            logger.warning("/generate_pnml provider quota exceeded: %s", e)
            return _v2_error(
                429,
                "rate_limited",
                (
                    "Provider quota or rate limit exceeded. "
                    "Try again later or use another model."
                ),
            )

        status = "500"
        logger.exception("/generate_pnml failed: %s", e)
        return _v2_error(500, "upstream_error", "The LLM provider call failed.")
    finally:
        REQUEST_COUNT.labels(
            method="POST", endpoint="/generate_pnml", status=status
        ).inc()
        REQUEST_LATENCY.labels(method="POST", endpoint="/generate_pnml").observe(
            time.time() - start_time
        )
