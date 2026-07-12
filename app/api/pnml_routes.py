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

On the XML path the model writes that document and nothing else, so nothing
is extracted from its reply and nothing is translated: what it answers is
what the validator judges. The JSON path is the one place where a document
is built rather than received; see :mod:`app.services.pnml_json`.
"""

import logging
import time

from flasgger import swag_from
from flask import Response, current_app, jsonify, request
from flask_cors import cross_origin

from app.api import bp
from app.api.errors import provider_error_response
from app.api.routes import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    _extract_bearer_key,
    _llm_service,
    _validate_generate_payload,
)
from app.services import model_registry
from app.services.llm_service import PNML_FORMATS

logger = logging.getLogger(__name__)


def _debug_requested():
    """Whether the caller asked for the attempt-history JSON (demo only)."""
    return request.args.get("debug") == "1"


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


def _pnml_swagger(summary, description):
    """The two direct endpoints differ in prose only; their contract is one."""
    return {
        "tags": ["pnml-direct-experiment"],
        "summary": summary,
        "description": description,
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


def _generate_pnml_response(fmt, endpoint, prompt_key):
    """Run one direct generation and render it under the shared contract.

    The XML and the JSON path differ only in what the model is asked to write.
    Auth, payload validation, the correction loop, the delivery rule (body is
    pure PNML, remaining issues travel in the header), the debug view, the
    error mapping and the metrics are the same for both, so they live here once.
    """
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
            "Invoking LLMService.generate_pnml (provider=%s, model=%s, fmt=%s)",
            provider,
            model,
            fmt,
        )
        generation = _llm_service.generate_pnml(
            api_key=api_key,
            provider=provider,
            model=model,
            user_text=data["user_text"],
            system_prompt=current_app.config[prompt_key],
            fmt=fmt,
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
        body, status_code = provider_error_response(
            endpoint, e, artefact=PNML_FORMATS[fmt].artefact
        )
        status = str(status_code)
        return body, status_code
    finally:
        REQUEST_COUNT.labels(
            method="POST", endpoint=endpoint, status=status
        ).inc()
        REQUEST_LATENCY.labels(method="POST", endpoint=endpoint).observe(
            time.time() - start_time
        )


# Browser demo clients call these endpoints directly; expose the issues header
# so cross-origin JavaScript may read it.
@bp.route("/generate_pnml_direct", methods=["POST"])
@cross_origin(expose_headers=["X-Validation-Issues"])
@swag_from(
    _pnml_swagger(
        "Generate PNML directly (experimental)",
        "Generate a PNML Petri net directly from process text via a "
        "PNML-enriched prompt, bypassing the model-transformer. The model "
        "writes the PNML document itself.",
    )
)
def generate_pnml():
    """Generate a PNML document directly from a process description."""
    return _generate_pnml_response("xml", "/generate_pnml_direct", "PNML_SYSTEM_PROMPT")


@bp.route("/generate_pnml_direct_json", methods=["POST"])
@cross_origin(expose_headers=["X-Validation-Issues"])
@swag_from(
    _pnml_swagger(
        "Generate PNML via a JSON net (experimental)",
        "Same direct path, without the model-transformer and without BPMN: the "
        "model answers with a schema-constrained JSON net, which this service "
        "serializes to PNML. Well-formedness, unique ids, arc ids and the "
        "initial marking are therefore construction, not validation.",
    )
)
def generate_pnml_json():
    """Generate PNML from a schema-constrained JSON net."""
    return _generate_pnml_response(
        "json", "/generate_pnml_direct_json", "PNML_JSON_SYSTEM_PROMPT"
    )
