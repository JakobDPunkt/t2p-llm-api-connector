"""The connector's one error pipeline.

Every endpoint answers a failure in the same shape -- ``{"error": {"code",
"message", "details"?}}`` -- and every provider failure is classified in one
place. Nothing about that classification depends on what an endpoint produces:
an empty reply, a reply cut off at the output-token limit and an exhausted quota
are provider events, not properties of BPMN or PNML. Only the noun in the
message differs, and a noun is an argument (``artefact``), not a second code
path.

The status class carries meaning downstream: t2p-2.0 relays a 4xx body to WoPeD
unchanged and collapses everything else into a generic "connector error". So a
failure that a retry cannot heal -- an empty reply, a truncated one, an
exhausted quota -- is reported as 4xx, where its message survives the trip to
the user; only genuinely unknown failures are 5xx.
"""

import logging
from collections import namedtuple

from flask import jsonify

from app.services.llm_service import EmptyResponseError, TruncatedResponseError

logger = logging.getLogger(__name__)

#: How much of a raw provider reply to surface as a diagnostic detail.
_REPLY_EXCERPT_LIMIT = 500

#: One classified failure: HTTP status, contract code, the message the caller
#: sees, and optional diagnostic strings.
ProviderFailure = namedtuple("ProviderFailure", "status code message details")


def error_response(status_code, code, message, details=None):
    """Build the standard connector error body and status tuple.

    ``details`` is an optional list of extra diagnostic strings, surfaced by the
    demo's error banner (contract: ``{"error": {code, message, details?}}``).
    """
    error = {"code": code, "message": message}
    if details:
        error["details"] = details
    return jsonify({"error": error}), status_code


def is_quota_error(exc):
    """Return True for provider quota/rate-limit style exceptions."""
    text = str(exc or "").lower()
    indicators = (
        "quota",
        "resourceexhausted",
        "too many requests",
        "rate limit",
        "perday",
    )
    return any(token in text for token in indicators)


def reply_excerpt(reply):
    """A single-line, length-capped excerpt of a raw provider reply."""
    if not reply:
        return ""
    collapsed = " ".join(str(reply).split())
    if len(collapsed) <= _REPLY_EXCERPT_LIMIT:
        return collapsed
    return collapsed[:_REPLY_EXCERPT_LIMIT] + " […]"


def classify_provider_error(exc, artefact="response"):
    """Classify a provider exception into the one contract shape.

    ``artefact`` names what the failed call was meant to produce ("PNML
    document", "BPMN JSON model"), so the same classification speaks the
    endpoint's language without branching on it.
    """
    excerpt = reply_excerpt(getattr(exc, "raw_reply", None))

    if isinstance(exc, TruncatedResponseError):
        return ProviderFailure(
            400,
            "response_truncated",
            (
                f"The model reached its output token limit before finishing the "
                f"{artefact}. The process may be too long for this model. Try a "
                "shorter description or a larger model."
            ),
            [f"Partial reply: {excerpt}"] if excerpt else None,
        )

    if isinstance(exc, EmptyResponseError):
        return ProviderFailure(
            400,
            "invalid_request",
            f"The LLM provider returned no usable {artefact}.",
            [f"Provider reply: {excerpt}"] if excerpt else None,
        )

    if is_quota_error(exc):
        return ProviderFailure(
            429,
            "rate_limited",
            (
                "Provider quota or rate limit exceeded. "
                "Try again later or use another model."
            ),
            None,
        )

    return ProviderFailure(500, "upstream_error", "The LLM provider call failed.", None)


def _log_failure(endpoint, exc, failure):
    """Log a classified failure; only the unclassified kind warrants a traceback."""
    if failure.status >= 500:
        logger.exception("%s failed: %s", endpoint, exc)
    else:
        logger.warning(
            "%s answered %d %s: %s", endpoint, failure.status, failure.code, exc
        )


def provider_error_response(endpoint, exc, artefact="response"):
    """Log a provider failure and render it as the contract error response."""
    failure = classify_provider_error(exc, artefact)
    _log_failure(endpoint, exc, failure)
    return error_response(
        failure.status, failure.code, failure.message, failure.details
    )


def provider_job_error(endpoint, exc, artefact="response"):
    """Log a provider failure and render it for an async job's status body.

    The job body carries one ``detail`` string where the response carries a
    ``details`` list. t2p-2.0 polls that shape, so it stays as it is; the code
    and message come from the same classification either way.
    """
    failure = classify_provider_error(exc, artefact)
    _log_failure(endpoint, exc, failure)
    return {"code": failure.code, "message": failure.message, "detail": str(exc)}
