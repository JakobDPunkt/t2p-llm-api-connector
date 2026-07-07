"""Post-processing for the experimental direct text-to-PNML path.

Plays the role that JSON extraction, ``ModelValidator`` and the repair prompt
play on the standard ``/generate`` path — but for PNML XML output. All
functions are pure (no I/O, no LLM calls); the calling service decides what
to do with detected issues.
"""

import logging
import re

logger = logging.getLogger(__name__)

_PNML_BLOCK_RE = re.compile(r"<pnml\b.*?</pnml>", re.DOTALL | re.IGNORECASE)


def extract_pnml(raw_text):
    """Cut the PNML document out of a raw LLM reply.

    Replies may wrap the XML in markdown fences or surrounding prose despite
    the system prompt forbidding it (the XML counterpart of
    ``_extract_json_object`` on the JSON path). Returns the ``<pnml>...</pnml>``
    block with an XML declaration, or ``None`` if the reply contains none.
    """
    if not raw_text:
        return None
    match = _PNML_BLOCK_RE.search(raw_text)
    if match is None:
        return None
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{match.group(0)}'


# TODO(pnml-demo): register real validators here. Each check is a callable
# (pnml_xml) -> list[str] of human-readable issue strings. Planned checks:
# well-formed XML, bipartite arcs only, exactly one marked source place,
# exactly one sink place, unique ids, resolvable arc references.
_CHECKS = []


def validate_pnml(pnml_xml):
    """Validate generated PNML; returns a list of issue strings (empty = ok).

    TODO(pnml-demo): intentionally still a stub — ``_CHECKS`` is empty, so no
    validation happens yet. The interface mirrors
    ``ModelValidator.validate_model`` on the JSON path so validators and a
    repair/retry loop can be plugged in without touching callers.
    """
    issues = []
    for check in _CHECKS:
        issues.extend(check(pnml_xml))
    return issues


def build_repair_prompt(user_text, pnml_xml, issues):
    """Build the corrective follow-up prompt for an invalid PNML reply.

    Counterpart of ``LLMService._build_repair_prompt`` on the JSON path.
    TODO(pnml-demo): not yet wired into a retry loop — the service currently
    only logs validation issues (the standard few-shot path runs exactly one
    repair pass; final retry policy is still to be decided).
    """
    issue_lines = "\n".join(f"- {issue}" for issue in issues)
    return (
        "The PNML you produced for the following process description is "
        "invalid.\n\n"
        f"Process description:\n{user_text}\n\n"
        f"Your previous PNML:\n{pnml_xml}\n\n"
        f"Detected issues:\n{issue_lines}\n\n"
        "Return the corrected model as exactly one well-formed PNML XML "
        "document and nothing else."
    )
