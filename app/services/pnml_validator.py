"""Validator for directly generated PNML.

Structural counterpart of ``ModelValidator`` on the standard BPMN-JSON path,
with one deliberate difference: there is NO heuristic sanitize step. Silent
deterministic repairs often mask real defects with plausible-but-wrong
results; invalid output is reported and corrected via the LLM instead.

The checks are layered; each level only runs when the previous one passed,
so every issue is reported in the terms of the lowest level it breaks:

- Level 0: the output is valid XML at all.
- Level 1: PNML document structure per docs/PNML_DIRECT_CONTRACT.md
  (TODO(pnml-demo): to be designed with Jakob).
"""

import xml.etree.ElementTree as ET


class PnmlValidator:
    """Validate directly generated PNML XML."""

    def validate_pnml(self, pnml_xml, source_text=""):
        """Return a list of validation issues found in the PNML document."""
        # Level 0: valid XML.
        issues = self._check_valid_xml(pnml_xml)
        if issues:
            return issues

        # TODO(pnml-demo): Level 1 — PNML document structure (contract doc).
        return []

    @staticmethod
    def _check_valid_xml(pnml_xml):
        """Level 0: the output must parse as XML."""
        if not isinstance(pnml_xml, str) or not pnml_xml.strip():
            return ["output is not valid XML: document is empty"]
        try:
            ET.fromstring(pnml_xml)
        except ET.ParseError as exc:
            return [f"output is not valid XML: {exc}"]
        return []
