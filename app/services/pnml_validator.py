"""Validator for directly generated PNML.

Structural counterpart of ``ModelValidator`` on the standard BPMN-JSON path,
with one deliberate difference: there is NO heuristic sanitize step. Silent
deterministic repairs often mask real defects with plausible-but-wrong
results; invalid output is reported and corrected via the LLM instead.
"""


class PnmlValidator:
    """Validate directly generated PNML XML."""

    def validate_pnml(self, pnml_xml, source_text=""):
        """Return a list of validation issues found in the PNML document.

        TODO(pnml-demo): content deliberately empty — the concrete checks
        follow the structural invariants in docs/PNML_DIRECT_CONTRACT.md
        (well-formed XML, bipartite arcs, marked source/sink places, unique
        ids, resolvable arc references, connectivity).
        """
        return []
