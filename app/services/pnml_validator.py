"""Validator for directly generated PNML.

Structural counterpart of ``ModelValidator`` on the standard BPMN-JSON path:
``sanitize_pnml`` applies deterministic repairs, ``validate_pnml`` reports
remaining issues as human-readable strings. Position and architecture mirror
the standard path; only the checked content differs (Petri net structure
instead of BPMN elements).
"""


class PnmlValidator:
    """Validate and sanitize directly generated PNML XML."""

    def sanitize_pnml(self, pnml_xml):
        """Apply deterministic repairs to the PNML document.

        TODO(pnml-demo): content deliberately empty — the concrete repair
        heuristics (counterpart of ``ModelValidator.sanitize_model``: e.g.
        dropping duplicate ids and arcs with unresolvable endpoints) are
        decided and implemented together with Jakob.
        """
        return pnml_xml

    def validate_pnml(self, pnml_xml, source_text=""):
        """Return a list of validation issues found in the PNML document.

        TODO(pnml-demo): content deliberately empty — the concrete checks
        follow the structural invariants in docs/PNML_DIRECT_CONTRACT.md
        (well-formed XML, bipartite arcs, marked source/sink places, unique
        ids, resolvable arc references, connectivity).
        """
        return []
