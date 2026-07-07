"""Validator for directly generated PNML.

Structural counterpart of ``ModelValidator`` on the standard BPMN-JSON path,
with one deliberate difference: there is NO heuristic sanitize step. Silent
deterministic repairs often mask real defects with plausible-but-wrong
results; invalid output is reported and corrected via the LLM instead.

The checks are layered; each level only runs when the previous one passed,
so every issue is reported in the terms of the lowest level it breaks:

- Level 0: the output is valid XML at all.
- Level 1: PNML document structure per docs/PNML_DIRECT_CONTRACT.md —
  everything visible locally in the document, without graph reasoning.
- Level 2: graph / workflow-net structure — the static workflow-net
  properties (contract invariants 2-6): resolvable arc references,
  bipartiteness, exactly one marked source place and one sink place,
  transition connectivity, and every node on a path from source to sink.
  Behavioural soundness (token game analysis) is deliberately out of scope.

XML namespaces are tolerated throughout (matching the namespace-agnostic
downstream parsers): elements are matched by local name.
"""

import xml.etree.ElementTree as ET

_PT_NET_TYPE = "http://www.informatik.hu-berlin.de/top/pntd/ptNetb"
_ALLOWED_NET_CHILDREN = {"place", "transition", "arc"}
_FORBIDDEN_ELEMENTS = {
    "graphics": "the output is geometry-free; layouting happens downstream",
    "toolspecific": "native standard PNML only, no tool-specific blocks",
    "inscription": "arc inscriptions are not part of the contract",
    "page": "the net's elements sit directly under <net>",
}


def _local_name(tag):
    """Return the tag name without a Clark-notation namespace prefix."""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _iter_local(root, local_name):
    """Iterate elements by local tag name, tolerating any XML namespace."""
    for element in root.iter():
        if _local_name(element.tag) == local_name:
            yield element


def _child_text(element, child_local_name):
    """Return the ``<text>`` content of a child element, or ``None``.

    ``None`` means the child is absent; the empty string means the child is
    present but carries no usable text.
    """
    for child in element:
        if _local_name(child.tag) != child_local_name:
            continue
        for sub in child:
            if _local_name(sub.tag) == "text":
                return (sub.text or "").strip()
        return ""
    return None


class PnmlValidator:
    """Validate directly generated PNML XML."""

    def validate_pnml(self, pnml_xml, source_text=""):
        """Return a list of validation issues found in the PNML document."""
        issues, root = self._check_valid_xml(pnml_xml)
        if issues:
            return issues

        issues = self._check_document_structure(root)
        if issues:
            return issues

        return self._check_graph_structure(root)

    # --- Level 0 -----------------------------------------------------------

    @staticmethod
    def _check_valid_xml(pnml_xml):
        """Level 0: the output must parse as XML. Returns (issues, root)."""
        if not isinstance(pnml_xml, str) or not pnml_xml.strip():
            return ["output is not valid XML: document is empty"], None
        try:
            return [], ET.fromstring(pnml_xml)
        except ET.ParseError as exc:
            return [f"output is not valid XML: {exc}"], None

    # --- Level 1 -----------------------------------------------------------

    def _check_document_structure(self, root):
        """Level 1: PNML document structure per the return-format contract."""
        issues = []

        if _local_name(root.tag) != "pnml":
            return [f"root element must be <pnml>, found <{_local_name(root.tag)}>"]

        nets = list(_iter_local(root, "net"))
        if len(nets) != 1:
            return [f"document must contain exactly one <net>, found {len(nets)}"]
        net = nets[0]

        if not net.get("id"):
            issues.append("<net> is missing its 'id' attribute")
        net_type = net.get("type")
        if net_type != _PT_NET_TYPE:
            issues.append(
                f"<net> type must be '{_PT_NET_TYPE}', found '{net_type}'"
            )

        for child in net:
            child_name = _local_name(child.tag)
            if child_name not in _ALLOWED_NET_CHILDREN | set(_FORBIDDEN_ELEMENTS):
                issues.append(
                    f"unexpected element <{child_name}> under <net>; allowed "
                    "are <place>, <transition> and <arc>"
                )

        for element in root.iter():
            element_name = _local_name(element.tag)
            if element_name in _FORBIDDEN_ELEMENTS:
                issues.append(
                    f"forbidden element <{element_name}>: "
                    f"{_FORBIDDEN_ELEMENTS[element_name]}"
                )

        seen_ids = set()
        for kind in ("place", "transition", "arc"):
            for element in _iter_local(net, kind):
                element_id = element.get("id")
                if not element_id:
                    issues.append(f"<{kind}> without 'id' attribute")
                    continue
                if element_id in seen_ids:
                    issues.append(f"duplicate id '{element_id}'")
                seen_ids.add(element_id)

        for arc in _iter_local(net, "arc"):
            arc_id = arc.get("id") or "<no id>"
            if not arc.get("source") or not arc.get("target"):
                issues.append(
                    f"arc '{arc_id}' is missing its 'source' or 'target' attribute"
                )

        for kind in ("place", "transition"):
            for element in _iter_local(net, kind):
                element_id = element.get("id") or "<no id>"
                name_text = _child_text(element, "name")
                if name_text == "":
                    issues.append(
                        f"<name> of {kind} '{element_id}' must contain a "
                        "non-empty <text>"
                    )

        for place in _iter_local(net, "place"):
            place_id = place.get("id") or "<no id>"
            marking_text = _child_text(place, "initialMarking")
            if marking_text is None:
                continue
            try:
                if int(marking_text) < 0:
                    issues.append(
                        f"initialMarking of place '{place_id}' must be >= 0"
                    )
            except ValueError:
                issues.append(
                    f"initialMarking of place '{place_id}' must be an integer, "
                    f"found '{marking_text}'"
                )

        return issues

    # --- Level 2 -----------------------------------------------------------

    @staticmethod
    def _check_graph_structure(root):
        """Level 2: static workflow-net structure (contract invariants 2-6).

        Runs only on a document that passed level 1, so ids exist and are
        unique and every arc carries source and target attributes.
        """
        issues = []

        place_ids = {p.get("id") for p in _iter_local(root, "place")}
        transition_ids = {t.get("id") for t in _iter_local(root, "transition")}
        node_ids = place_ids | transition_ids

        edges = []
        incoming = {}
        outgoing = {}
        for arc in _iter_local(root, "arc"):
            arc_id = arc.get("id")
            source = arc.get("source")
            target = arc.get("target")

            # Invariant 3 (rest): references resolve, no self-loops.
            if source == target:
                issues.append(f"arc '{arc_id}' connects '{source}' to itself")
            unresolved = False
            for role, ref in (("source", source), ("target", target)):
                if ref not in node_ids:
                    issues.append(
                        f"arc '{arc_id}' references unknown {role} '{ref}'"
                    )
                    unresolved = True
            if unresolved:
                continue

            # Invariant 2: the net is bipartite.
            if source in place_ids and target in place_ids:
                issues.append(
                    f"arc '{arc_id}' connects place '{source}' to place "
                    f"'{target}' (arcs must connect a place and a transition)"
                )
            elif source in transition_ids and target in transition_ids:
                issues.append(
                    f"arc '{arc_id}' connects transition '{source}' to "
                    f"transition '{target}' (arcs must connect a place and a "
                    "transition)"
                )

            edges.append((source, target))
            outgoing.setdefault(source, set()).add(target)
            incoming.setdefault(target, set()).add(source)

        # Invariant 4: exactly one source place and one sink place, and the
        # initial marking sits on the source. The marking is structurally
        # redundant, but it makes the net immediately playable in WoPeD and
        # cross-checks the structure: if marking and source disagree, the
        # messages below point at the exact modelling error.
        sources = sorted(p for p in place_ids if not incoming.get(p))
        sinks = sorted(p for p in place_ids if not outgoing.get(p))
        if len(sources) != 1:
            issues.append(
                "expected exactly one start place without incoming arcs, "
                f"found {len(sources)} ({', '.join(sources) or 'none'})"
            )
        if len(sinks) != 1:
            issues.append(
                "expected exactly one end place without outgoing arcs, "
                f"found {len(sinks)} ({', '.join(sinks) or 'none'})"
            )

        marked = {}
        for place in _iter_local(root, "place"):
            marking_text = _child_text(place, "initialMarking")
            if marking_text is None:
                continue
            try:
                marking = int(marking_text)
            except ValueError:
                continue  # value errors are reported by level 1
            if marking >= 1:
                marked[place.get("id")] = marking

        if len(marked) != 1:
            issues.append(
                "expected exactly one place with an initialMarking >= 1, "
                f"found {len(marked)}"
            )
        else:
            (marked_id, marking), = marked.items()
            if marking != 1:
                issues.append(
                    f"initialMarking of start place '{marked_id}' must be "
                    f"exactly 1, found {marking}"
                )
            if incoming.get(marked_id):
                issues.append(
                    f"place '{marked_id}' carries the initial marking but is "
                    "not the structural start place (it has incoming arcs)"
                )

        # Invariant 5: every transition takes part in the flow (mirrors
        # validate_pnml_connectivity in t2p-2.0).
        for tid in sorted(transition_ids):
            if not incoming.get(tid):
                issues.append(f"transition '{tid}' has no inbound arc")
            if not outgoing.get(tid):
                issues.append(f"transition '{tid}' has no outbound arc")

        # Invariant 6: every node lies on a path from source to sink.
        # Needs an unambiguous source and sink; their absence is already
        # reported above.
        if len(sources) == 1 and len(sinks) == 1:
            reverse = {}
            for source, target in edges:
                reverse.setdefault(target, set()).add(source)

            def _closure(start, neighbours):
                seen = {start}
                frontier = [start]
                while frontier:
                    for node in neighbours.get(frontier.pop(), set()):
                        if node not in seen:
                            seen.add(node)
                            frontier.append(node)
                return seen

            reachable = _closure(sources[0], outgoing)
            co_reachable = _closure(sinks[0], reverse)
            for node in sorted(node_ids - (reachable & co_reachable)):
                issues.append(
                    f"node '{node}' lies on no path from the start place "
                    "to the end place"
                )

        return issues
