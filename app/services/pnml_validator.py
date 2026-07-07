"""Validator for directly generated PNML.

Structural counterpart of ``ModelValidator`` on the standard BPMN-JSON path,
with one deliberate difference: there is NO heuristic sanitize step. Silent
deterministic repairs often mask real defects with plausible-but-wrong
results; invalid output is reported and corrected via the LLM instead.

The checks are grouped by level, but gating is limited to what genuinely
invalidates further checking — everything else is collected into ONE combined
issue list so a single correction pass can fix as many real issues as
possible:

- Level 0 (gate): the output is valid XML at all.
- Document skeleton (gate): a <pnml> root with exactly one <net>.
- Graph prerequisites (gate for level 2 only): ids present and unique, arcs
  carry source and target. Without them the graph is not well-defined and
  level-2 findings would be artifacts of these defects.
- Document-local checks (never gate): net attributes, forbidden elements,
  name texts — independent of the graph, always reported.
- Level 2: static workflow-net structure (contract invariants 2-6):
  resolvable arc references, bipartiteness, no duplicate arcs, exactly one
  start and one end place, the single one-token initial marking on the start
  place, transition connectivity, and every node on a path from start to end.
  Behavioural soundness (token game analysis) is deliberately out of scope.

Issue messages state the violated rule and, where all causes allow it, the
possible fixes — deliberately without one-sided repair heuristics that could
point the correction in a wrong direction.

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

        issues = self._check_document_skeleton(root)
        if issues:
            return issues

        local_issues = self._check_document_local(root)
        prerequisite_issues = self._check_graph_prerequisites(root)
        if prerequisite_issues:
            return local_issues + prerequisite_issues

        return local_issues + self._check_graph_structure(root)

    # --- Level 0: valid XML (gate) ------------------------------------------

    @staticmethod
    def _check_valid_xml(pnml_xml):
        """The output must parse as XML. Returns (issues, root)."""
        if not isinstance(pnml_xml, str) or not pnml_xml.strip():
            return ["output is not valid XML: document is empty"], None
        try:
            return [], ET.fromstring(pnml_xml)
        except ET.ParseError as exc:
            return [f"output is not valid XML: {exc}"], None

    # --- Document skeleton (gate) -------------------------------------------

    @staticmethod
    def _check_document_skeleton(root):
        """The document must be a <pnml> root with exactly one <net>."""
        if _local_name(root.tag) != "pnml":
            return [
                f"root element must be <pnml>, found <{_local_name(root.tag)}>"
            ]
        nets = list(_iter_local(root, "net"))
        if len(nets) != 1:
            return [
                f"document must contain exactly one <net>, found {len(nets)}"
            ]
        return []

    # --- Document-local checks (never gate) ----------------------------------

    @staticmethod
    def _check_document_local(root):
        """Local document rules, independent of the graph structure."""
        issues = []
        net = next(_iter_local(root, "net"))

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
                    f"forbidden element <{element_name}> "
                    f"({_FORBIDDEN_ELEMENTS[element_name]}); remove it"
                )

        for kind in ("place", "transition"):
            for element in _iter_local(net, kind):
                element_id = element.get("id") or "<no id>"
                name_text = _child_text(element, "name")
                if name_text == "":
                    issues.append(
                        f"<name> of {kind} '{element_id}' must contain a "
                        "non-empty <text>; add the text or drop the empty "
                        "<name>"
                    )

        return issues

    # --- Graph prerequisites (gate for level 2) -------------------------------

    @staticmethod
    def _check_graph_prerequisites(root):
        """Ids and arc endpoints the graph checks depend on."""
        issues = []
        net = next(_iter_local(root, "net"))

        seen_ids = set()
        for kind in ("place", "transition", "arc"):
            for element in _iter_local(net, kind):
                element_id = element.get("id")
                if not element_id:
                    issues.append(
                        f"<{kind}> without 'id' attribute; every element "
                        "needs a unique id"
                    )
                    continue
                if element_id in seen_ids:
                    issues.append(
                        f"duplicate id '{element_id}'; ids must be unique "
                        "across places, transitions and arcs"
                    )
                seen_ids.add(element_id)

        for arc in _iter_local(net, "arc"):
            arc_id = arc.get("id") or "<no id>"
            if not arc.get("source") or not arc.get("target"):
                issues.append(
                    f"arc '{arc_id}' is missing its 'source' or 'target' "
                    "attribute; every arc references the ids of the two "
                    "nodes it connects"
                )

        return issues

    # --- Level 2: static workflow-net structure ------------------------------

    @staticmethod
    def _check_graph_structure(root):
        """Contract invariants 2-6 on the graph.

        Runs only when the graph prerequisites passed, so ids exist and are
        unique and every arc carries source and target attributes.
        """
        issues = []

        place_ids = {p.get("id") for p in _iter_local(root, "place")}
        transition_ids = {t.get("id") for t in _iter_local(root, "transition")}
        node_ids = place_ids | transition_ids

        edges = []
        incoming = {}
        outgoing = {}
        seen_connections = set()
        for arc in _iter_local(root, "arc"):
            arc_id = arc.get("id")
            source = arc.get("source")
            target = arc.get("target")

            # Invariant 3 (rest): references resolve, no self-loops.
            if source == target:
                issues.append(
                    f"arc '{arc_id}' connects '{source}' to itself; arcs "
                    "must connect two different nodes"
                )
            unresolved = False
            for role, ref in (("source", source), ("target", target)):
                if ref not in node_ids:
                    issues.append(
                        f"arc '{arc_id}' references unknown {role} '{ref}'; "
                        "either the reference is misspelled or the node is "
                        "missing — the arc must reference an existing place "
                        "or transition"
                    )
                    unresolved = True
            if unresolved:
                continue

            # Invariant 2: the net is bipartite.
            if source in place_ids and target in place_ids:
                issues.append(
                    f"arc '{arc_id}' connects place '{source}' to place "
                    f"'{target}'; arcs must alternate between places and "
                    "transitions — either a transition is missing in between "
                    "or the arc itself is wrong"
                )
            elif source in transition_ids and target in transition_ids:
                issues.append(
                    f"arc '{arc_id}' connects transition '{source}' to "
                    f"transition '{target}'; arcs must alternate between "
                    "places and transitions — either a place is missing in "
                    "between or the arc itself is wrong"
                )

            # Duplicate parallel arcs: a typical LLM repetition pattern, and
            # downstream a second identical arc reads as an arc weight of 2.
            if (source, target) in seen_connections:
                issues.append(
                    f"duplicate arc from '{source}' to '{target}' (arc "
                    f"'{arc_id}'); only one arc per direction is allowed "
                    "between two nodes — remove the duplicates"
                )
            seen_connections.add((source, target))

            edges.append((source, target))
            outgoing.setdefault(source, set()).add(target)
            incoming.setdefault(target, set()).add(source)

        # Invariant 4: exactly one start place and one end place.
        sources = sorted(p for p in place_ids if not incoming.get(p))
        sinks = sorted(p for p in place_ids if not outgoing.get(p))
        if len(sources) != 1:
            issues.append(
                "the net must have exactly one start place (a place without "
                f"incoming arcs), found {len(sources)} "
                f"({', '.join(sources) or 'none'})"
            )
        if len(sinks) != 1:
            issues.append(
                "the net must have exactly one end place (a place without "
                f"outgoing arcs), found {len(sinks)} "
                f"({', '.join(sinks) or 'none'})"
            )

        # The single marking rule: exactly one place carries <initialMarking>,
        # its value is exactly one token, and it is the structural start
        # place. One start implies one marking, so the whole rule lives here.
        marked = {
            place.get("id"): _child_text(place, "initialMarking")
            for place in _iter_local(root, "place")
            if _child_text(place, "initialMarking") is not None
        }
        if len(marked) != 1:
            issues.append(
                "expected exactly one place with an <initialMarking>, found "
                f"{len(marked)}; exactly the start place must carry "
                "<initialMarking><text>1</text></initialMarking>"
            )
        else:
            (marked_id, marking_text), = marked.items()
            if marking_text != "1":
                issues.append(
                    f"initialMarking of start place '{marked_id}' must be "
                    f"exactly 1, found '{marking_text}'"
                )
            if incoming.get(marked_id):
                issues.append(
                    f"place '{marked_id}' carries the initial marking but "
                    "has incoming arcs; the marking must sit on the start "
                    "place — either the marking is on the wrong place or the "
                    f"incoming arcs of '{marked_id}' are wrong"
                )

        # Invariant 5: every transition takes part in the flow (mirrors
        # validate_pnml_connectivity in t2p-2.0).
        for tid in sorted(transition_ids):
            for direction, degree in (
                ("incoming", incoming),
                ("outgoing", outgoing),
            ):
                if not degree.get(tid):
                    issues.append(
                        f"transition '{tid}' has no {direction} arc; every "
                        "transition needs incoming and outgoing arcs — "
                        "connect it to the flow or remove it if it is not "
                        "part of the process"
                    )

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
                    f"node '{node}' lies on no path from the start place to "
                    "the end place; every node must take part in the flow — "
                    "connect it or remove it"
                )

        return issues
