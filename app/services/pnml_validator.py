"""Validator for directly generated PNML.

Structural counterpart of ``ModelValidator`` on the standard BPMN-JSON path.
``check()`` answers three questions in one pass, in this order:

1. **Gates**: is this a PNML document at all? Valid XML with a ``<pnml>``
   root and exactly one ``<net>``. Nothing else can be said otherwise.
2. **Normalization**: deterministic clean-up of everything that carries no
   P/T-net semantics: ``<graphics>``, ``<toolspecific>``, ``<inscription>``,
   unexpected elements, ``<net>`` attributes, empty place names. ``<page>``
   is unwrapped rather than dropped: it is a container, and its children are
   the net. Every clean-up is reported in ``stripped``.
3. **Structural checks**: the contract invariants that make the document a
   workflow net: transition labels, resolvable and unique ids, bipartiteness,
   no duplicate or self arcs, exactly one start and one end place, the single
   one-token initial marking on the start place, transition connectivity, and
   every node on a path from start to end. Reported in ``issues``.

Normalization is NOT the heuristic sanitize step this validator deliberately
omits: it never invents structure. It only removes what the structural checks
do not read, so it cannot mask a real defect. Where a fix would require
guessing the model's intent (duplicate parallel arcs, self-loops, dangling
nodes), the finding stays an issue and is corrected by the LLM instead.

Only ``issues`` warrant a correction pass; ``stripped`` is a record of the
model's cosmetic habits, paid for with zero LLM calls.

Gating is limited to what genuinely invalidates further checking; everything
else is collected into ONE combined issue list so a single correction pass can
fix as many real issues as possible. Behavioural soundness IS in scope, but only
once the structure is clean: a bounded token game (:meth:`_check_soundness`) then
reports transitions that can never fire, places that accumulate tokens, and runs
that cannot complete -- the deadlocks that pass every structural check yet leave
the net unrunnable. Beyond a short time budget it stays silent rather than guess.

Issue messages name the offending element and state the violated rule, and
stop there: naming a repair would point the correction at one of several
possible causes, and the model is better placed to pick between them. What a
finding may carry is a ``hint``: not the repair, but the constraint the repair
must respect, so a correction pass cannot fix one rule by breaking another.
Hints live on the check that produces them and reach the LLM only when that
check fired; see :class:`Issue`.

Findings name the net, not the document: a node by its id, an arc by its two
endpoints. Both representations of the direct path -- the PNML the model wrote
itself, and the JSON net this service serialized -- declare exactly those, so
one finding is actionable in either. Only the gates and the prerequisites,
which no JSON net can violate, speak of elements and attributes.

XML namespaces are tolerated throughout (matching the namespace-agnostic
downstream parsers): elements are matched by local name.
"""

import time
import xml.etree.ElementTree as ET
from collections import namedtuple

_PT_NET_TYPE = "http://www.informatik.hu-berlin.de/top/pntd/ptNetb"
#: Ordered, because NetCounts mirrors this order.
_NET_CHILDREN = ("place", "transition", "arc")
_ALLOWED_NET_CHILDREN = set(_NET_CHILDREN)

# Removed wherever they appear: none of the structural checks read them, so
# dropping them cannot hide a defect. <page> is handled separately: it is a
# container whose children are the net itself.
_STRIPPED_ELEMENTS = {
    "graphics": "the output is geometry-free; layouting happens downstream",
    "toolspecific": "native standard PNML only, no tool-specific blocks",
    "inscription": "arc inscriptions are not part of the contract",
}

class Issue(str):
    """A structural finding, and the constraints a correction must respect.

    The string is the finding the model reads; ``hints`` are the sentences the
    repair prompt adds when this check fired, and nothing when it did not. They
    sit here rather than in the prompt because a rule and the trap that comes
    with fixing it belong together: neither can be improved without the other
    coming along.

    Subclassing ``str`` keeps a finding a finding: it joins into the response
    header, serializes into the debug view and compares between passes exactly
    as the plain strings the gates still return.
    """

    def __new__(cls, text, *hints):
        issue = super().__new__(cls, text)
        issue.hints = hints
        return issue


def hints_for(issues):
    """The hints of the findings at hand, deduplicated, in order of appearance.

    Plain strings carry none, so a gate finding contributes nothing.
    """
    return list(
        dict.fromkeys(
            hint for issue in issues for hint in getattr(issue, "hints", ())
        )
    )


#: Both directions of the alternation rule are one trap: a node is reconnected
#: through a neighbour of the other kind, and where none exists it must be
#: created. Without this, a correction pass satisfies "connect p_x" by drawing
#: an arc to the nearest place and reports back a net with a new defect.
_HINT_ALTERNATION = (
    "Arcs alternate between places and transitions, so a place is reconnected "
    "from a transition and a transition from a place: never join two places or "
    "two transitions to attach one. Where the flow offers no legal neighbour, "
    "insert the transition or place the connection needs; that insertion is a "
    "minimal change, not an addition to the process."
)

#: The one node kind that must be deleted rather than reconnected. Named only
#: when a node is actually unreachable or disconnected, because sent
#: unprompted it invites the deletion of the ordinary post-activity places
#: (p_analyzed, p_geprueft) that look superficially alike.
_HINT_CONDITION_OUTCOME = (
    "If a reported node names the outcome of a condition ('approved', "
    "'authenticated', 'passed', 'retry limit exceeded'), the net has no room "
    "for it: the outcome is which transition fires at the choice place. Delete "
    "it and route the alternatives from that place instead. A place that "
    "merely follows an activity is the normal case and stays."
)

#: The one trap of the soundness checks: a transition joins what is not parallel.
#: A loop back or an alternative merged into a transition makes it wait forever
#: for a token that only ever arrives on one of its inputs. Sent whenever the
#: token game finds a transition that can never fire or a run that cannot finish.
_HINT_SYNC_MERGE = (
    "A transition fires only when every incoming arc carries a token at the same "
    "moment. A branch that is an alternative (only one runs) or a loop back must "
    "therefore rejoin at the PLACE that feeds the transition on its forward path, "
    "giving that transition a single incoming arc -- never point the loop or the "
    "alternative into the transition itself. Only genuinely parallel branches, "
    "split off earlier by one transition, may be joined at a transition."
)

#: For a place that collects more than one token: the mirror of the above.
_HINT_ACCUMULATE = (
    "Alternatives must not both feed one place either -- they rejoin at a place "
    "that then continues once. Parallel branches must synchronize at a joining "
    "transition that consumes one token from each; a place that several branches "
    "deposit into just piles the tokens up and never clears."
)

#: A surplus end place is almost never a second real ending; it is usually a
#: step whose continuation was dropped (the happy path that stops where it
#: should hand on). Named only for extra sinks, so the fix is to reconnect, not
#: to merge distinct outcomes or delete the place to satisfy the count.
_HINT_DEAD_END = (
    "A place with incoming but no outgoing arcs stops the flow there. If that is "
    "not where the process ends, a continuation is missing: route it onward to the "
    "step that actually follows it. Only a branch that genuinely ends belongs on "
    "the single end place -- do not send every dead end straight to the end, and "
    "do not delete the place just to satisfy the count."
)

#: Size of the net. Zero throughout when the document did not pass the gates.
NetCounts = namedtuple("NetCounts", "places transitions arcs")

#: Result of :meth:`PnmlValidator.check`.
#:
#: ``pnml``     -- the normalized document (unchanged string when nothing was
#:                 stripped, so well-formed input keeps its original layout)
#: ``stripped`` -- human-readable record of the deterministic clean-ups
#: ``issues``   -- structural defects; non-empty means a correction is due
#: ``counts``   -- the net's size, so callers need not parse the XML again to
#:                 tell whether a correction pass shrank the net
PnmlCheck = namedtuple("PnmlCheck", "pnml stripped issues counts")


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


def _remove_everywhere(root, predicate):
    """Remove every element matching ``predicate``. Returns the count.

    ElementTree has no parent pointers, so the parent map is rebuilt after
    each sweep: removing a subtree invalidates it for the nested matches.
    """
    removed = 0
    while True:
        matches = [
            (parent, child)
            for parent in root.iter()
            for child in parent
            if predicate(child)
        ]
        if not matches:
            return removed
        for parent, child in matches:
            parent.remove(child)
        removed += len(matches)


def _serialize(root):
    """Render the (possibly mutated) tree back to a PNML document string."""
    if root.tag.startswith("{"):
        ET.register_namespace("", root.tag[1:].split("}", 1)[0])
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
        root, encoding="unicode"
    )


class PnmlValidator:
    """Normalize and validate directly generated PNML XML."""

    def check(self, pnml_xml):
        """Normalize the document and report its structural defects."""
        gate_issues, root = self._check_valid_xml(pnml_xml)
        if gate_issues:
            return PnmlCheck(pnml_xml, [], gate_issues, NetCounts(0, 0, 0))

        gate_issues = self._check_document_skeleton(root)
        if gate_issues:
            return PnmlCheck(pnml_xml, [], gate_issues, NetCounts(0, 0, 0))

        stripped = self._normalize(root)
        pnml = _serialize(root) if stripped else pnml_xml
        net = next(_iter_local(root, "net"))
        counts = NetCounts(
            *(len(list(_iter_local(net, kind))) for kind in _NET_CHILDREN)
        )

        prerequisite_issues = self._check_graph_prerequisites(net)
        if prerequisite_issues:
            return PnmlCheck(pnml, stripped, prerequisite_issues, counts)

        structure_issues = self._check_graph_structure(net)
        if structure_issues:
            return PnmlCheck(pnml, stripped, structure_issues, counts)

        # The graph is a well-formed workflow net; only now can the token game run.
        return PnmlCheck(pnml, stripped, self._check_soundness(net), counts)

    # --- Gates ---------------------------------------------------------------

    @staticmethod
    def _check_valid_xml(pnml_xml):
        """The output must parse as XML. Returns (issues, root)."""
        if not isinstance(pnml_xml, str) or not pnml_xml.strip():
            return ["output is not valid XML: document is empty"], None
        try:
            return [], ET.fromstring(pnml_xml)
        except ET.ParseError as exc:
            return [f"output is not valid XML: {exc}"], None

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

    # --- Normalization (deterministic, never guesses) -------------------------

    @classmethod
    def _normalize(cls, root):
        """Strip everything without P/T-net semantics. Returns a record."""
        stripped = []
        net = next(_iter_local(root, "net"))

        pages = cls._unwrap_pages(net)
        if pages:
            stripped.append(f"unwrapped {pages} <page> container(s) into <net>")

        for name, reason in _STRIPPED_ELEMENTS.items():
            count = _remove_everywhere(
                root, lambda child, n=name: _local_name(child.tag) == n
            )
            if count:
                stripped.append(f"removed {count} <{name}> element(s) ({reason})")

        unexpected = [
            child
            for child in net
            if _local_name(child.tag) not in _ALLOWED_NET_CHILDREN
        ]
        for child in unexpected:
            net.remove(child)
        if unexpected:
            names = ", ".join(sorted({_local_name(c.tag) for c in unexpected}))
            stripped.append(
                f"removed {len(unexpected)} unexpected element(s) under <net> "
                f"({names}); allowed are <place>, <transition> and <arc>"
            )

        # Intermediate places may be nameless per contract, so an empty <name>
        # is noise rather than a defect. Transitions are checked, not stripped.
        empty_names = 0
        for place in _iter_local(net, "place"):
            for child in list(place):
                if _local_name(child.tag) == "name" and not _child_text(
                    place, "name"
                ):
                    place.remove(child)
                    empty_names += 1
        if empty_names:
            stripped.append(
                f"removed {empty_names} empty <name> element(s) from places"
            )

        if not net.get("id"):
            net.set("id", "noID")
            stripped.append("set the missing <net> 'id' attribute to 'noID'")
        if net.get("type") != _PT_NET_TYPE:
            net.set("type", _PT_NET_TYPE)
            stripped.append(f"set the <net> 'type' attribute to '{_PT_NET_TYPE}'")

        return stripped

    @staticmethod
    def _unwrap_pages(net):
        """Hoist the children of every <page> into <net>. Returns the count."""
        unwrapped = 0
        while True:
            pages = [
                child for child in net if _local_name(child.tag) == "page"
            ]
            if not pages:
                return unwrapped
            for page in pages:
                index = list(net).index(page)
                net.remove(page)
                for offset, child in enumerate(list(page)):
                    net.insert(index + offset, child)
            unwrapped += len(pages)

    # --- Structural checks ----------------------------------------------------

    @staticmethod
    def _check_graph_prerequisites(net):
        """Ids and arc endpoints the graph checks depend on.

        Gates the graph structure checks: without them the graph is not
        well-defined and the findings below would be artifacts of these
        defects.
        """
        issues = []

        seen_ids = set()
        for kind in _NET_CHILDREN:
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

    @staticmethod
    def _check_graph_structure(net):
        """The workflow-net invariants on the graph.

        Runs only when the graph prerequisites passed, so ids exist and are
        unique and every arc carries source and target attributes.
        """
        issues = []

        place_ids = {p.get("id") for p in _iter_local(net, "place")}
        transition_ids = {t.get("id") for t in _iter_local(net, "transition")}
        node_ids = place_ids | transition_ids

        edges = []
        incoming = {}
        outgoing = {}
        seen_connections = set()
        for arc in _iter_local(net, "arc"):
            source = arc.get("source")
            target = arc.get("target")
            # Named by its endpoints, never by its id: on the JSON path the id
            # is derived during serialization, so the model would be sent
            # looking for an 'a7' it never wrote.
            arc_name = f"the arc from '{source}' to '{target}'"

            if source == target:
                issues.append(
                    f"{arc_name} connects a node to itself; arcs must connect "
                    "two different nodes"
                )
            unresolved = False
            for role, ref in (("source", source), ("target", target)):
                if ref not in node_ids:
                    issues.append(
                        f"{arc_name} references unknown {role} '{ref}'; every "
                        "arc must reference an existing place or transition"
                    )
                    unresolved = True
            if unresolved:
                continue

            if source in place_ids and target in place_ids:
                issues.append(
                    Issue(
                        f"{arc_name} connects two places; arcs must alternate "
                        "between places and transitions",
                        _HINT_ALTERNATION,
                    )
                )
            elif source in transition_ids and target in transition_ids:
                issues.append(
                    Issue(
                        f"{arc_name} connects two transitions; arcs must "
                        "alternate between places and transitions",
                        _HINT_ALTERNATION,
                    )
                )

            # A second identical arc reads as an arc weight of 2 downstream;
            # removing it silently would guess at the intent, so it stays an
            # issue for the LLM to resolve.
            if (source, target) in seen_connections:
                issues.append(
                    f"{arc_name} occurs twice; only one arc per direction is "
                    "allowed between two nodes"
                )
            seen_connections.add((source, target))

            edges.append((source, target))
            outgoing.setdefault(source, set()).add(target)
            incoming.setdefault(target, set()).add(source)

        # A node with no arcs at all is disconnected. Report it once and
        # plainly: otherwise a single isolated place surfaces confusingly as
        # both a spurious extra start place and a spurious extra end place, and
        # a correction pass told only "2 start places" cannot tell what to
        # reconnect.
        disconnected = {
            n for n in node_ids if not incoming.get(n) and not outgoing.get(n)
        }
        for node in sorted(disconnected):
            kind = "place" if node in place_ids else "transition"
            issues.append(
                Issue(
                    f"{kind} '{node}' has no arcs at all; it is disconnected "
                    "from the flow, so connect it to the process or remove it",
                    _HINT_ALTERNATION,
                    _HINT_CONDITION_OUTCOME,
                )
            )

        # Start and end places exclude the disconnected ones, so an isolated
        # node is not also miscounted as a second start and a second end.
        sources = sorted(
            p for p in place_ids if not incoming.get(p) and p not in disconnected
        )
        sinks = sorted(
            p for p in place_ids if not outgoing.get(p) and p not in disconnected
        )
        if len(sources) != 1:
            issues.append(
                "the net must have exactly one start place (a place without "
                f"incoming arcs), found {len(sources)} "
                f"({', '.join(sources) or 'none'})"
            )
        if len(sinks) != 1:
            issues.append(
                Issue(
                    "the net must have exactly one end place (a place without "
                    f"outgoing arcs), found {len(sinks)} "
                    f"({', '.join(sinks) or 'none'})",
                    *((_HINT_DEAD_END,) if len(sinks) > 1 else ()),
                )
            )

        # The single marking rule: exactly one place carries <initialMarking>,
        # its value is exactly one token, and it is the structural start
        # place. One start implies one marking, so the whole rule lives here.
        #
        # Gated on a unique start place, because the marking follows from it.
        # Reported without one, the finding is unactionable: on the XML path it
        # points at a symptom whose cause is already listed above, and on the
        # JSON path the model cannot act on it at all -- its schema has no
        # marking, this service derives it, and it derives none precisely when
        # the start place is ambiguous.
        if len(sources) == 1:
            marked = {
                place.get("id"): _child_text(place, "initialMarking")
                for place in _iter_local(net, "place")
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
                        "has incoming arcs; the initial marking must sit on the "
                        "start place"
                    )

        # Every transition takes part in the flow (mirrors
        # validate_pnml_connectivity in t2p-2.0). Fully isolated transitions
        # are already reported as disconnected above, so skip them here to
        # avoid a redundant second message.
        for tid in sorted(transition_ids):
            if tid in disconnected:
                continue
            for direction, degree in (
                ("incoming", incoming),
                ("outgoing", outgoing),
            ):
                if not degree.get(tid):
                    issues.append(
                        Issue(
                            f"transition '{tid}' has no {direction} arc; every "
                            "transition needs at least one incoming and one "
                            "outgoing arc",
                            _HINT_ALTERNATION,
                        )
                    )

        # A transition without a <name> is a silent transition, a standard
        # workflow-net construct (van der Aalst's routing "control tasks",
        # WoPeD's silent transitions). It is never required to carry a label:
        # forcing one would push the model to invent an activity the text does
        # not describe. Labelling activities is encouraged in the prompt, not
        # enforced here.

        # Every node lies on a path from source to sink. Needs an unambiguous
        # source and sink; their absence is already reported above.
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
            for node in sorted(node_ids - (reachable & co_reachable) - disconnected):
                issues.append(
                    Issue(
                        f"node '{node}' lies on no path from the start place to "
                        "the end place; every node must lie on such a path",
                        _HINT_ALTERNATION,
                        _HINT_CONDITION_OUTCOME,
                    )
                )

        return issues

    # --- Behavioural soundness (bounded token game) --------------------------

    #: The token cap makes the reachable state space finite, so the analysis
    #: always terminates on its own; the only bound that matters in the request
    #: path is wall-clock time. A net whose exploration runs past the budget
    #: (heavy concurrency) is left unjudged rather than allowed to stall the
    #: request. The 1-safe cap additionally signals accumulation.
    _SOUNDNESS_TIME_LIMIT_S = 2.0
    _SOUNDNESS_TOKEN_CAP = 2

    @classmethod
    def _check_soundness(cls, net):
        """Whether the net can actually run: reach the end, one token, no stall.

        A structurally valid workflow net can still be unrunnable. The recurring
        case: a transition that joins a loop back or an alternative at itself
        waits for a token on every incoming arc at once; that token never arrives
        together with the others, so the transition never fires and the flow
        stalls before the end. Nothing structural sees this -- the graph is a
        proper workflow net -- but the token game does.

        Plays the game from one token on the start place under hard bounds, then
        reports only what it can prove: transitions that can never fire, places
        that accumulate more than one token, and runs that can no longer reach
        the end. Beyond a short time budget it stays silent rather than guess. This
        is the one check that looks past structure at behaviour; its findings, like
        the others, name the net and travel with a hint into the correction pass.
        """
        places = {p.get("id") for p in _iter_local(net, "place")}
        transitions = {t.get("id") for t in _iter_local(net, "transition")}
        incoming, outgoing = {}, {}
        for arc in _iter_local(net, "arc"):
            outgoing.setdefault(arc.get("source"), []).append(arc.get("target"))
            incoming.setdefault(arc.get("target"), []).append(arc.get("source"))

        starts = [p for p in places if p not in incoming]
        ends = [p for p in places if p not in outgoing]
        if len(starts) != 1 or len(ends) != 1:
            return []  # the structure checks own this; the game needs one of each
        start, end = starts[0], ends[0]
        tinp = {t: incoming.get(t, []) for t in transitions}
        tout = {t: outgoing.get(t, []) for t in transitions}

        def key(marking):
            return tuple(sorted((p, c) for p, c in marking.items() if c > 0))

        final = ((end, 1),)
        seen = {key({start: 1})}
        adjacency = {}
        stack = [{start: 1}]
        fired_ever = set()
        marked_places = {start}
        accumulating = set()
        improper = False
        truncated = False
        deadline = time.monotonic() + cls._SOUNDNESS_TIME_LIMIT_S
        while stack:
            if time.monotonic() > deadline:
                truncated = True
                break
            marking = stack.pop()
            successors = adjacency.setdefault(key(marking), [])
            for t in transitions:
                ins = tinp[t]
                if not ins or any(marking.get(p, 0) < 1 for p in ins):
                    continue
                fired_ever.add(t)
                nxt = dict(marking)
                for p in ins:
                    nxt[p] -= 1
                for p in tout[t]:
                    nxt[p] = nxt.get(p, 0) + 1
                over = [p for p, c in nxt.items() if c >= cls._SOUNDNESS_TOKEN_CAP]
                for p in over:
                    accumulating.add(p)
                    nxt[p] = cls._SOUNDNESS_TOKEN_CAP
                if nxt.get(end, 0) >= 1 and any(c > 0 for p, c in nxt.items() if p != end):
                    improper = True
                marked_places.update(p for p, c in nxt.items() if c > 0)
                child = key(nxt)
                successors.append(child)
                if child not in seen:
                    seen.add(child)
                    stack.append(nxt)

        if truncated:
            return []  # too large to judge; silence beats a false alarm

        # Which markings can still reach the clean final marking (option to complete)?
        can_complete = set()
        if final in seen:
            reverse = {}
            for parent, children in adjacency.items():
                for child in children:
                    reverse.setdefault(child, set()).add(parent)
            frontier = [final]
            can_complete.add(final)
            while frontier:
                for parent in reverse.get(frontier.pop(), ()):
                    if parent not in can_complete:
                        can_complete.add(parent)
                        frontier.append(parent)

        issues = []
        dead = sorted(t for t in transitions if tinp[t] and t not in fired_ever)
        # The stall points -- dead transitions a token actually reaches on some but
        # never all inputs -- are the offending merges; the rest are their downstream
        # victims. Report the stalls, naming the input that never arrives.
        stalls = [
            (t, [p for p in tinp[t] if p not in marked_places])
            for t in dead
            if any(p in marked_places for p in tinp[t])
        ]
        for t, missing in stalls[:3]:
            never = ", ".join(f"'{p}'" for p in missing) or "one of its inputs"
            issues.append(
                Issue(
                    f"transition '{t}' can never fire: a token reaches it on some "
                    f"incoming arcs but never on {never} at the same time, so the "
                    "flow stalls here",
                    _HINT_SYNC_MERGE,
                )
            )
        if not stalls:
            for t in dead[:2]:
                issues.append(
                    Issue(
                        f"transition '{t}' can never fire; no run ever enables it",
                        _HINT_SYNC_MERGE,
                    )
                )
        for p in sorted(accumulating)[:4]:
            issues.append(
                Issue(
                    f"place '{p}' can hold more than one token at once: several "
                    "branches deposit into it instead of synchronizing at a "
                    "transition",
                    _HINT_ACCUMULATE,
                )
            )
        if not issues and improper:
            issues.append(
                Issue(
                    "a run can mark the end place while other places still hold "
                    "tokens; a completed run leaves exactly one token, on the end "
                    "place",
                    _HINT_ACCUMULATE,
                )
            )
        if not issues and final not in seen:
            issues.append(
                Issue(
                    "no run of the net reaches the end place; the flow cannot "
                    "complete",
                    _HINT_SYNC_MERGE,
                )
            )
        elif not issues and (seen - can_complete):
            issues.append(
                Issue(
                    "the net can reach a state from which the end place can no "
                    "longer be marked; a run can get stuck before completing",
                    _HINT_SYNC_MERGE,
                )
            )
        return issues
