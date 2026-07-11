"""The JSON dialect of the direct-PNML path: schema, extraction, serialization.

The XML path asks the model for the PNML document itself. This path asks for a
JSON object under a provider-enforced schema and builds the PNML here, which
removes a class of defects by construction rather than by validation: the
document is always well-formed, arc ids and the initial marking are derived
rather than transcribed, and no ``<graphics>``/``<toolspecific>`` noise can
appear.

The net is described *once*, by the ``arcs``. The ``nodes`` list is a glossary,
not a second declaration of what exists: it annotates an id with its kind and
label, and an entry no arc references is never emitted. This is what makes the
two defects that used to dominate this path impossible rather than merely
reported:

* a **disconnected node** cannot occur, because a node exists only by appearing
  in an arc -- nothing here iterates the glossary to create one; and
* a **duplicate id** cannot occur, because the node set is the arc endpoints
  (each emitted once) and the glossary is read as a first-wins lookup.

That is the line the construction draws. Everything the net's *semantics* decide
stays the validator's business: an arc between two places, a transition without
an incoming arc, a node no path reaches. Repairing those would mean guessing
intent, so they remain defects :mod:`pnml_validator` reports, exactly as on the
XML path.

The two paths therefore share one validator, one correction loop and one
response contract; only format, parser and repair prompt differ.

:class:`PetriNet` is that schema, as a Pydantic model rather than a hand-written
JSON schema, because both providers compile it themselves: OpenAI into its strict
JSON schema, google-genai into its own ``Schema``. One declaration, two enforced
dialects, no translation to keep in sync -- and the same field order on both.
The model is a *schema*, not a parser: replies are still read as plain dicts, so
a provider that ignores the schema is still handled rather than trusted.
"""

import json
import re
import xml.etree.ElementTree as ET
from typing import Literal, Optional

from pydantic import BaseModel

_PT_NET_TYPE = "http://www.informatik.hu-berlin.de/top/pntd/ptNetb"

#: The two node kinds a glossary entry may carry.
_KINDS = ("place", "transition")

#: First balanced-looking JSON object in a reply, for providers that wrap the
#: object in prose or a markdown fence despite the schema. Kept as a safety net:
#: a model whose provider does not enforce the schema still has to be parsed.
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class Node(BaseModel):
    """A glossary entry: an id, whether it is a place or a transition, its label.

    This is not a declaration that the node exists -- the ``arcs`` decide that.
    It is the metadata the arcs cannot carry: which of the two kinds the id is,
    and its label, which is ``null`` when the node merely routes.

    ``kind`` and ``name`` both carry NO default, on purpose. OpenAI's strict mode
    forces every property into ``required``, so a nullable type (not a default)
    is the only way to say "may have no label" while keeping the two providers'
    ``required`` lists identical; google-genai would otherwise honour a default
    and let Gemini omit the key entirely.
    """

    id: str
    kind: Literal["place", "transition"]
    name: Optional[str]


class Arc(BaseModel):
    """A directed edge between two nodes, referenced by their ids."""

    source: str
    target: str


class PetriNet(BaseModel):
    """The provider-enforced answer shape, and the one source both providers read.

    OpenAI compiles it into a strict JSON schema (``responses``/``chat`` parse
    helpers); google-genai turns the same class into its own ``Schema``. Neither
    translation is written by hand, so the two cannot drift apart.

    Field order matters and survives both compilers -- OpenAI generates in
    schema order, google-genai emits ``property_ordering`` from the field order.
    The model therefore fills the object in this order: the plan (``activities``,
    ``branches``) before the ``arcs``, and the arcs before the ``nodes`` glossary
    that labels them. That is the order the prompt asks for, and it is why a node
    cannot be invented before the flow that would have to reach it. Reordering
    these fields silently removes the model's scratchpad.

    ``activities`` and ``branches`` are never read: they exist so the model
    commits to a plan while it still can act on one.
    """

    activities: list[str]
    branches: list[str]
    arcs: list[Arc]
    nodes: list[Node]


def extract_json_document(text):
    """Return the JSON object from a model reply, or ``None``.

    Under a strict schema the reply is already bare JSON; the regex fallback
    only covers providers that ignore the schema and wrap it anyway.
    """
    if not text:
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        match = _JSON_BLOCK_RE.search(text)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except ValueError:
            return None
    return payload if isinstance(payload, dict) else None


def _entries(payload, key):
    """The list under ``key``, tolerating a model that omitted or mistyped it."""
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _arc_pairs(payload):
    """``(source, target)`` for every well-formed arc entry."""
    for entry in _entries(payload, "arcs"):
        if not isinstance(entry, dict):
            continue
        source, target = entry.get("source"), entry.get("target")
        if isinstance(source, str) and isinstance(target, str):
            yield source, target


def _node_glossary(payload):
    """``id -> (kind, name)`` from the ``nodes`` list; the first entry of an id wins.

    A glossary, not a bill of existence: it is only ever consulted for ids the
    arcs already introduced. Reading it first-wins is what makes a repeated id
    resolve to one node instead of reaching the document twice. ``kind`` is kept
    only when it is one of the two valid values, ``name`` only when non-blank;
    either may be absent and is then supplied by :func:`_resolve_kinds` or left
    unnamed.
    """
    glossary = {}
    for entry in _entries(payload, "nodes"):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        node_id = entry["id"]
        if node_id in glossary:
            continue
        kind = entry.get("kind")
        name = entry.get("name")
        glossary[node_id] = (
            kind if kind in _KINDS else None,
            name if isinstance(name, str) and name.strip() else None,
        )
    return glossary


def _ordered_node_ids(arcs):
    """The ids the arcs reference, in first-appearance order and without repeats.

    This is the net's node set: a node exists exactly when an arc touches it.
    """
    seen = {}
    for source, target in arcs:
        seen.setdefault(source)
        seen.setdefault(target)
    return list(seen)


def _other_kind(kind):
    return "transition" if kind == "place" else "place"


def _resolve_kinds(node_ids, arcs, glossary):
    """Assign every arc-referenced node a kind: from the glossary, else derived.

    A node the glossary types is taken at its word. One it omits is typed from
    the net itself: arcs alternate between the two kinds, so a neighbour of a
    typed node is the opposite kind, propagated to a fixpoint. Whatever that
    still leaves open -- a component the glossary never anchors -- falls back to
    the id convention (``t`` for transitions, else a place), which the prompt
    teaches and the ids follow.
    """
    kinds = {
        node_id: glossary[node_id][0]
        for node_id in node_ids
        if glossary.get(node_id, (None, None))[0] is not None
    }

    changed = True
    while changed:
        changed = False
        for source, target in arcs:
            for known, other in ((source, target), (target, source)):
                if known in kinds and other not in kinds:
                    kinds[other] = _other_kind(kinds[known])
                    changed = True

    for node_id in node_ids:
        kinds.setdefault(node_id, "transition" if node_id.startswith("t") else "place")
    return kinds


def _add_node(net, tag, node_id, name):
    """Append ``<place id=..>`` or ``<transition id=..>``, named only if it has one."""
    element = ET.SubElement(net, tag, {"id": node_id})
    if name:
        ET.SubElement(ET.SubElement(element, "name"), "text").text = name
    return element


def _start_place(place_ids, arcs):
    """The one place no arc points at, or None when it is not unique."""
    targets = {target for _, target in arcs}
    candidates = [place_id for place_id in place_ids if place_id not in targets]
    return candidates[0] if len(candidates) == 1 else None


def pnml_from_json(payload):
    """Build a PNML document from the model's JSON net.

    The ``arcs`` are the net: the nodes are their endpoints, each emitted once,
    typed and labelled from the ``nodes`` glossary. A glossary entry no arc
    references contributes nothing, so a disconnected node cannot be written; a
    repeated id resolves to one node, so a duplicate id cannot be written.

    Arc ids and the initial marking are likewise derived, not transcribed. Where
    the start place is not unique the marking is omitted: that is one defect, and
    the validator names it once. Inventing a marking would add a second.
    """
    arcs = list(_arc_pairs(payload))
    glossary = _node_glossary(payload)
    node_ids = _ordered_node_ids(arcs)
    kinds = _resolve_kinds(node_ids, arcs, glossary)

    place_ids = [node_id for node_id in node_ids if kinds[node_id] == "place"]
    transition_ids = [node_id for node_id in node_ids if kinds[node_id] == "transition"]
    start_place = _start_place(place_ids, arcs)

    root = ET.Element("pnml")
    net = ET.SubElement(root, "net", {"id": "noID", "type": _PT_NET_TYPE})

    for place_id in place_ids:
        _, name = glossary.get(place_id, (None, None))
        place = _add_node(net, "place", place_id, name)
        if place_id == start_place:
            ET.SubElement(ET.SubElement(place, "initialMarking"), "text").text = "1"

    for transition_id in transition_ids:
        _, name = glossary.get(transition_id, (None, None))
        _add_node(net, "transition", transition_id, name)

    for index, (source, target) in enumerate(arcs, start=1):
        ET.SubElement(net, "arc", {"id": f"a{index}", "source": source, "target": target})

    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")
