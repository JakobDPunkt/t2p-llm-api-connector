"""The JSON dialect of the direct-PNML path.

What is worth testing here is the claim the path is built on: everything the
serializer derives, the model can no longer get wrong. Everything the net's
semantics decide is still the validator's business, and must survive.
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import create_app
from app.services.llm_service import PNML_FORMATS, LLMService
from app.services.pnml_json import (
    Arc,
    Node,
    PetriNet,
    extract_json_document,
    pnml_from_json,
)
from app.services.pnml_validator import PnmlValidator
from config import BaseConfig, TestingConfig

MINIMAL = {
    "activities": ["check order"],
    "branches": [],
    "arcs": [
        {"source": "p_start", "target": "t_check_order"},
        {"source": "t_check_order", "target": "p_end"},
    ],
    "nodes": [
        {"id": "p_start", "kind": "place", "name": "start"},
        {"id": "t_check_order", "kind": "transition", "name": "check order"},
        {"id": "p_end", "kind": "place", "name": "end"},
    ],
}


class TestSchema(unittest.TestCase):
    """One Pydantic class, two providers. What must hold for them to agree.

    Both SDKs compile the class themselves, so the compilers are not under test
    here -- the properties of the class that make their outputs equivalent are.
    """

    def test_no_field_carries_a_default(self):
        # The one way to make the two providers disagree silently. OpenAI's
        # strict mode forces every field into "required" regardless; google-genai
        # honours a default and drops the field from "required". Gemini could
        # then omit a key OpenAI must always send.
        for model in (PetriNet, Node, Arc):
            for name, field in model.model_fields.items():
                self.assertTrue(
                    field.is_required(),
                    f"{model.__name__}.{name} has a default; the two providers "
                    "would compile different 'required' lists",
                )

    def test_field_order_puts_the_plan_before_the_arcs(self):
        # Both compilers preserve this order (OpenAI generates in schema order,
        # google-genai emits property_ordering from it), so it is the model's
        # scratchpad: reorder these and the plan stops governing the arcs.
        self.assertEqual(
            list(PetriNet.model_fields),
            ["activities", "branches", "arcs", "nodes"],
        )
        self.assertEqual(list(Node.model_fields), ["id", "kind", "name"])

    def test_routing_nodes_are_expressible_as_nameless(self):
        self.assertIsNone(Node(id="t_split", kind="transition", name=None).name)

    def test_the_schema_is_not_the_parser(self):
        # A provider that ignores the schema still has to be handled: replies
        # are read as plain dicts, never as validated PetriNet instances.
        self.assertIsInstance(extract_json_document(json.dumps(MINIMAL)), dict)


class TestExtraction(unittest.TestCase):
    def test_bare_json(self):
        self.assertEqual(extract_json_document('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(extract_json_document('```json\n{"a": 1}\n```'), {"a": 1})

    def test_no_object(self):
        self.assertIsNone(extract_json_document("sorry, I cannot"))
        self.assertIsNone(extract_json_document("[1, 2]"))
        self.assertIsNone(extract_json_document(""))


class TestSerializer(unittest.TestCase):
    def check(self, payload):
        return PnmlValidator().check(pnml_from_json(payload))

    def test_minimal_net_is_valid_pnml(self):
        result = self.check(MINIMAL)
        self.assertEqual(result.issues, [])
        self.assertEqual(tuple(result.counts), (2, 1, 2))

    def test_nothing_is_stripped_because_nothing_superfluous_is_written(self):
        self.assertEqual(self.check(MINIMAL).stripped, [])

    def test_arc_ids_are_derived_not_transcribed(self):
        pnml = pnml_from_json(MINIMAL)
        self.assertIn('id="a1"', pnml)
        self.assertIn('id="a2"', pnml)

    def test_initial_marking_lands_on_the_place_without_an_incoming_arc(self):
        pnml = pnml_from_json(MINIMAL)
        start = pnml.index("p_start")
        end = pnml.index("p_end")
        marking = pnml.index("initialMarking")
        self.assertLess(start, marking)
        self.assertLess(marking, end)

    def test_no_marking_when_the_start_place_is_not_unique(self):
        """Two starts is one defect. Guessing a marking would add a second."""
        payload = dict(
            MINIMAL,
            nodes=MINIMAL["nodes"] + [{"id": "p_x", "kind": "place", "name": None}],
            arcs=MINIMAL["arcs"] + [{"source": "p_x", "target": "t_check_order"}],
        )
        pnml = pnml_from_json(payload)
        self.assertNotIn("initialMarking", pnml)
        issues = PnmlValidator().check(pnml).issues
        self.assertTrue(any("exactly one start place" in i for i in issues))

    def _with_transition_name(self, name):
        """MINIMAL with the sole activity transition relabelled in the glossary."""
        return dict(
            MINIMAL,
            nodes=[
                {"id": "p_start", "kind": "place", "name": "start"},
                {"id": "t_check_order", "kind": "transition", "name": name},
                {"id": "p_end", "kind": "place", "name": "end"},
            ],
        )

    def test_routing_transition_stays_unnamed(self):
        pnml = pnml_from_json(self._with_transition_name(None))
        self.assertIn('<transition id="t_check_order" />', pnml)
        self.assertIn("<text>start</text>", pnml)  # named places keep their label

    def test_blank_name_counts_as_unnamed(self):
        pnml = pnml_from_json(self._with_transition_name("  "))
        self.assertIn('<transition id="t_check_order" />', pnml)

    def test_labels_are_xml_escaped(self):
        pnml = pnml_from_json(self._with_transition_name("a < b & c"))
        self.assertIn("a &lt; b &amp; c", pnml)
        self.assertEqual(PnmlValidator().check(pnml).issues, [])

    def test_malformed_entries_are_skipped_not_crashed_on(self):
        payload = dict(MINIMAL, arcs=MINIMAL["arcs"] + ["nonsense", {"source": "p_x"}])
        self.assertEqual(self.check(payload).counts.arcs, 2)

    def test_missing_keys_degrade_to_an_empty_net(self):
        self.assertEqual(tuple(self.check({}).counts), (0, 0, 0))


class TestConstructionRemovesDefects(unittest.TestCase):
    """What the arc-as-net construction makes impossible, not merely reported.

    The node set is the arc endpoints, so a node with no arc cannot be written;
    the glossary is read first-wins over a set of ids, so a duplicate id cannot
    be written. Both used to be the validator's most common findings on this
    path; here they never reach the document.
    """

    def test_a_glossary_entry_no_arc_references_is_dropped(self):
        # The old failure mode: a node declared but never wired. It is now simply
        # not part of the net -- no floating node, and nothing for the validator
        # to report.
        payload = dict(
            MINIMAL,
            nodes=MINIMAL["nodes"] + [{"id": "p_orphan", "kind": "place", "name": "stray"}],
        )
        pnml = pnml_from_json(payload)
        self.assertNotIn("p_orphan", pnml)
        self.assertEqual(PnmlValidator().check(pnml).issues, [])

    def test_a_repeated_glossary_id_resolves_to_one_node(self):
        # A second entry for an existing id is ignored; the node the arcs
        # reference is emitted exactly once, so no duplicate id can occur.
        payload = dict(
            MINIMAL,
            nodes=MINIMAL["nodes"] + [{"id": "p_start", "kind": "place", "name": "other"}],
        )
        pnml = pnml_from_json(payload)
        self.assertEqual(pnml.count('id="p_start"'), 1)
        self.assertEqual(PnmlValidator().check(pnml).issues, [])

    def test_a_node_absent_from_the_glossary_is_still_typed_and_emitted(self):
        # An arc may reference an id the glossary omits. Unlike a floating node
        # this is recoverable: the node exists (an arc reaches it), its kind is
        # derived from the alternation, and it is emitted unnamed. Here it forms
        # a second sink, which the validator surfaces structurally.
        payload = dict(
            MINIMAL,
            arcs=MINIMAL["arcs"] + [{"source": "t_check_order", "target": "p_ghost"}],
        )
        pnml = pnml_from_json(payload)
        self.assertIn('<place id="p_ghost" />', pnml)  # typed as a place, unnamed
        issues = PnmlValidator().check(pnml).issues
        self.assertTrue(any("p_ghost" in i for i in issues))


class TestSemanticDefectsStillSurface(unittest.TestCase):
    """The serializer must not paper over what the net's semantics got wrong."""

    def test_place_to_place_arc_is_reported(self):
        payload = dict(
            MINIMAL,
            arcs=[{"source": "p_start", "target": "p_end"}],
            nodes=[
                {"id": "p_start", "kind": "place", "name": "start"},
                {"id": "p_end", "kind": "place", "name": "end"},
            ],
        )
        issues = PnmlValidator().check(pnml_from_json(payload)).issues
        self.assertTrue(any("alternate between places and transitions" in i for i in issues))


class TestFormatsShareOneContract(unittest.TestCase):
    def test_json_format_parses_a_reply_into_pnml(self):
        pnml = PNML_FORMATS["json"].to_pnml(json.dumps(MINIMAL))
        self.assertEqual(PnmlValidator().check(pnml).issues, [])

    def test_xml_format_takes_the_reply_as_the_document(self):
        # Nothing is extracted: the model was asked for the document itself, so
        # whatever it wrote is what the validator judges.
        reply = "<pnml><net/></pnml>"
        self.assertEqual(PNML_FORMATS["xml"].to_pnml(reply), reply)

    def test_unusable_replies_yield_none_on_both_paths(self):
        # The paths agree on what "nothing usable" means, but not on where the
        # judgement falls: no JSON object can be parsed out of prose, while
        # prose around PNML is a defect the validator reports.
        self.assertIsNone(PNML_FORMATS["json"].to_pnml("sorry"))
        self.assertIsNone(PNML_FORMATS["json"].to_pnml(""))
        self.assertIsNone(PNML_FORMATS["xml"].to_pnml(""))

    def test_only_the_json_path_echoes_the_model_s_own_reply(self):
        """Showing it PNML would invite a PNML answer the schema forbids."""
        self.assertIsNotNone(PNML_FORMATS["json"].schema)
        self.assertIsNone(PNML_FORMATS["xml"].schema)

    def test_repair_prompt_asks_for_the_right_artefact(self):
        service = LLMService()
        xml_prompt = service._build_pnml_repair_prompt("ship it", "<pnml/>", ["bad"], "xml")
        json_prompt = service._build_pnml_repair_prompt("ship it", "{}", ["bad"], "json")
        self.assertIn("well-formed PNML XML document", xml_prompt)
        self.assertIn("JSON object matching the required schema", json_prompt)
        for prompt in (xml_prompt, json_prompt):
            self.assertIn("ship it", prompt)
            self.assertIn("bad", prompt)

    def test_only_the_json_path_carries_a_schema(self):
        self.assertIsNone(PNML_FORMATS["xml"].schema)
        self.assertIs(PNML_FORMATS["json"].schema, PetriNet)


class TestNoSilentFallback(unittest.TestCase):
    """A model that cannot honour the schema is the wrong model. Say so."""

    class _RejectingClient:
        def __init__(self, message):
            self.calls = 0
            self.message = message
            completions = SimpleNamespace(create=self._send, parse=self._send)
            self.chat = SimpleNamespace(completions=completions)
            self.beta = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        def _send(self, **kwargs):
            self.calls += 1
            raise RuntimeError(self.message)

    def test_unsupported_schema_propagates(self):
        client = self._RejectingClient("response_format is not supported with this model")
        with self.assertRaises(RuntimeError):
            LLMService._openai_generate_once(
                client, "sys", "gpt-old", "text", schema=PetriNet
            )
        self.assertEqual(client.calls, 1)  # no second, weaker attempt

    def test_unsupported_temperature_still_retries(self):
        """The one retry that exists asks for the same thing, not for less."""
        client = self._RejectingClient("temperature is unsupported")
        with self.assertRaises(RuntimeError):
            LLMService._openai_generate_once(client, "sys", "gpt-4", "text")
        self.assertEqual(client.calls, 2)

    def test_a_schema_goes_through_parse_not_create(self):
        # `create` cannot compile a Pydantic class; only `parse` can. Sending it
        # to the wrong helper would drop the constraint silently.
        seen = {}

        def record(where):
            def send(**kwargs):
                seen["helper"] = where
                seen["response_format"] = kwargs.get("response_format")
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="{}", refusal=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                    model="m",
                )

            return send

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=record("create"))),
            beta=SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(parse=record("parse")))
            ),
        )
        LLMService._openai_generate_once(client, "sys", "gpt-4", "t", schema=PetriNet)
        self.assertEqual(seen["helper"], "parse")
        self.assertIs(seen["response_format"], PetriNet)

        LLMService._openai_generate_once(client, "sys", "gpt-4", "t")
        self.assertEqual(seen["helper"], "create")
        self.assertIsNone(seen["response_format"])


class TestGeminiAdapterMirrorsOpenAI(unittest.TestCase):
    """The direct path's two adapters must differ in dialect, not in behaviour."""

    def _client(self, text, finish_reason="STOP"):
        self.config = {}

        def generate_content(model, contents, config):
            self.config = config
            return SimpleNamespace(
                text=text,
                candidates=[SimpleNamespace(finish_reason=finish_reason)],
                usage_metadata=None,
            )

        return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    def _call(self, text, finish_reason="STOP", schema=None):
        return LLMService._gemini_pnml_generate_once(
            self._client(text, finish_reason), "sys", "gemini-2.0-flash", "t", 8192,
            schema=schema,
        )

    def test_the_schema_reaches_the_provider(self):
        self._call(json.dumps(MINIMAL), schema=PetriNet)
        self.assertIs(self.config.response_schema, PetriNet)
        self.assertEqual(self.config.response_mime_type, "application/json")

    def test_no_schema_means_no_json_constraint(self):
        self._call("<pnml/>")
        self.assertIsNone(self.config.response_schema)
        self.assertIsNone(self.config.response_mime_type)

    def test_decoding_is_greedy(self):
        self._call("<pnml/>")
        self.assertEqual(
            (self.config.temperature, self.config.top_k, self.config.top_p),
            (0.0, 1, 1.0),
        )

    def test_truncation_is_checked_before_emptiness(self):
        # A reply cut off at the token limit is exhausted budget, not a refusal
        # -- even when nothing came back with it. Same order as the OpenAI side.
        from app.services.llm_service import TruncatedResponseError

        with self.assertRaises(TruncatedResponseError):
            self._call("", finish_reason="MAX_TOKENS")

    def test_an_empty_reply_without_truncation_is_an_empty_reply(self):
        from app.services.llm_service import EmptyResponseError

        with self.assertRaises(EmptyResponseError):
            self._call("")


class TestSharedPromptSemantics(unittest.TestCase):
    """One rule set, spliced into both prompts: it cannot drift apart."""

    def test_both_prompts_carry_the_shared_semantics(self):
        rule = "The outcome of a condition"
        self.assertIn(rule, BaseConfig.PNML_SYSTEM_PROMPT)
        self.assertIn(rule, BaseConfig.PNML_JSON_SYSTEM_PROMPT)

    def test_no_unresolved_placeholder_remains(self):
        self.assertNotIn("{SEMANTICS}", BaseConfig.PNML_SYSTEM_PROMPT)
        self.assertNotIn("{SEMANTICS}", BaseConfig.PNML_JSON_SYSTEM_PROMPT)

    def test_each_prompt_demands_its_own_format(self):
        self.assertIn("PNML XML document", BaseConfig.PNML_SYSTEM_PROMPT)
        self.assertIn("JSON object", BaseConfig.PNML_JSON_SYSTEM_PROMPT)
        self.assertNotIn("<?xml", BaseConfig.PNML_JSON_SYSTEM_PROMPT)

    def test_the_json_examples_serialize_to_valid_nets(self):
        """The few-shot must not teach a net the validator would reject."""
        text = BaseConfig.PNML_JSON_SYSTEM_PROMPT
        decoder, index, examples = json.JSONDecoder(), 0, 0
        while True:
            index = text.find("\n{", index)
            if index < 0:
                break
            try:
                payload, index = decoder.raw_decode(text, index + 1)
            except ValueError:
                index += 1
                continue
            examples += 1
            self.assertEqual(PnmlValidator().check(pnml_from_json(payload)).issues, [])
        self.assertEqual(examples, 5)  # the target format plus four examples

    def test_the_xml_examples_are_valid_and_sound(self):
        """Parity with the JSON examples: each PNML document the XML prompt shows
        must pass the same validator, via the XML path's own to_pnml step."""
        import re

        docs = re.findall(r"<\?xml.*?</pnml>", BaseConfig.PNML_SYSTEM_PROMPT, re.DOTALL)
        self.assertEqual(len(docs), 5)  # the target format plus four examples
        for doc in docs:
            pnml = PNML_FORMATS["xml"].to_pnml(doc)
            self.assertEqual(PnmlValidator().check(pnml).issues, [])


class TestJsonEndpoint(unittest.TestCase):
    """The route: same contract, same debug view, different prompt and format."""

    def setUp(self):
        self.app = create_app(TestingConfig)
        self.client = self.app.test_client()

    def _post(self, path, query=""):
        return self.client.post(
            path + query,
            json={"user_text": "check the order", "provider": "openai", "model": "m"},
            headers={"Authorization": "Bearer k"},
        )

    def test_json_route_returns_pnml_built_from_the_model_s_json(self):
        with patch.object(
            LLMService, "_openai_generate_once", return_value=json.dumps(MINIMAL)
        ):
            response = self._post("/generate_pnml_direct_json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/xml")
        body = response.get_data(as_text=True)
        self.assertIn('<place id="p_start">', body)
        self.assertNotIn("X-Validation-Issues", response.headers)

    def test_json_route_reports_remaining_issues_in_the_header(self):
        # A place-to-place arc is a semantic defect construction cannot remove,
        # so it is delivered best-effort with the finding in the header.
        broken = dict(
            MINIMAL, arcs=MINIMAL["arcs"] + [{"source": "p_start", "target": "p_end"}]
        )
        with patch.object(
            LLMService, "_openai_generate_once", return_value=json.dumps(broken)
        ):
            response = self._post("/generate_pnml_direct_json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("alternate", response.headers["X-Validation-Issues"])

    def test_json_route_serves_the_shared_debug_payload(self):
        with patch.object(
            LLMService, "_openai_generate_once", return_value=json.dumps(MINIMAL)
        ):
            payload = self._post("/generate_pnml_direct_json", "?debug=1").get_json()
        self.assertEqual(payload["delivered_index"], 0)
        self.assertEqual(payload["attempts"][0]["counts"]["transitions"], 1)
        self.assertIn("<pnml>", payload["pnml"])

    def test_json_route_rejects_a_reply_that_carries_no_object(self):
        with patch.object(LLMService, "_openai_generate_once", return_value="sorry"):
            response = self._post("/generate_pnml_direct_json")
        self.assertEqual(response.status_code, 400)
        self.assertIn("JSON net", response.get_json()["error"]["message"])

    def test_each_route_uses_its_own_system_prompt(self):
        seen = {}

        def capture(_client, system_prompt, *args, **kwargs):
            seen["prompt"] = system_prompt
            seen["schema"] = kwargs.get("schema")
            return json.dumps(MINIMAL)

        with patch.object(LLMService, "_openai_generate_once", staticmethod(capture)):
            self._post("/generate_pnml_direct_json")
        self.assertIn("JSON object", seen["prompt"])
        self.assertIsNotNone(seen["schema"])

        def capture_xml(_client, system_prompt, *args, **kwargs):
            seen["prompt"] = system_prompt
            seen["schema"] = kwargs.get("schema")
            return "<pnml><net/></pnml>"

        with patch.object(LLMService, "_openai_generate_once", staticmethod(capture_xml)):
            self._post("/generate_pnml_direct")
        self.assertIn("TARGET FORMAT", seen["prompt"])
        self.assertIn("<?xml", seen["prompt"])
        self.assertIsNone(seen["schema"])


if __name__ == "__main__":
    unittest.main()
