import re
import unittest
from unittest.mock import patch

from app import create_app
from app.services.llm_service import (
    EmptyResponseError,
    LLMService,
    PnmlAttempt,
    PnmlGeneration,
)
from app.services.pnml_validator import NetCounts, PnmlValidator
from config import BaseConfig, TestingConfig

PT_NET_TYPE = "http://www.informatik.hu-berlin.de/top/pntd/ptNetb"

PNML_DOC = (
    f'<pnml><net type="{PT_NET_TYPE}" id="noID">'
    '<place id="p1"/><transition id="t1"/>'
    '<arc id="a1" source="p1" target="t1"/>'
    "</net></pnml>"
)


def _net(inner, net_type=PT_NET_TYPE):
    return f'<pnml><net id="noID" type="{net_type}">{inner}</net></pnml>'


# The contract example: normalized without changes, no structural issues.
VALID_INNER = (
    '<place id="p1"><name><text>start</text></name>'
    "<initialMarking><text>1</text></initialMarking></place>"
    '<transition id="t1"><name><text>check order</text></name></transition>'
    '<place id="p2"><name><text>end</text></name></place>'
    '<arc id="a1" source="p1" target="t1"/>'
    '<arc id="a2" source="t1" target="p2"/>'
)
VALID_NET = _net(VALID_INNER)

# Nameless t2/t5 route the flow (AND-split / AND-join); only the 1-in/1-out
# transitions t1, t3, t4 model activities and carry a label.
AND_SPLIT_JOIN_NET = _net(
    '<place id="p1"><name><text>start</text></name>'
    "<initialMarking><text>1</text></initialMarking></place>"
    '<transition id="t1"><name><text>check order</text></name></transition>'
    '<place id="p2"/><transition id="t2"/><place id="p3"/><place id="p4"/>'
    '<transition id="t3"><name><text>pack goods</text></name></transition>'
    '<transition id="t4"><name><text>print invoice</text></name></transition>'
    '<place id="p5"/><place id="p6"/><transition id="t5"/>'
    '<place id="p7"><name><text>end</text></name></place>'
    '<arc id="a1" source="p1" target="t1"/><arc id="a2" source="t1" target="p2"/>'
    '<arc id="a3" source="p2" target="t2"/>'
    '<arc id="a4" source="t2" target="p3"/><arc id="a5" source="t2" target="p4"/>'
    '<arc id="a6" source="p3" target="t3"/><arc id="a7" source="p4" target="t4"/>'
    '<arc id="a8" source="t3" target="p5"/><arc id="a9" source="t4" target="p6"/>'
    '<arc id="a10" source="p5" target="t5"/><arc id="a11" source="p6" target="t5"/>'
    '<arc id="a12" source="t5" target="p7"/>'
)


def _check(pnml):
    return PnmlValidator().check(pnml)


def _issues(pnml):
    return _check(pnml).issues


def _generation(pnml, issues=()):
    """A single-attempt PnmlGeneration, for tests that mock out the service."""
    attempt = PnmlAttempt(
        pnml=pnml, issues=list(issues), stripped=[], counts=NetCounts(0, 0, 0)
    )
    return PnmlGeneration(attempts=[attempt], best_index=0)

# A marked start place, so only the level under test reports issues.
MARKED_START = (
    '<place id="p1"><initialMarking><text>1</text></initialMarking></place>'
)


class TestPnmlExtraction(unittest.TestCase):
    def test_plain_pnml_is_returned_with_declaration(self):
        result = LLMService._extract_pnml_document(PNML_DOC)
        self.assertTrue(result.startswith("<?xml"))
        self.assertIn(PNML_DOC, result)

    def test_markdown_fences_and_prose_are_stripped(self):
        raw = f"Sure! Here is your net:\n```xml\n{PNML_DOC}\n```\nEnjoy."
        result = LLMService._extract_pnml_document(raw)
        self.assertIn(PNML_DOC, result)
        self.assertNotIn("```", result)
        self.assertNotIn("Sure!", result)

    def test_reply_without_pnml_returns_none(self):
        self.assertIsNone(LLMService._extract_pnml_document("no net here"))
        self.assertIsNone(LLMService._extract_pnml_document(""))
        self.assertIsNone(LLMService._extract_pnml_document(None))


class TestRepairPrompt(unittest.TestCase):
    def test_repair_prompt_embeds_context(self):
        prompt = LLMService()._build_pnml_repair_prompt(
            "ship the order", PNML_DOC, ["arc a1 connects two places"]
        )
        self.assertIn("ship the order", prompt)
        self.assertIn(PNML_DOC, prompt)
        self.assertIn("arc a1 connects two places", prompt)


class TestPnmlGates(unittest.TestCase):
    """Nothing can be said about a document that is not PNML at all."""

    def assert_issue(self, pnml, fragment):
        issues = _issues(pnml)
        self.assertEqual(len(issues), 1, issues)
        self.assertIn(fragment, issues[0])

    def test_contract_example_passes_untouched(self):
        result = _check(VALID_NET)
        self.assertEqual(result.issues, [])
        self.assertEqual(result.stripped, [])
        self.assertEqual(result.pnml, VALID_NET)

    def test_truncated_xml_is_reported(self):
        self.assert_issue("<pnml><net id='n1'>", "not valid XML")

    def test_empty_output_is_reported(self):
        for bad in ("", "   ", None):
            self.assert_issue(bad, "not valid XML")

    def test_wrong_root_element(self):
        self.assert_issue("<foo/>", "root element must be <pnml>")

    def test_exactly_one_net(self):
        self.assert_issue("<pnml/>", "exactly one <net>")
        self.assert_issue("<pnml><net/><net/></pnml>", "exactly one <net>")

    def test_a_gated_document_is_returned_unchanged(self):
        self.assertEqual(_check("<foo/>").pnml, "<foo/>")

    def test_namespaced_document_is_tolerated(self):
        doc = VALID_NET.replace(
            "<pnml>",
            '<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">',
        )
        self.assertEqual(_issues(doc), [])


class TestPnmlNormalization(unittest.TestCase):
    """Deterministic clean-up: recorded in `stripped`, never an issue."""

    def assert_stripped(self, pnml, fragment):
        result = _check(pnml)
        self.assertTrue(
            any(fragment in entry for entry in result.stripped),
            f"expected a clean-up containing {fragment!r}, got: {result.stripped}",
        )
        return result

    def test_semantics_free_elements_are_removed(self):
        for element, markup in (
            ("graphics", '<place id="p1"><graphics/></place>'),
            ("toolspecific", '<transition id="t1"><toolspecific tool="WoPeD"/></transition>'),
            ("inscription", '<arc id="a1" source="p1" target="t1"><inscription/></arc>'),
        ):
            result = self.assert_stripped(_net(markup), f"removed 1 <{element}>")
            self.assertNotIn(element, result.pnml)

    def test_page_container_is_unwrapped_keeping_its_children(self):
        # Dropping <page> instead of unwrapping it would delete the whole net.
        result = self.assert_stripped(
            _net(f'<page id="pg1">{VALID_INNER}</page>'), "unwrapped 1 <page>"
        )
        self.assertEqual(result.issues, [])
        self.assertNotIn("page", result.pnml)

    def test_unexpected_net_child_is_removed(self):
        result = self.assert_stripped(
            _net('<name><text>my net</text></name><place id="p1"/>'),
            "unexpected element(s) under <net> (name)",
        )
        self.assertNotIn("my net", result.pnml)

    def test_net_id_and_type_are_normalized(self):
        self.assert_stripped(
            f'<pnml><net type="{PT_NET_TYPE}"><place id="p1"/></net></pnml>',
            "'id' attribute to 'noID'",
        )
        result = self.assert_stripped(
            _net('<place id="p1"/>', net_type="x"), "'type' attribute"
        )
        self.assertIn(PT_NET_TYPE, result.pnml)

    def test_empty_place_name_is_removed(self):
        # Intermediate places may be nameless, so an empty <name> is noise.
        result = self.assert_stripped(
            _net('<place id="p1"><name/></place>'), "empty <name> element(s)"
        )
        self.assertNotIn("<name", result.pnml)

    def test_normalization_never_hides_a_structural_issue(self):
        doc = VALID_NET.replace(
            '<transition id="t1">', '<transition id="t1"><graphics/>'
        ).replace("<initialMarking><text>1</text></initialMarking>", "")
        result = _check(doc)
        self.assertTrue(any("<graphics>" in s for s in result.stripped))
        self.assertTrue(any("<initialMarking>" in i for i in result.issues))


class TestPnmlStructure(unittest.TestCase):
    """The workflow-net invariants. Only these warrant a correction pass."""

    def assert_issue(self, pnml, fragment):
        issues = _issues(pnml)
        self.assertTrue(
            any(fragment in issue for issue in issues),
            f"expected an issue containing {fragment!r}, got: {issues}",
        )

    def test_nameless_transition_is_allowed(self):
        # A transition without a <name> is a silent transition, a standard
        # workflow-net construct. It is not reported, regardless of arc degree:
        # forcing a label would make the model invent an undescribed activity.
        for unlabelled in ('<transition id="t1"/>', '<transition id="t1"><name/></transition>'):
            doc = VALID_NET.replace(
                '<transition id="t1"><name><text>check order</text></name></transition>',
                unlabelled,
            )
            self.assertEqual(_issues(doc), [])

    def test_nameless_routing_transitions_are_allowed(self):
        # Split/join routing transitions likewise need no label.
        self.assertEqual(_issues(AND_SPLIT_JOIN_NET), [])

    def test_ids_required_and_unique(self):
        self.assert_issue(_net("<place/>"), "<place> without 'id'")
        self.assert_issue(
            _net('<place id="x"/><transition id="x"/>'), "duplicate id 'x'"
        )

    def test_arc_requires_source_and_target(self):
        self.assert_issue(_net('<arc id="a1" source="p1"/>'), "missing its 'source' or 'target'")

    def test_unresolved_arc_reference(self):
        self.assert_issue(
            _net(MARKED_START + '<arc id="a1" source="p1" target="ghost"/>'),
            "references unknown target 'ghost'",
        )

    def test_self_loop(self):
        self.assert_issue(
            _net(MARKED_START + '<arc id="a1" source="p1" target="p1"/>'),
            "connects 'p1' to itself",
        )

    def test_bipartiteness(self):
        doc = _net(
            MARKED_START + '<place id="p2"/><transition id="t1"/>'
            '<arc id="a1" source="p1" target="p2"/>'
            '<arc id="a2" source="p2" target="t1"/>'
            '<arc id="a3" source="t1" target="p2"/>'
        )
        self.assert_issue(doc, "place 'p1' to place 'p2'")

    def test_exactly_one_source_and_sink(self):
        two_sources = _net(
            MARKED_START + '<place id="p2"/><transition id="t1"/><place id="p3"/>'
            '<arc id="a1" source="p1" target="t1"/>'
            '<arc id="a2" source="p2" target="t1"/>'
            '<arc id="a3" source="t1" target="p3"/>'
        )
        self.assert_issue(two_sources, "exactly one start place")

    def test_marking_is_required(self):
        unmarked = _net(
            '<place id="p1"/><transition id="t1"/><place id="p2"/>'
            '<arc id="a1" source="p1" target="t1"/>'
            '<arc id="a2" source="t1" target="p2"/>'
        )
        self.assert_issue(
            unmarked, "exactly one place with an <initialMarking>, found 0"
        )

    def test_marking_must_be_exactly_one_token(self):
        for bad_value in ("3", "-1", "one"):
            doc = VALID_NET.replace(
                "<initialMarking><text>1</text></initialMarking>",
                f"<initialMarking><text>{bad_value}</text></initialMarking>",
            )
            self.assert_issue(doc, f"must be exactly 1, found '{bad_value}'")

    def test_marking_must_sit_on_the_source(self):
        doc = _net(
            '<place id="p1"/><transition id="t1"/>'
            '<place id="p2"><initialMarking><text>1</text></initialMarking></place>'
            '<arc id="a1" source="p1" target="t1"/>'
            '<arc id="a2" source="t1" target="p2"/>'
        )
        self.assert_issue(doc, "'p2' carries the initial marking but has incoming arcs")

    def test_transition_missing_one_direction_is_reported(self):
        # t2 has an incoming arc but no outgoing one: the specific message.
        doc = _net(
            MARKED_START + '<transition id="t1"/><place id="p2"/>'
            '<transition id="t2"/>'
            '<arc id="a1" source="p1" target="t1"/>'
            '<arc id="a2" source="t1" target="p2"/>'
            '<arc id="a3" source="p2" target="t2"/>'
        )
        self.assert_issue(doc, "'t2' has no outgoing arc")

    def test_isolated_node_is_reported_as_disconnected_once(self):
        # A node with no arcs at all surfaces once as disconnected, not as a
        # spurious extra start/end place or a doubled no-incoming/no-outgoing
        # pair. p_iso (a place) and t2 (a transition) are both fully isolated.
        doc = _net(
            MARKED_START + '<transition id="t1"/><place id="p2"/>'
            '<place id="p_iso"/><transition id="t2"/>'
            '<arc id="a1" source="p1" target="t1"/>'
            '<arc id="a2" source="t1" target="p2"/>'
        )
        issues = _issues(doc)
        self.assertTrue(any("'p_iso' has no arcs at all" in i for i in issues), issues)
        self.assertTrue(any("'t2' has no arcs at all" in i for i in issues), issues)
        # Not miscounted as a second start/end, nor doubled per direction.
        self.assertFalse(any("start place" in i for i in issues), issues)
        self.assertFalse(any("no incoming arc" in i for i in issues), issues)

    def test_duplicate_parallel_arcs_are_reported(self):
        doc = VALID_NET.replace(
            '<arc id="a2" source="t1" target="p2"/>',
            '<arc id="a2" source="t1" target="p2"/>'
            '<arc id="a3" source="t1" target="p2"/>',
        )
        self.assert_issue(doc, "duplicate arc from 't1' to 'p2'")

    def test_stranded_nodes_are_reported(self):
        doc = _net(
            MARKED_START + '<transition id="t1"/><place id="p2"/>'
            '<transition id="t2"/><place id="p3"/>'
            '<arc id="a1" source="p1" target="t1"/>'
            '<arc id="a2" source="t1" target="p2"/>'
            '<arc id="a3" source="p3" target="t2"/>'
            '<arc id="a4" source="t2" target="p3"/>'
        )
        self.assert_issue(doc, "'t2' lies on no path")

    def test_graph_prerequisites_gate_the_graph_checks(self):
        # A duplicate id makes the graph ill-defined; the graph findings would
        # be artifacts and must be suppressed.
        doc = _net(
            '<place id="x"/><transition id="x"/>'
            '<arc id="a1" source="x" target="x"/>'
        )
        issues = _issues(doc)
        self.assertTrue(any("duplicate id 'x'" in i for i in issues))
        self.assertFalse(any("itself" in i for i in issues))


def _with_duplicate_arc(arc_id):
    """VALID_NET plus one duplicate t1->p2 arc; exactly one level-2 issue."""
    return VALID_NET.replace(
        '<arc id="a2" source="t1" target="p2"/>',
        '<arc id="a2" source="t1" target="p2"/>'
        f'<arc id="{arc_id}" source="t1" target="p2"/>',
    )


class TestPnmlCorrectionLoop(unittest.TestCase):
    """The generate -> validate -> correct loop with a mocked provider."""

    def setUp(self):
        self.app = create_app(TestingConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.service = LLMService()

    def tearDown(self):
        self.app_context.pop()

    def _generate(self, mock_once):
        with patch.object(
            LLMService, "_openai_generate_once", side_effect=mock_once
        ) as mocked:
            generation = self.service.generate_pnml(
                api_key="test-key",
                provider="openai",
                model="gpt-4o",
                user_text="ship the order",
                system_prompt="prompt under test",
            )
        return generation, mocked

    def test_valid_first_attempt_needs_no_correction(self):
        generation, mocked = self._generate([VALID_NET])
        self.assertEqual(generation.best.issues, [])
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(len(generation.attempts), 1)

    def test_correction_fixes_the_document(self):
        generation, mocked = self._generate(
            [_with_duplicate_arc("a3"), VALID_NET]
        )
        self.assertEqual(generation.best.issues, [])
        self.assertEqual(mocked.call_count, 2)
        # The correction prompt carries the previous document and the issue.
        correction_prompt = mocked.call_args_list[1].args[3]
        self.assertIn("duplicate arc", correction_prompt)
        self.assertIn("<pnml>", correction_prompt)

    def test_identical_issues_stop_the_loop(self):
        broken = _with_duplicate_arc("a3")
        generation, mocked = self._generate([broken, broken, broken])
        # Initial call plus one correction; the identical result stops it.
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(len(generation.best.issues), 1)

    def test_budget_is_one_generation_plus_three_corrections(self):
        replies = [
            _with_duplicate_arc("a3"),
            _with_duplicate_arc("a4"),
            _with_duplicate_arc("a5"),
            _with_duplicate_arc("a6"),
            _with_duplicate_arc("a7"),
        ]
        generation, mocked = self._generate(replies)
        self.assertEqual(mocked.call_count, 4)
        self.assertEqual(len(generation.attempts), 4)
        self.assertEqual(len(generation.best.issues), 1)

    def test_correction_without_pnml_keeps_previous_attempt(self):
        broken = _with_duplicate_arc("a3")
        generation, mocked = self._generate([broken, "no xml in here"])
        self.assertEqual(mocked.call_count, 2)
        self.assertIn(broken, generation.best.pnml)
        self.assertEqual(len(generation.best.issues), 1)
        # The unusable reply is not an attempt: nothing was validated.
        self.assertEqual(len(generation.attempts), 1)

    def test_best_attempt_with_fewest_issues_is_delivered(self):
        two_issues = VALID_NET.replace(
            '<arc id="a1" source="p1" target="t1"/>',
            '<arc id="a1" source="p1" target="t1"/>'
            '<arc id="a0" source="p1" target="t1"/>',
        ).replace(
            '<arc id="a2" source="t1" target="p2"/>',
            '<arc id="a2" source="t1" target="p2"/>'
            '<arc id="a3" source="t1" target="p2"/>',
        )
        one_issue = _with_duplicate_arc("a4")
        # 2 issues -> 1 issue -> identical (stop); the 1-issue attempt wins.
        generation, mocked = self._generate([two_issues, one_issue, one_issue])
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(generation.best_index, 1)
        self.assertEqual(len(generation.best.issues), 1)
        self.assertIn('id="a4"', generation.best.pnml)

    def test_history_records_the_unaided_first_shot(self):
        # The first attempt survives even though a later one is delivered.
        generation, _ = self._generate([_with_duplicate_arc("a3"), VALID_NET])
        first = generation.attempts[0]
        self.assertEqual(len(first.issues), 1)
        self.assertIn("duplicate arc", first.issues[0])
        self.assertEqual(generation.best_index, 1)
        self.assertEqual(generation.best.issues, [])

    def test_history_records_normalization_and_net_size(self):
        dirty = VALID_NET.replace('<transition id="t1">', '<transition id="t1"><graphics/>')
        generation, mocked = self._generate([dirty])
        first = generation.attempts[0]
        # Cosmetics are stripped, so they cost no correction pass.
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(first.issues, [])
        self.assertTrue(any("<graphics>" in entry for entry in first.stripped))
        self.assertEqual(tuple(first.counts), (2, 1, 2))

    def test_a_shrinking_correction_is_called_out(self):
        # A transition without arcs is fixed by deleting it -- the loop accepts
        # that (fewer issues), so the lost node must at least be logged.
        stranded = VALID_NET.replace(
            '<place id="p2"><name><text>end</text></name></place>',
            '<place id="p2"><name><text>end</text></name></place>'
            '<transition id="t9"><name><text>archive order</text></name></transition>',
        )
        with self.assertLogs("app.services.llm_service", level="WARNING") as logs:
            generation, _ = self._generate([stranded, VALID_NET])
        self.assertEqual(generation.attempts[0].counts.transitions, 2)
        self.assertEqual(generation.best.counts.transitions, 1)
        self.assertTrue(any("shrank the net" in line for line in logs.output))

    @patch("app.services.llm_service.genai")
    def test_gemini_provider_is_dispatched(self, mock_genai):
        with patch.object(
            LLMService, "_gemini_generate_once", side_effect=[VALID_NET]
        ) as mocked:
            generation = self.service.generate_pnml(
                api_key="test-key",
                provider="gemini",
                model="gemini-2.0-flash",
                user_text="ship the order",
                system_prompt="prompt under test",
            )
        self.assertEqual(generation.best.issues, [])
        mocked.assert_called_once()
        mock_genai.configure.assert_called_once_with(api_key="test-key")


class TestGeneratePnmlIntegration(unittest.TestCase):
    """The complete chain HTTP -> route -> service -> validator -> correction
    loop -> response; only the bare provider call is mocked."""

    @patch("app.model_registry.refresh_model_cache")
    def setUp(self, mock_refresh_model_cache):
        self.app = create_app(TestingConfig)
        self.client = self.app.test_client()

    def _post(self, replies):
        with patch("app.api.routes.model_registry.is_valid", return_value=True), \
             patch("app.api.pnml_routes.model_registry.refresh_model_cache"), \
             patch.object(LLMService, "_openai_generate_once", side_effect=replies):
            return self.client.post(
                "/generate_pnml_direct",
                json={
                    "user_text": "ship the order",
                    "provider": "openai",
                    "model": "gpt-4o",
                },
                headers={"Authorization": "Bearer test-key"},
            )

    def test_invalid_first_reply_is_corrected_end_to_end(self):
        response = self._post([_with_duplicate_arc("a3"), VALID_NET])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/xml")
        self.assertTrue(response.data.startswith(b"<?xml"))
        self.assertNotIn("X-Validation-Issues", response.headers)

    def test_persistent_issues_deliver_best_effort_with_header(self):
        broken = _with_duplicate_arc("a3")
        response = self._post([broken, broken, broken, broken])
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="a3"', response.data)
        self.assertIn("duplicate arc", response.headers["X-Validation-Issues"])


class TestDemoPage(unittest.TestCase):
    @patch("app.model_registry.refresh_model_cache")
    def setUp(self, mock_refresh_model_cache):
        self.app = create_app(TestingConfig)
        self.client = self.app.test_client()

    def test_demo_page_is_served(self):
        response = self.client.get("/demo")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/html")
        self.assertIn(b"Text to PNML Demo", response.data)


class TestGeneratePnmlRoute(unittest.TestCase):
    @patch("app.model_registry.refresh_model_cache")
    def setUp(self, mock_refresh_model_cache):
        self.app = create_app(TestingConfig)
        self.client = self.app.test_client()

    def _post(self, **overrides):
        payload = {
            "user_text": "ship the order",
            "provider": "openai",
            "model": "gpt-4o",
        }
        payload.update(overrides)
        return self.client.post(
            "/generate_pnml_direct",
            json=payload,
            headers={"Authorization": "Bearer test-key"},
        )

    @patch("app.api.pnml_routes.model_registry.refresh_model_cache")
    @patch("app.api.routes.model_registry.is_valid", return_value=True)
    @patch(
        "app.api.pnml_routes._llm_service.generate_pnml",
        return_value=_generation(PNML_DOC),
    )
    def test_valid_request_returns_pnml_as_xml(self, mock_generate, _valid, _refresh):
        response = self._post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/xml")
        self.assertIn(b"<pnml>", response.data)
        self.assertNotIn("X-Validation-Issues", response.headers)
        mock_generate.assert_called_once()
        self.assertEqual(
            mock_generate.call_args.kwargs["system_prompt"],
            self.app.config["PNML_SYSTEM_PROMPT"],
        )

    @patch("app.api.pnml_routes.model_registry.refresh_model_cache")
    @patch("app.api.routes.model_registry.is_valid", return_value=True)
    @patch(
        "app.api.pnml_routes._llm_service.generate_pnml",
        return_value=_generation(PNML_DOC, ["transition 't9' has no inbound arc"]),
    )
    def test_remaining_issues_are_reported_in_header(self, _gen, _valid, _refresh):
        response = self._post()
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<pnml>", response.data)
        self.assertEqual(
            response.headers["X-Validation-Issues"],
            "transition 't9' has no inbound arc",
        )

    def test_missing_auth_returns_401(self):
        response = self.client.post(
            "/generate_pnml_direct",
            json={"user_text": "x", "provider": "openai", "model": "gpt-4o"},
        )
        self.assertEqual(response.status_code, 401)

    @patch("app.api.pnml_routes.model_registry.refresh_model_cache")
    @patch("app.api.routes.model_registry.is_valid", return_value=True)
    @patch(
        "app.api.pnml_routes._llm_service.generate_pnml",
        side_effect=EmptyResponseError("no PNML"),
    )
    def test_reply_without_pnml_maps_to_400(self, _generate, _valid, _refresh):
        response = self._post()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json()["error"]["code"], "invalid_request"
        )

    @patch("app.api.pnml_routes.model_registry.refresh_model_cache")
    @patch("app.api.routes.model_registry.is_valid", return_value=True)
    @patch(
        "app.api.pnml_routes._llm_service.generate_pnml",
        side_effect=Exception("Rate limit exceeded for requests"),
    )
    def test_provider_quota_maps_to_429(self, _generate, _valid, _refresh):
        response = self._post()
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.get_json()["error"]["code"], "rate_limited")


class TestPromptValidatorParity(unittest.TestCase):
    """The system prompt may never teach a format its own validation rejects:
    every PNML document embedded in the prompt (skeleton and examples) must
    pass unchanged -- no structural issue, and nothing left to strip."""

    def test_every_prompt_document_passes_validation(self):
        prompt = BaseConfig.PNML_SYSTEM_PROMPT
        documents = re.findall(r"<pnml\b.*?</pnml>", prompt, re.DOTALL)
        self.assertGreaterEqual(len(documents), 3)
        for document in documents:
            result = _check(document)
            self.assertEqual(result.issues, [], document[:100])
            self.assertEqual(result.stripped, [], document[:100])


if __name__ == "__main__":
    unittest.main()
