import re
import unittest
from unittest.mock import patch

from app import create_app
from app.services.llm_service import EmptyResponseError, LLMService
from app.services.pnml_validator import PnmlValidator
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


# The contract example: passes every validation level.
VALID_NET = _net(
    '<place id="p1"><name><text>start</text></name>'
    "<initialMarking><text>1</text></initialMarking></place>"
    '<transition id="t1"><name><text>check order</text></name></transition>'
    '<place id="p2"><name><text>end</text></name></place>'
    '<arc id="a1" source="p1" target="t1"/>'
    '<arc id="a2" source="t1" target="p2"/>'
)

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


class TestPnmlValidatorLevel0(unittest.TestCase):
    def test_valid_xml_passes(self):
        self.assertEqual(PnmlValidator().validate_pnml(VALID_NET), [])

    def test_truncated_xml_is_reported(self):
        issues = PnmlValidator().validate_pnml("<pnml><net id='n1'>")
        self.assertEqual(len(issues), 1)
        self.assertIn("not valid XML", issues[0])

    def test_empty_output_is_reported(self):
        for bad in ("", "   ", None):
            issues = PnmlValidator().validate_pnml(bad)
            self.assertEqual(len(issues), 1)
            self.assertIn("not valid XML", issues[0])

    def test_repair_prompt_embeds_context(self):
        prompt = LLMService()._build_pnml_repair_prompt(
            "ship the order", PNML_DOC, ["arc a1 connects two places"]
        )
        self.assertIn("ship the order", prompt)
        self.assertIn(PNML_DOC, prompt)
        self.assertIn("arc a1 connects two places", prompt)


class TestPnmlValidatorLevel1(unittest.TestCase):
    def assert_issue(self, pnml, fragment):
        issues = PnmlValidator().validate_pnml(pnml)
        self.assertTrue(
            any(fragment in issue for issue in issues),
            f"expected an issue containing {fragment!r}, got: {issues}",
        )

    def test_contract_example_passes(self):
        self.assertEqual(PnmlValidator().validate_pnml(VALID_NET), [])

    def test_namespaced_document_is_tolerated(self):
        doc = VALID_NET.replace(
            "<pnml>",
            '<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">',
        )
        self.assertEqual(PnmlValidator().validate_pnml(doc), [])

    def test_wrong_root_element(self):
        self.assert_issue("<foo/>", "root element must be <pnml>")

    def test_exactly_one_net(self):
        self.assert_issue("<pnml/>", "exactly one <net>")
        self.assert_issue("<pnml><net/><net/></pnml>", "exactly one <net>")

    def test_net_requires_id_and_ptnetb_type(self):
        self.assert_issue(
            f'<pnml><net type="{PT_NET_TYPE}"><place id="p1"/></net></pnml>',
            "missing its 'id'",
        )
        self.assert_issue(_net('<place id="p1"/>', net_type="x"), "type must be")

    def test_forbidden_elements_are_reported(self):
        self.assert_issue(
            _net('<place id="p1"><graphics/></place>'),
            "forbidden element <graphics>",
        )
        self.assert_issue(
            _net('<transition id="t1"><toolspecific tool="WoPeD"/></transition>'),
            "forbidden element <toolspecific>",
        )

    def test_unexpected_net_child_is_reported(self):
        self.assert_issue(
            _net('<name><text>my net</text></name><place id="p1"/>'),
            "unexpected element <name> under <net>",
        )

    def test_ids_required_and_unique(self):
        self.assert_issue(_net("<place/>"), "<place> without 'id'")
        self.assert_issue(
            _net('<place id="x"/><transition id="x"/>'), "duplicate id 'x'"
        )

    def test_arc_requires_source_and_target(self):
        self.assert_issue(_net('<arc id="a1" source="p1"/>'), "missing its 'source' or 'target'")

    def test_empty_name_text_is_reported(self):
        self.assert_issue(
            _net('<place id="p1"><name/></place>'),
            "must contain a non-empty <text>",
        )

class TestPnmlValidatorLevel2(unittest.TestCase):
    def assert_issue(self, pnml, fragment):
        issues = PnmlValidator().validate_pnml(pnml)
        self.assertTrue(
            any(fragment in issue for issue in issues),
            f"expected an issue containing {fragment!r}, got: {issues}",
        )

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
        self.assert_issue(doc, "the marking must sit on the start place")

    def test_transition_connectivity(self):
        doc = _net(
            MARKED_START + '<transition id="t1"/><place id="p2"/>'
            '<transition id="t2"/>'
            '<arc id="a1" source="p1" target="t1"/>'
            '<arc id="a2" source="t1" target="p2"/>'
        )
        self.assert_issue(doc, "'t2' has no incoming arc")
        self.assert_issue(doc, "'t2' has no outgoing arc")

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

    def test_local_and_graph_issues_are_reported_together(self):
        # A forbidden element must not hide graph findings (and vice versa).
        doc = VALID_NET.replace(
            '<transition id="t1">',
            '<transition id="t1"><graphics/>',
        ).replace(
            "<initialMarking><text>1</text></initialMarking>", ""
        )
        issues = PnmlValidator().validate_pnml(doc)
        self.assertTrue(any("forbidden element <graphics>" in i for i in issues))
        self.assertTrue(any("<initialMarking>" in i for i in issues))

    def test_graph_prerequisites_gate_level_2(self):
        # A duplicate id makes the graph ill-defined; level-2 findings would
        # be artifacts and must be suppressed.
        doc = _net(
            '<place id="x"/><transition id="x"/>'
            '<arc id="a1" source="x" target="x"/>'
        )
        issues = PnmlValidator().validate_pnml(doc)
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
            pnml, issues = self.service.generate_pnml(
                api_key="test-key",
                provider="openai",
                model="gpt-4o",
                user_text="ship the order",
                system_prompt="prompt under test",
            )
        return pnml, issues, mocked

    def test_valid_first_attempt_needs_no_correction(self):
        pnml, issues, mocked = self._generate([VALID_NET])
        self.assertEqual(issues, [])
        self.assertEqual(mocked.call_count, 1)

    def test_correction_fixes_the_document(self):
        pnml, issues, mocked = self._generate(
            [_with_duplicate_arc("a3"), VALID_NET]
        )
        self.assertEqual(issues, [])
        self.assertEqual(mocked.call_count, 2)
        # The correction prompt carries the previous document and the issue.
        correction_prompt = mocked.call_args_list[1].args[3]
        self.assertIn("duplicate arc", correction_prompt)
        self.assertIn("<pnml>", correction_prompt)

    def test_identical_issues_stop_the_loop(self):
        broken = _with_duplicate_arc("a3")
        pnml, issues, mocked = self._generate([broken, broken, broken])
        # Initial call plus one correction; the identical result stops it.
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(len(issues), 1)

    def test_budget_is_one_generation_plus_three_corrections(self):
        replies = [
            _with_duplicate_arc("a3"),
            _with_duplicate_arc("a4"),
            _with_duplicate_arc("a5"),
            _with_duplicate_arc("a6"),
            _with_duplicate_arc("a7"),
        ]
        pnml, issues, mocked = self._generate(replies)
        self.assertEqual(mocked.call_count, 4)
        self.assertEqual(len(issues), 1)

    def test_correction_without_pnml_keeps_previous_attempt(self):
        broken = _with_duplicate_arc("a3")
        pnml, issues, mocked = self._generate([broken, "no xml in here"])
        self.assertEqual(mocked.call_count, 2)
        self.assertIn(broken, pnml)
        self.assertEqual(len(issues), 1)

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
        pnml, issues, mocked = self._generate([two_issues, one_issue, one_issue])
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(len(issues), 1)
        self.assertIn('id="a4"', pnml)

    @patch("app.services.llm_service.genai")
    def test_gemini_provider_is_dispatched(self, mock_genai):
        with patch.object(
            LLMService, "_gemini_generate_once", side_effect=[VALID_NET]
        ) as mocked:
            pnml, issues = self.service.generate_pnml(
                api_key="test-key",
                provider="gemini",
                model="gemini-2.0-flash",
                user_text="ship the order",
                system_prompt="prompt under test",
            )
        self.assertEqual(issues, [])
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
                "/generate_pnml",
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
            "/generate_pnml",
            json=payload,
            headers={"Authorization": "Bearer test-key"},
        )

    @patch("app.api.pnml_routes.model_registry.refresh_model_cache")
    @patch("app.api.routes.model_registry.is_valid", return_value=True)
    @patch(
        "app.api.pnml_routes._llm_service.generate_pnml",
        return_value=(PNML_DOC, []),
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
        return_value=(PNML_DOC, ["transition 't9' has no inbound arc"]),
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
            "/generate_pnml",
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
    pass all validation levels."""

    def test_every_prompt_document_passes_validation(self):
        prompt = BaseConfig.PNML_SYSTEM_PROMPT
        documents = re.findall(r"<pnml\b.*?</pnml>", prompt, re.DOTALL)
        self.assertGreaterEqual(len(documents), 3)
        for document in documents:
            self.assertEqual(
                PnmlValidator().validate_pnml(document), [], document[:100]
            )


if __name__ == "__main__":
    unittest.main()
