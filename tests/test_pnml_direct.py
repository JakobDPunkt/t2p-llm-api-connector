import unittest
from unittest.mock import patch

from app import create_app
from app.services.llm_service import EmptyResponseError, LLMService
from app.services.pnml_validator import PnmlValidator
from config import TestingConfig

PNML_DOC = (
    '<pnml><net type="x" id="noID">'
    '<place id="p1"/><transition id="t1"/>'
    '<arc id="a1" source="p1" target="t1"/>'
    "</net></pnml>"
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


class TestPnmlValidatorStub(unittest.TestCase):
    def test_sanitize_stub_passes_document_through(self):
        # TODO(pnml-demo): replace once real repair heuristics exist.
        self.assertEqual(PnmlValidator().sanitize_pnml(PNML_DOC), PNML_DOC)

    def test_validate_stub_reports_no_issues(self):
        # TODO(pnml-demo): replace once real checks are implemented.
        self.assertEqual(PnmlValidator().validate_pnml(PNML_DOC), [])

    def test_repair_prompt_embeds_context(self):
        prompt = LLMService()._build_pnml_repair_prompt(
            "ship the order", PNML_DOC, ["arc a1 connects two places"]
        )
        self.assertIn("ship the order", prompt)
        self.assertIn(PNML_DOC, prompt)
        self.assertIn("arc a1 connects two places", prompt)


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
    @patch("app.api.pnml_routes._llm_service.generate_pnml", return_value=PNML_DOC)
    def test_valid_request_returns_pnml_as_xml(self, mock_generate, _valid, _refresh):
        response = self._post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/xml")
        self.assertIn(b"<pnml>", response.data)
        mock_generate.assert_called_once()
        self.assertEqual(
            mock_generate.call_args.kwargs["system_prompt"],
            self.app.config["PNML_SYSTEM_PROMPT"],
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


if __name__ == "__main__":
    unittest.main()
