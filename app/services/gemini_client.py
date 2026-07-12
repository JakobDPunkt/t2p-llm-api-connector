"""One Gemini client for the whole service.

Everything Gemini -- model discovery and the provider calls behind every
endpoint -- goes through the unified ``google-genai`` SDK. It replaces the
deprecated ``google-generativeai``, for two reasons that both bite in a web
service:

* that SDK set the API key process-wide via ``genai.configure()``. With a
  per-request key, two concurrent requests raced for it, and one could issue its
  call under the other's key. A client here is built per call and carries its
  own key.
* only the unified SDK compiles a Pydantic class into a provider schema
  including ``property_ordering``, which the direct-PNML JSON path needs so the
  model plans before it builds.
"""

from google import genai


def as_base_url(endpoint):
    """A configured Gemini host as the base URL the SDK expects.

    ``GEMINI_API_ENDPOINT`` is configured as a bare host
    ("generativelanguage.googleapis.com"), while ``HttpOptions.base_url`` needs a
    scheme. Values that already carry one pass through, so an http:// proxy stays
    http://.
    """
    endpoint = endpoint.strip().rstrip("/")
    return endpoint if "://" in endpoint else f"https://{endpoint}"


def build_client(api_key, api_endpoint=None):
    """Build a Gemini client for one caller's key and the configured host."""
    kwargs = {"api_key": api_key}
    if api_endpoint:
        kwargs["http_options"] = genai.types.HttpOptions(
            base_url=as_base_url(api_endpoint)
        )
    return genai.Client(**kwargs)
