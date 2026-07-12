import os
from pathlib import Path


def _load_prompt_from_txt(prompt_dir, file_name):
    """Load a system prompt text from a prompt template file under app/utils."""
    prompt_file = Path(__file__).parent / "app" / "utils" / prompt_dir / file_name
    try:
        return prompt_file.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _load_system_prompt_from_txt():
    """Load system prompt text from the zero-shot prompt template file."""
    return _load_prompt_from_txt("zero-shot-prompts", "00_zero_shot_prompt.txt")


def _load_pnml_prompt(file_name):
    """Load a direct-PNML prompt, splicing in the shared modelling semantics.

    The XML and the JSON path must model a process identically; only their
    output format differs. Keeping the rules in one file and substituting them
    into both prompts makes that structural rather than a promise nobody
    checks: a rule can no longer be improved on one path and forgotten on the
    other.
    """
    template = _load_prompt_from_txt("pnml-prompts", file_name)
    semantics = _load_prompt_from_txt("pnml-prompts", "_pnml_semantics.txt")
    return template.replace("{SEMANTICS}", semantics)


def _env_bool(name, default=False):
    """Parse common truthy/falsey string values from environment."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# === Base Configuration ===
class BaseConfig:
    SYSTEM_PROMPT = _load_system_prompt_from_txt()
    # System prompts for the two experimental direct text-to-PNML endpoints:
    # the model writes the PNML itself, or a JSON net this service serializes.
    PNML_SYSTEM_PROMPT = _load_pnml_prompt("00_pnml_system_prompt.txt")
    PNML_JSON_SYSTEM_PROMPT = _load_pnml_prompt("01_pnml_json_system_prompt.txt")
    # Optional provider hosts/base URLs (useful for proxies, gateways, or
    # enterprise endpoints).
    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_HOST")
    GEMINI_API_ENDPOINT = os.environ.get("GEMINI_API_ENDPOINT") or os.environ.get(
        "GEMINI_HOST"
    )
    TESTING = False
    WTF_CSRF_ENABLED = _env_bool("WTF_CSRF_ENABLED", default=False)
    REDIS_HOST = os.environ.get("REDIS_HOST") or "127.0.0.1"
    REDIS_PORT = int(os.environ.get("REDIS_PORT") or 6379)
    REDIS_DB = int(os.environ.get("REDIS_DB") or 0)
    REDIS_URL = os.environ.get("REDIS_URL") or (
        f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    )
    REDIS_USE_MOCK = _env_bool("REDIS_USE_MOCK", default=False)
    INTERNAL_ASYNC_ENABLED = _env_bool("INTERNAL_ASYNC_ENABLED", default=True)
    # The comparison demo page at /demo. On by default so the experiment can be
    # tried out wherever the connector runs; a deployment that wants only the
    # API surface sets PNML_DEMO_ENABLED=false. The endpoints it drives stay
    # available either way.
    PNML_DEMO_ENABLED = _env_bool("PNML_DEMO_ENABLED", default=True)
    ASYNC_JOB_TTL_SECONDS = int(os.environ.get("ASYNC_JOB_TTL_SECONDS") or 3600)
    SECRET_KEY = (
        os.environ.get("SECRET_KEY")
        or "fj92348759t182htpoihf9sd8gu98341hrpasdhuq8gpsiodfh9823r"
    )


# === Development Configuration ===
class DevelopmentConfig(BaseConfig):
    DEBUG = True
    REDIS_USE_MOCK = _env_bool("REDIS_USE_MOCK", default=True)


# === Production Configuration ===
class ProductionConfig(BaseConfig):
    DEBUG = False


# === Testing Configuration ===
class TestingConfig(BaseConfig):
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or "test-openai-key"
    GEMINI_API_KEY = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or "test-gemini-key"
    )
    REDIS_USE_MOCK = True


# === Select Configuration Class Based on Environment ===
def get_config():
    env = os.getenv("FLASK_ENV", "development").lower()

    if env == "production":
        return ProductionConfig
    elif env == "testing":
        return TestingConfig
    else:
        return DevelopmentConfig
