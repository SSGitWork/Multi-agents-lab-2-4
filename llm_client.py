"""
Shared LLM configuration for all three framework implementations.

Carried forward from Week 1 and extended with AutoGen and CrewAI
config helpers. You do not need to modify this file.

All three implementations route through the same Helicone proxy so
token counts appear in a single dashboard for comparison.
"""

import os

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from crewai import LLM

load_dotenv(override=True)

_HELICONE_BASE = os.getenv("HELICONE_BASE_URL")
_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
_HELICONE_API_KEY = os.getenv("HELICONE_API_KEY")

_AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
_AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
_AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
_AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


def _clean_azure_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")

    for suffix in ("/openai/v1", "/openai"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[:-len(suffix)].rstrip("/")

    return endpoint


def get_llm_client() -> OpenAI | AzureOpenAI:
    """
    Return an LLM client. Prefers Azure OpenAI if environment variables
    are set, otherwise falls back to Helicone.
    """
    if _AZURE_OPENAI_API_KEY and _AZURE_OPENAI_ENDPOINT:
        return AzureOpenAI(
            api_key=_AZURE_OPENAI_API_KEY,
            api_version=_AZURE_OPENAI_API_VERSION,
            azure_endpoint=_clean_azure_endpoint(_AZURE_OPENAI_ENDPOINT),
        )

    helicone_api_key = os.environ.get("HELICONE_API_KEY")
    if not helicone_api_key:
        raise EnvironmentError(
            "Neither Azure OpenAI nor Helicone API keys are set. "
            "Please ensure AZURE_OPENAI_API_KEY or HELICONE_API_KEY is in your .env file."
        )

    return OpenAI(
        api_key=_OPENROUTER_API_KEY,
        base_url=_HELICONE_BASE,
        default_headers={"Helicone-Auth": f"Bearer {_HELICONE_API_KEY}"},
    )


def get_autogen_llm_config() -> dict:
    """
    Return the llm_config dict expected by AutoGen agents.

    AutoGen uses a list of config dicts under the "config_list" key.
    Routing through Helicone requires setting base_url on each entry.

    Usage
    -----
    llm_config = get_autogen_llm_config()
    assistant = autogen.AssistantAgent("pm", llm_config=llm_config)
    """
    if _AZURE_OPENAI_API_KEY and _AZURE_OPENAI_ENDPOINT:
        return {
            "config_list": [
                {
                    "model": _AZURE_OPENAI_DEPLOYMENT,
                    "api_key": _AZURE_OPENAI_API_KEY,
                    "base_url": _clean_azure_endpoint(_AZURE_OPENAI_ENDPOINT),
                    "api_type": "azure",
                    "api_version": _AZURE_OPENAI_API_VERSION,
                }
            ],
            "temperature": 0,
            "cache_seed": None,
        }

    helicone_api_key = os.environ.get("HELICONE_API_KEY")
    if not helicone_api_key:
        raise EnvironmentError("HELICONE_API_KEY is not set.")
    return {
        "config_list": [
            {
                "model":    "gpt-4.1-mini",
                "api_key":  _OPENROUTER_API_KEY,
                "base_url": _HELICONE_BASE,
                "api_type": "openai",
            }
        ],
        "temperature": 0,
    }


def get_crewai_llm_config() -> dict:
    """
    Return the LLM configuration dict for CrewAI agents.

    CrewAI accepts an `llm` parameter on Agent() that can be a dict or
    an LLM instance. Using a dict with model + base_url routes through
    Helicone.

    Usage
    -----
    from crewai import Agent
    agent = Agent(role="PM", llm=get_crewai_llm_config(), ...)
    """
    if _AZURE_OPENAI_API_KEY and _AZURE_OPENAI_ENDPOINT:
        # Fallback to standard dict if called, though crewai_impl now uses get_crewai_llm()
        return {
            "model": f"azure/{_AZURE_OPENAI_DEPLOYMENT}",
            "api_key": _AZURE_OPENAI_API_KEY,
            "base_url": _clean_azure_endpoint(_AZURE_OPENAI_ENDPOINT),
        }

    helicone_api_key = os.environ.get("HELICONE_API_KEY")
    if not helicone_api_key:
        raise EnvironmentError("HELICONE_API_KEY is not set.")
    return {
        "model":    DEFAULT_MODEL,
        "api_key":  _OPENROUTER_API_KEY,
        "base_url": _HELICONE_BASE,
    }


def get_crewai_llm():
    from crewai import LLM

    if _AZURE_OPENAI_API_KEY and _AZURE_OPENAI_ENDPOINT:
        return LLM(
            model=f"azure/{_AZURE_OPENAI_DEPLOYMENT}",
            api_key=_AZURE_OPENAI_API_KEY,
            endpoint=_clean_azure_endpoint(_AZURE_OPENAI_ENDPOINT),
            api_version=_AZURE_OPENAI_API_VERSION,
            temperature=0,
        )
    return LLM(
        model=DEFAULT_MODEL,
        api_key=_OPENROUTER_API_KEY,
        base_url=_HELICONE_BASE,
        temperature=0,
    )


if _AZURE_OPENAI_API_KEY and _AZURE_OPENAI_ENDPOINT:
    MODEL = _AZURE_OPENAI_DEPLOYMENT or "build-model1-npe"
else:
    MODEL = "gpt-4.1-mini"

# Backward-compatible alias used by the Lab 1.4 reference implementations.
DEFAULT_MODEL = MODEL
