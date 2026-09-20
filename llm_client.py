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


def get_llm_client() -> OpenAI | AzureOpenAI:
    """
    Return an LLM client. Prefers Azure OpenAI if environment variables
    are set, otherwise falls back to Helicone.
    """
    if _AZURE_OPENAI_API_KEY and _AZURE_OPENAI_ENDPOINT:
        clean_endpoint = _AZURE_OPENAI_ENDPOINT.strip()
        if clean_endpoint.endswith("/openai/v1/"):
            clean_endpoint = clean_endpoint.replace("/openai/v1/", "")
        elif clean_endpoint.endswith("/openai/v1"):
            clean_endpoint = clean_endpoint.replace("/openai/v1", "")

        return AzureOpenAI(
            api_key=_AZURE_OPENAI_API_KEY,
            api_version=_AZURE_OPENAI_API_VERSION,
            azure_endpoint=clean_endpoint,
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
        clean_endpoint = _AZURE_OPENAI_ENDPOINT.strip()
        if clean_endpoint.endswith("/openai/v1/"):
            clean_endpoint = clean_endpoint.replace("/openai/v1/", "")
        elif clean_endpoint.endswith("/openai/v1"):
            clean_endpoint = clean_endpoint.replace("/openai/v1", "")
        return {
            "config_list": [
                {
                    "model": _AZURE_OPENAI_DEPLOYMENT or "build-model1-npe",
                    "api_key": _AZURE_OPENAI_API_KEY,
                    "base_url": clean_endpoint,
                    "api_type": "azure",
                    "api_version": _AZURE_OPENAI_API_VERSION,
                }
            ],
            "temperature": 0,
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
        clean_endpoint = _AZURE_OPENAI_ENDPOINT.strip()
        if clean_endpoint.endswith("/openai/v1/"):
            clean_endpoint = clean_endpoint.replace("/openai/v1/", "")
        elif clean_endpoint.endswith("/openai/v1"):
            clean_endpoint = clean_endpoint.replace("/openai/v1", "")
        return {
            "model": _AZURE_OPENAI_DEPLOYMENT or "build-model1-npe",
            "api_key": _AZURE_OPENAI_API_KEY,
            "base_url": clean_endpoint,
            "api_version": _AZURE_OPENAI_API_VERSION,
        }

    helicone_api_key = os.environ.get("HELICONE_API_KEY")
    if not helicone_api_key:
        raise EnvironmentError("HELICONE_API_KEY is not set.")
    return {
        "model":    "gpt-4.1-mini",
        "api_key":  _OPENROUTER_API_KEY,
        "base_url": _HELICONE_BASE,
    }


def get_crewai_llm():
    if _AZURE_OPENAI_API_KEY and _AZURE_OPENAI_ENDPOINT:
        clean_endpoint = _AZURE_OPENAI_ENDPOINT.strip()
        if clean_endpoint.endswith("/openai/v1/"):
            clean_endpoint = clean_endpoint.replace("/openai/v1/", "")
        elif clean_endpoint.endswith("/openai/v1"):
            clean_endpoint = clean_endpoint.replace("/openai/v1", "")
        return LLM(
            model=_AZURE_OPENAI_DEPLOYMENT or "build-model1-npe",
            api_key=_AZURE_OPENAI_API_KEY,
            base_url=clean_endpoint,
            api_version=_AZURE_OPENAI_API_VERSION,
        )
    return LLM(
        model="gpt-4.1-mini",
        api_key=_OPENROUTER_API_KEY,
        base_url=_HELICONE_BASE,
    )


if _AZURE_OPENAI_API_KEY and _AZURE_OPENAI_ENDPOINT:
    MODEL = _AZURE_OPENAI_DEPLOYMENT or "build-model1-npe"
else:
    MODEL = "gpt-4.1-mini"
