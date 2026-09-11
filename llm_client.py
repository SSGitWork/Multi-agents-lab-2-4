"""
Shared LLM configuration for all three framework implementations.

Carried forward from Week 1 and extended with AutoGen and CrewAI
config helpers. You do not need to modify this file.

All three implementations route through the same Helicone proxy so
token counts appear in a single dashboard for comparison.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

from crewai import LLM

_HELICONE_BASE = os.getenv("HELICONE_BASE_URL")
_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
_HELICONE_API_KEY = os.getenv("HELICONE_API_KEY")

def get_llm_client() -> OpenAI:
    """
    Return an OpenAI client pre-configured to route through Helicone.
    Used by the LangGraph implementation (already written in Lab 2.1).
    """
    helicone_api_key = os.environ.get("HELICONE_API_KEY")
    if not helicone_api_key:
        raise EnvironmentError(
            "HELICONE_API_KEY is not set. "
            "In Codespaces it is injected automatically. "
            "Locally, add it to a .env file."
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
    helicone_api_key = os.environ.get("HELICONE_API_KEY")
    if not helicone_api_key:
        raise EnvironmentError("HELICONE_API_KEY is not set.")
    return {
        "model":    "gpt-4.1-mini",
        "api_key":  _OPENROUTER_API_KEY,
        "base_url": _HELICONE_BASE,
    }


def get_crewai_llm():
    return LLM(
        model="gpt-4.1-mini",
        api_key=_OPENROUTER_API_KEY,
        base_url=_HELICONE_BASE,
    )
MODEL = "gpt-4.1-mini"
