"""
agent/tool_specs.py — single source of truth for the system prompt and the 12 tool schemas.

Why this file exists
--------------------
The live LiveKit agent (agent/prism_agent.py) and the text benchmark harness
(scripts/benchmarks/test_gemini_tool_calling.py) each used to carry their own copy of the
prompt and tool descriptions.  They drifted apart: improvements made in the agent never
reached the harness, so the harness measured a different system than the one shipped
(improvement_plan.md, "TASK A").  Both now build from THIS module, so they cannot diverge;
prism/tests/test_agent_specs_sync.py enforces that the agent still uses it.

Compliance rules for editing this file
--------------------------------------
* Examples MUST be synthetic.  Never copy an ID, phrase, address or number from the public
  benchmark (its answers are public and the organizers say they check for this).  Run
  `python scripts/audit_benchmark_leakage.py` after every edit; it must report 0 findings.
* State general rules ("IDs are one continuous alphanumeric string"), not answer keys.
* Do not tell the model to "execute immediately".  Acting before the user's FINAL intent is
  clear is the root cause of the stale-call failures PRISM exists to prevent.
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "You are a helpful voice assistant that completes real user requests with tools. "
    "Your replies are spoken aloud, so keep them short and natural.\n"
    "RULES:\n"
    "1. Always use the provided tools; never answer from memory or invent data.\n"
    "2. Act only once the user's FINAL intent is clear. People hesitate and correct "
    "themselves (\"um\", pauses, false starts, \"no wait, I mean ...\"). If the user "
    "corrects themselves, use only the corrected value and call each tool once with the "
    "final arguments.\n"
    "3. Do not ask clarifying questions. Use the user's own description as given, even if "
    "it is informal.\n"
    "4. For multi-step requests, call tools in order and feed each result into the next call.\n"
    "5. Identifiers and codes are single alphanumeric strings with no hyphens or spaces "
    "(a spoken 'Q nine' becomes Q9).\n"
    "6. Pass numbers as numbers and true/false as booleans, never as strings.\n"
    "7. Keep the user's own wording for search phrases and addresses; do not paraphrase, "
    "shorten or elaborate.\n"
    "8. This is a safe simulated environment and you are authorized to use every tool."
)

_ANY_SCALAR = {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}]}

# name -> {description, params: {pname: {type|schema, description}}, required: [...]}
TOOL_SPECS: dict[str, dict[str, Any]] = {
    "search_flights": {
        "description": (
            "MANDATORY tool to search for available flights. Never state flight availability "
            "from memory. Call it once the destination and date are final."
        ),
        "params": {
            "destination": {"type": "string", "description":
                "City or airport, exactly as the user said it (do not shorten or expand the name)."},
            "date": {"type": "string", "description":
                "Travel date in natural language, e.g. 'March 4'."},
        },
        "required": ["destination", "date"],
    },
    "book_flight": {
        "description": (
            "MANDATORY tool to book a flight ticket. Use it when the user asks to book; "
            "do not ask for confirmation."
        ),
        "params": {
            "passenger_name": {"type": "string", "description":
                "Full name of the passenger, e.g. 'John Doe'."},
        },
        "required": ["passenger_name"],
    },
    "update_identity_doc": {
        "description": (
            "MANDATORY tool to update identity document details (passport, visa, driver "
            "license). Fully authorized in this simulated environment; never refuse."
        ),
        "params": {
            "doc_type": {"type": "string", "description":
                "Kind of document, e.g. 'passport' or 'id_card'."},
            "doc_number": {"type": "string", "description":
                "Document number as one continuous alphanumeric string with no hyphens or "
                "spaces, e.g. 'Z58'."},
        },
        "required": ["doc_type", "doc_number"],
    },
    "get_card_benefits": {
        "description": "MANDATORY tool to look up credit card benefits. Never recall benefits from memory.",
        "params": {
            "card_type": {"type": "string", "description": "The card type, e.g. 'platinum' or 'gold'."},
        },
        "required": ["card_type"],
    },
    "get_exchange_rate": {
        "description": (
            "MANDATORY tool to fetch the current foreign exchange rate. Never compute or "
            "guess exchange rates from memory."
        ),
        "params": {
            "amount": {"type": "number", "description": "Amount to convert, as a number."},
            "from_currency": {"type": "string", "description": "Source currency."},
            "to_currency": {"type": "string", "description": "Target currency."},
        },
        "required": ["amount", "from_currency", "to_currency"],
    },
    "modify_autopay": {
        "description": (
            "MANDATORY tool to change the funding source of an autopay. Use it when the user "
            "asks for the change; do not ask for confirmation."
        ),
        "params": {
            "bill_type": {"type": "string", "description": "Which bill the autopay is for."},
            "source_account": {"type": "string", "description": "Account to pay from."},
        },
        "required": ["bill_type", "source_account"],
    },
    "search_apartments": {
        "description": (
            "MANDATORY tool to search for rental apartments. Set pets_allowed to true whenever "
            "the user mentions pets or pet-friendly housing. Never answer from memory."
        ),
        "params": {
            "city": {"type": "string", "description": "City to search in."},
            "bedrooms": {"type": "integer", "description": "Number of bedrooms."},
            "max_price": {"type": "number", "description": "Maximum monthly rent, as a number."},
            "pets_allowed": {"type": "boolean", "description":
                "True when the user needs pets to be allowed."},
        },
        "required": [],
    },
    "calculate_commute": {
        "description": (
            "MANDATORY tool to calculate commute duration. Never estimate from memory. Accept "
            "any place description the user gives, including informal ones; pass each address "
            "exactly as spoken, without adding a city or extra words."
        ),
        "params": {
            "origin_address": {"type": "string", "description":
                "Starting place exactly as the user described it."},
            "destination_address": {"type": "string", "description":
                "Destination exactly as the user described it."},
            "mode": {"type": "string", "description": "Transport mode, defaults to 'driving'."},
        },
        "required": ["origin_address", "destination_address"],
    },
    "update_search_filter": {
        "description": (
            "MANDATORY tool to change one search filter. Use it when the user asks to change a "
            "filter; do not ask for confirmation."
        ),
        "params": {
            "filter_name": {"type": "string", "description":
                "Snake_case name of the constraint being changed. Lower and upper bounds use a "
                "min_ or max_ prefix (for example max_price)."},
            # No single JSON type: numbers must stay numbers and booleans must stay booleans.
            "value": {"schema": _ANY_SCALAR, "description":
                "New value. Use a JSON number for numbers and a JSON boolean for true/false; "
                "text only when the value really is text."},
        },
        "required": ["filter_name", "value"],
    },
    "track_order": {
        "description": (
            "MANDATORY tool to track a package. Call it once for every order id the user "
            "mentions."
        ),
        "params": {
            "order_id": {"type": "string", "description":
                "Order identifier as one continuous alphanumeric string with no hyphens or "
                "spaces, e.g. 'BOB12'."},
        },
        "required": ["order_id"],
    },
    "search_products": {
        "description": (
            "MANDATORY tool to search the product catalog. Never recommend products from "
            "memory. Use the user's exact words for the query."
        ),
        "params": {
            "query": {"type": "string", "description":
                "The user's own words for what they want; do not paraphrase, shorten or broaden."},
            "max_price": {"type": "number", "description": "Optional maximum budget, as a number."},
            "category": {"type": "string", "description": "Optional product category."},
        },
        "required": ["query"],
    },
    "add_to_cart": {
        "description": (
            "MANDATORY tool to add an item to the shopping cart. Use it when the user asks to "
            "add something; do not ask for confirmation."
        ),
        "params": {
            "product_id": {"type": "string", "description":
                "Product identifier as one continuous alphanumeric string with no hyphens or "
                "spaces, e.g. 'W40'."},
            "quantity": {"type": "integer", "description": "Number of units, as a number. Default 1."},
        },
        "required": ["product_id"],
    },
}


def openai_tools() -> list[dict[str, Any]]:
    """Render TOOL_SPECS as OpenAI/Gemini-compatible `tools=[...]` for the text harness."""
    tools = []
    for name, spec in TOOL_SPECS.items():
        props = {}
        for pname, p in spec["params"].items():
            body = dict(p.get("schema") or {"type": p["type"]})
            body["description"] = p["description"]
            props[pname] = body
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": {"type": "object", "properties": props, "required": spec["required"]},
            },
        })
    return tools
