"""
Unit tests for constrained_output.py.

Only ValidateRepairBackend is tested here — it needs nothing but `jsonschema` and a
fake generate_fn, so it's fully testable in CI / this sandbox.

XGrammarBackend is NOT tested here on purpose: it needs `xgrammar` installed AND a real
local model + tokenizer loaded, which is too heavy for a unit test and wasn't something
I could actually run in the environment I built this in (see constrained_output.py's
module docstring for the honesty note on that). Before the demo, your team should add an
integration test against your real local model — at minimum, assert that
`XGrammarBackend(tokenizer, vocab_size).generate(schema, model, tokenizer, prompt).payload`
validates against the same schema (belt-and-suspenders check, since the grammar should
already guarantee this).
"""

from module4.constrained_output import (
    SchemaEnforcementError,
    ValidateRepairBackend,
    enforce_json_schema,
)

SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["text", "count"],
}


def test_valid_json_on_first_attempt():
    calls = []

    def gen(prompt):
        calls.append(prompt)
        return '{"text": "hi", "count": 3}'

    result = enforce_json_schema(SCHEMA, gen, "base prompt")
    assert result.payload == {"text": "hi", "count": 3}
    assert result.attempts == 1
    assert len(calls) == 1


def test_repairs_after_invalid_first_attempt():
    responses = iter([
        '{"text": "hi"}',                    # missing required "count"
        '{"text": "hi", "count": 3}',        # now valid
    ])

    def gen(prompt):
        return next(responses)

    result = enforce_json_schema(SCHEMA, gen, "base prompt", ValidateRepairBackend(max_retries=2))
    assert result.payload == {"text": "hi", "count": 3}
    assert result.attempts == 2


def test_strips_markdown_fences():
    def gen(prompt):
        return '```json\n{"text": "hi", "count": 1}\n```'

    result = enforce_json_schema(SCHEMA, gen, "base prompt")
    assert result.payload == {"text": "hi", "count": 1}


def test_raises_after_exhausting_retries():
    def gen(prompt):
        return '{"text": "hi"}'  # always missing "count"

    try:
        enforce_json_schema(SCHEMA, gen, "base prompt", ValidateRepairBackend(max_retries=1))
        assert False, "expected SchemaEnforcementError"
    except SchemaEnforcementError:
        pass


def test_repair_prompt_includes_previous_error():
    prompts_seen = []
    responses = iter(['not json at all', '{"text": "hi", "count": 1}'])

    def gen(prompt):
        prompts_seen.append(prompt)
        return next(responses)

    enforce_json_schema(SCHEMA, gen, "base prompt", ValidateRepairBackend(max_retries=2))
    assert "base prompt" in prompts_seen[0]
    assert "invalid" in prompts_seen[1].lower()
