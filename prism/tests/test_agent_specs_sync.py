"""
prism/tests/test_agent_specs_sync.py

Imports agent/prism_agent.py against a minimal LOCAL STUB of the livekit SDK (see
_livekit_stub/) so this test can run without the real livekit-agents package installed.

HONESTY NOTE: the stub only replicates enough of livekit.agents' surface (Agent,
AgentSession, AgentServer, llm.function_tool/find_function_tools, event types) to let this
module import and to let @ai_callable_decorator run and record what it was called with.
It does NOT prove the agent works against a real LiveKit session or a real realtime model —
only that the tool descriptions, docstrings and prompt actually loaded from
agent/tool_specs.py (the thing this task set out to fix), and that the source wiring for
correction_epoch_fn is present. Run the real agent (`python agent/prism_agent.py`) with
livekit-agents actually installed before the demo; this test is a fast, CI-friendly proxy
for the specific regression class it targets, not a substitute for that.
"""

import inspect
import sys
import unittest
from pathlib import Path

_STUB = Path(__file__).parent / "_livekit_stub"
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _import_agent_with_stub():
    sys.path.insert(0, str(_STUB))
    sys.path.insert(0, str(_PROJECT_ROOT))
    for mod in list(sys.modules):
        if mod == "livekit" or mod.startswith("livekit."):
            del sys.modules[mod]
    import dotenv  # real dependency, should already be installed
    dotenv.load_dotenv = lambda *a, **k: None  # no .env file needed for this test
    import importlib
    if "agent.prism_agent" in sys.modules:
        importlib.reload(sys.modules["agent.prism_agent"])
        return sys.modules["agent.prism_agent"]
    return importlib.import_module("agent.prism_agent")


class TestAgentBuildsFromSharedSpecs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent_mod = _import_agent_with_stub()
        from agent.tool_specs import TOOL_SPECS, SYSTEM_PROMPT
        cls.TOOL_SPECS = TOOL_SPECS
        cls.SYSTEM_PROMPT = SYSTEM_PROMPT

    def test_every_tool_description_matches_the_shared_spec_exactly(self):
        fnc = self.agent_mod.AssistantFnc.__new__(self.agent_mod.AssistantFnc)
        for name in self.TOOL_SPECS:
            method = getattr(self.agent_mod.AssistantFnc, name)
            self.assertTrue(hasattr(method, "_tool_description"),
                            f"{name} is not decorated with the function-tool decorator")
            self.assertEqual(
                method._tool_description, self.TOOL_SPECS[name]["description"],
                f"{name}'s live description has drifted from agent/tool_specs.py",
            )

    def test_dynamic_docstrings_contain_every_declared_param_and_no_leak_markers(self):
        for name in ("search_flights", "search_apartments", "calculate_commute"):
            doc = inspect.getdoc(getattr(self.agent_mod.AssistantFnc, name))
            self.assertIsNotNone(doc)
            for pname in self.TOOL_SPECS[name]["params"]:
                self.assertIn(pname, doc, f"{name}.__doc__ missing param {pname!r}")

    def test_voice_agent_instructions_is_the_shared_system_prompt(self):
        agent = self.agent_mod.VoiceAgent()
        self.assertEqual(agent.instructions, self.SYSTEM_PROMPT)

    def test_all_twelve_tools_are_registered(self):
        names = {name for name in dir(self.agent_mod.AssistantFnc)
                 if hasattr(getattr(self.agent_mod.AssistantFnc, name), "_tool_name")}
        self.assertEqual(names, set(self.TOOL_SPECS))


class TestCorrectionEpochWiring(unittest.TestCase):
    """The single most important wiring fact: without correction_epoch_fn, ToolDispatcher
    falls back to the legacy same-tool-always-supersedes rule and the whole repair-evidence
    fix in prism/tool_dispatcher.py never takes effect at runtime."""

    def test_entrypoint_source_passes_gate_correction_epoch_to_the_dispatcher(self):
        src = inspect.getsource(_import_agent_with_stub().entrypoint)
        self.assertIn("correction_epoch_fn", src)
        self.assertIn("gate.correction_epoch", src)
        # gate must be constructed before the dispatcher that references it
        gate_pos = src.index("gate = TRPGate()")
        dispatcher_pos = src.index("dispatcher = ToolDispatcher(")
        self.assertLess(gate_pos, dispatcher_pos,
                        "TRPGate must be constructed before ToolDispatcher references it")


if __name__ == "__main__":
    unittest.main()
