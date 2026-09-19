"""
tool_manifest.py
Classifies tools as read-only vs state-modifying.

CHANGED: Module 3 validates tool calls against scenario schemas itself at
the gate. This local classifier is still needed for OUR OWN decisions:
  - whether to set side_effect="STATE_MODIFYING" on the outgoing tool_call
  - which parameters are required, to know when the slot tracker is "ready"

Populate it from the TOOL_MANIFEST input event Module 3 delivers at
scenario start (see orchestrator.py's on_tool_manifest handler) rather
than a hardcoded dict.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List


class ToolKind(Enum):
    READ_ONLY = "READ_ONLY"
    STATE_MODIFYING = "STATE_MODIFYING"   # matches Module 3's side_effect string


_READ_VERBS = {"get", "list", "check", "search", "find", "fetch", "read", "lookup", "query"}
_WRITE_VERBS = {"book", "create", "update", "set", "delete", "cancel", "modify",
                "pay", "send", "schedule", "reserve", "confirm", "add", "remove"}


@dataclass
class ToolSpec:
    name: str
    kind: ToolKind
    parameters: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


class ToolManifest:
    def __init__(self):
        self.tools: Dict[str, ToolSpec] = {}

    def load_from_event_payload(self, tools: List[Dict[str, Any]]):
        """Call this from the TOOL_MANIFEST handler: payload['tools']."""
        for t in tools:
            self.tools[t["name"]] = self._classify(t)

    def _classify(self, t: Dict[str, Any]) -> ToolSpec:
        name = t["name"]
        params = t.get("parameters", {})

        if "side_effect" in t:
            raw = str(t["side_effect"]).upper()
            kind = ToolKind.READ_ONLY if raw in ("NONE", "READ_ONLY") else ToolKind.STATE_MODIFYING
        elif "read_only" in t:
            kind = ToolKind.READ_ONLY if t["read_only"] else ToolKind.STATE_MODIFYING
        else:
            first_word = name.lower().split("_")[0]
            if first_word in _WRITE_VERBS:
                kind = ToolKind.STATE_MODIFYING
            elif first_word in _READ_VERBS:
                kind = ToolKind.READ_ONLY
            else:
                kind = ToolKind.STATE_MODIFYING  # safest default

        return ToolSpec(name=name, kind=kind, parameters=params, raw=t)

    def get(self, tool_name: str) -> ToolSpec:
        if tool_name not in self.tools:
            raise KeyError(f"Tool '{tool_name}' not in manifest yet — has TOOL_MANIFEST arrived?")
        return self.tools[tool_name]

    def is_state_modifying(self, tool_name: str) -> bool:
        return self.get(tool_name).kind == ToolKind.STATE_MODIFYING

    def side_effect_str(self, tool_name: str) -> str:
        return self.get(tool_name).kind.value  # feeds straight into make_tool_call(side_effect=...)

    def required_slots_for(self, tool_name: str) -> List[str]:
        spec = self.get(tool_name)
        return [p for p, meta in spec.parameters.items() if meta.get("required")]
