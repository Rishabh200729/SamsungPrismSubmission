"""Validated scenario tool definitions."""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

class ToolSideEffect(str, Enum):
    READ_ONLY = "READ_ONLY"
    STATE_MODIFYING = "STATE_MODIFYING"

class ToolDefinition(BaseModel):
    name: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    side_effect: ToolSideEffect = ToolSideEffect.READ_ONLY
    idempotency_required: bool = False
    cancellable: bool = True
    timeout_ms: int | None = None

class ToolManifestRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self.version = 0
    def install(self, raw_tools: list[dict[str, Any]]) -> None:
        self._tools = {tool.name: tool for tool in (ToolDefinition.model_validate(x) for x in raw_tools)}
        self.version += 1
    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)
    def validate_args(self, tool: ToolDefinition, arguments: dict[str, Any]) -> None:
        required = tool.input_schema.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing:
            raise ValueError(f"missing tool arguments: {missing}")
