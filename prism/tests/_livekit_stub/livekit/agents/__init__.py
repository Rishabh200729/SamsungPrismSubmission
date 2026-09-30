"""Minimal stand-in for livekit.agents, just enough surface to import agent/trax_agent.py
and inspect it WITHOUT the real SDK installed. Not a functional agent runtime."""
import inspect as _inspect


class Agent:
    def __init__(self, instructions=""):
        self.instructions = instructions


class AgentSession:
    def __init__(self, **kw):
        self.kw = kw
    def on(self, event_name):
        def deco(fn):
            return fn
        return deco
    async def start(self, **kw):
        pass


class AgentServer:
    def rtc_session(self):
        def deco(fn):
            self.entrypoint = fn
            return fn
        return deco


class _FakeFunctionTool:
    """Records (name, description, signature) without altering call behaviour."""
    def __init__(self, description=""):
        self.description = description
    def __call__(self, fn):
        fn._tool_description = self.description
        fn._tool_name = fn.__name__
        fn._tool_signature = _inspect.signature(fn)
        return fn


class llm:
    @staticmethod
    def function_tool(description=""):
        return _FakeFunctionTool(description)

    @staticmethod
    def find_function_tools(obj):
        return [
            getattr(obj, name) for name in dir(obj)
            if callable(getattr(obj, name, None)) and hasattr(getattr(obj, name), "_tool_name")
        ]

    class voice:
        class UserInputTranscribedEvent: pass
        class AgentStateChangedEvent: pass
        class UserStateChangedEvent: pass


class voice:
    UserInputTranscribedEvent = llm.voice.UserInputTranscribedEvent
    AgentStateChangedEvent = llm.voice.AgentStateChangedEvent
    UserStateChangedEvent = llm.voice.UserStateChangedEvent


class JobContext:
    def __init__(self):
        class _Room:
            name = "stub-room"
        self.room = _Room()


class cli:
    @staticmethod
    def run_app(server):
        pass
