"""module3/runtime/clock/__init__.py"""
from .base import Clock
from .real_clock import RealClock
from .virtual_clock import VirtualClock
__all__ = ["Clock", "RealClock", "VirtualClock"]
