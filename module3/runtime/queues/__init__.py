"""
module3/runtime/queues/__init__.py
"""
from .input_queue import InputQueue
from .output_queue import OutputQueue
from .priority_input_queue import PriorityInputQueue

__all__ = ["InputQueue", "OutputQueue", "PriorityInputQueue"]
