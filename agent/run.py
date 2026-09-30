#!/usr/bin/env python3
"""
agent/run.py — CLI Runner for the TRAX Voice Agent
Usage:
    python -m agent.run dev
    python -m agent.run start
"""

import sys
from livekit import agents
from agent.trax_agent import server

if __name__ == "__main__":
    agents.cli.run_app(server)
