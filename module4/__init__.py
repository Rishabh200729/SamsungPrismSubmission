"""Module 4 — Multimodal Grounding & Protocol Compliance.

Public surface for other modules (mainly Module 2) to import:

    from module4.clarifier import GroundingPipeline
    from module4.visual_buffer import SessionVisualBuffer
    from module4.constrained_output import enforce_json_schema, ValidateRepairBackend
    from module4.adapter import Module4Adapter

See module4/README.md for the integration contract, the race-condition note on
END_OF_TURN handling, and full research references.
"""
