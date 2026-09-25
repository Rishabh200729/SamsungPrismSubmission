import unittest
import asyncio
from prism import TRPState
from prism.trp_gate import TRPGate, _EDITING_TERMS_RE, _INCOMPLETE_TAIL_RE


class TestTRPGate(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.gate = TRPGate()
        self.state_history = []

        async def listener(state):
            self.state_history.append(state)

        self.gate.add_state_listener(listener)

    def test_editing_term_regex_coverage(self):
        """Verify Levelt (1983) editing terms are matched robustly."""
        test_phrases = [
            "no wait, book for tomorrow",
            "actually make that London",
            "sorry, change date to Oct 7",
            "i mean 2 bedrooms",
            "i meant business class",
            "rather, let's fly out of SFO",
            "scratch that, look for apartments under 2000",
            "oops, wrong flight id",
            "instead of Paris make it Rome",
        ]
        for phrase in test_phrases:
            self.assertIsNotNone(
                _EDITING_TERMS_RE.search(phrase),
                f"Failed to match editing term in: '{phrase}'",
            )

    def test_syntactic_incompleteness_heuristic(self):
        """Dangling prepositions and conjunctions must be classified incomplete."""
        dangling_phrases = [
            "search for flights to",
            "I want to book on",
            "apartments in NYC with",
            "show me tickets and",
            "transfer the money because",
        ]
        for phrase in dangling_phrases:
            self.assertFalse(
                self.gate._is_syntactically_complete(phrase),
                f"Should be incomplete: '{phrase}'",
            )

        complete_phrases = [
            "book the flight to London.",
            "search apartments in Boston under 2500",
            "update search filter max price 3000",
        ]
        for phrase in complete_phrases:
            self.assertTrue(
                self.gate._is_syntactically_complete(phrase),
                f"Should be complete: '{phrase}'",
            )

    async def test_editing_term_triggers_repairing_state(self):
        """Editing term token shifts state to REPAIRING."""
        self.assertEqual(self.gate.state, TRPState.LISTENING)
        await self.gate.on_transcript_token("no wait make that Friday", is_final=False)
        self.assertEqual(self.gate.state, TRPState.REPAIRING)
        self.assertIn(TRPState.REPAIRING, self.state_history)

    async def test_silence_threshold_distinction(self):
        """In LISTENING, 300ms silence confirms TRP; in REPAIRING, 300ms holds floor."""
        # 1. In LISTENING state with complete buffer
        await self.gate.on_transcript_token("search flights to Paris on Friday", is_final=False)
        await self.gate.on_silence_detected(250)
        self.assertEqual(self.gate.state, TRPState.LISTENING)  # Below 300ms threshold
        await self.gate.on_silence_detected(350)
        self.assertEqual(self.gate.state, TRPState.TRP_CONFIRMED)

        # 2. Reset and transition to REPAIRING
        self.gate.reset_for_new_turn()
        self.assertEqual(self.gate.state, TRPState.LISTENING)
        await self.gate.on_transcript_token("actually make that Rome next Monday", is_final=False)
        self.assertEqual(self.gate.state, TRPState.REPAIRING)

        # 400ms silence in REPAIRING must NOT confirm TRP (needs 900ms)
        await self.gate.on_silence_detected(400)
        self.assertEqual(self.gate.state, TRPState.REPAIRING)

        # 950ms silence in REPAIRING confirms TRP
        await self.gate.on_silence_detected(950)
        self.assertEqual(self.gate.state, TRPState.TRP_CONFIRMED)

    async def test_force_abort(self):
        """force_abort shifts state to ABORTED and invokes listeners."""
        await self.gate.force_abort()
        self.assertEqual(self.gate.state, TRPState.ABORTED)
        self.assertIn(TRPState.ABORTED, self.state_history)

    def test_reset_for_new_turn(self):
        """reset_for_new_turn resets buffer and state to LISTENING."""
        self.gate._state = TRPState.TRP_CONFIRMED
        self.gate._buffer = "some completed query"
        self.gate.reset_for_new_turn()
        self.assertEqual(self.gate.state, TRPState.LISTENING)
        self.assertEqual(self.gate._buffer, "")


if __name__ == "__main__":
    unittest.main()
