"""
TRPGate precision + correction_epoch tests.

Bare "no", "rather" and "like" are everyday words.  Treating them as self-correction
markers put ordinary requests into REPAIRING (a 900 ms hold) and - now that supersession is
driven by correction_epoch - could wrongly discard a legitimate call.  These tests pin both
directions: ordinary speech stays quiet, genuine repairs and fillers still fire.
"""

import unittest

from prism import TRPState
from prism.trp_gate import TRPGate, _CORRECTION_TERMS_RE, _FILLER_TERMS_RE


class TestLexiconPrecision(unittest.IsolatedAsyncioTestCase):

    async def feed(self, text, is_final=False):
        gate = TRPGate()
        await gate.on_transcript_token(text, is_final=is_final)
        return gate

    async def test_ordinary_speech_is_not_a_repair(self):
        for phrase in [
            "I'd like to book a flight to Boston",
            "there is no problem with that date",
            "I would rather have the window seat",
            "No problem, go ahead",
            "I have no preference",
            "do you know the price",
            "the number is 55",
            "I know",
        ]:
            with self.subTest(phrase=phrase):
                gate = await self.feed(phrase)
                self.assertEqual(gate.state, TRPState.LISTENING)
                self.assertEqual(gate._correction_tier, 0)
                self.assertEqual(gate.correction_epoch, 0)

    async def test_genuine_repairs_still_fire(self):
        for phrase in [
            "no wait, make it Friday",
            "Miami \u2014 no, Orlando",
            "Miami... no, Orlando",
            "flights to Paris, or rather Rome",
            "rather, let's fly out of SFO",
            "actually make that London",
            "scratch that, look for apartments under 2000",
            "sorry, change date to Oct 7",
            "oops, wrong flight id",
            "instead of Paris make it Rome",
            "i mean 2 bedrooms",
        ]:
            with self.subTest(phrase=phrase):
                gate = await self.feed(phrase)
                self.assertEqual(gate.state, TRPState.REPAIRING)
                self.assertEqual(gate._correction_tier, 1)
                self.assertEqual(gate.correction_epoch, 1)

    async def test_fillers_delay_but_do_not_repair(self):
        for phrase in ["flights to um Boston", "uh I want a window seat",
                       "I want, like, a window seat", "it is, you know, near the station"]:
            with self.subTest(phrase=phrase):
                gate = await self.feed(phrase)
                self.assertEqual(gate.state, TRPState.LISTENING)
                self.assertEqual(gate._correction_tier, 2)
                self.assertEqual(gate.correction_epoch, 0)   # fillers are not repairs

    def test_backward_compatible_alias_exists(self):
        from prism.trp_gate import _EDITING_TERMS_RE
        self.assertIs(_EDITING_TERMS_RE, _CORRECTION_TERMS_RE)


class TestCorrectionEpoch(unittest.IsolatedAsyncioTestCase):

    async def test_epoch_is_monotonic_and_survives_turn_reset(self):
        gate = TRPGate()
        await gate.on_transcript_token("no wait, Berlin", is_final=False)
        self.assertEqual(gate.correction_epoch, 1)
        gate.reset_for_new_turn()
        self.assertEqual(gate.correction_epoch, 1, "epoch must never go backwards")
        await gate.on_transcript_token("actually make that Rome", is_final=False)
        self.assertEqual(gate.correction_epoch, 2)

    async def test_cumulative_interim_transcripts_count_a_marker_once(self):
        """ASR interims grow: 'no wait' -> 'no wait make' -> 'no wait make it Friday'."""
        gate = TRPGate()
        for chunk in ["no wait", "no wait make", "no wait make it Friday"]:
            await gate.on_transcript_token(chunk, is_final=False)
        self.assertEqual(gate.correction_epoch, 1)
        await gate.on_transcript_token("no wait make it Friday", is_final=True)
        self.assertEqual(gate.correction_epoch, 1)

    async def test_a_second_distinct_repair_counts_again(self):
        gate = TRPGate()
        await gate.on_transcript_token("Rome no wait Oslo", is_final=False)
        await gate.on_transcript_token("Rome no wait Oslo actually Bergen", is_final=False)
        self.assertEqual(gate.correction_epoch, 2)

    async def test_no_marker_no_epoch_change(self):
        gate = TRPGate()
        await gate.on_transcript_token("search flights to Paris on Friday", is_final=True)
        self.assertEqual(gate.correction_epoch, 0)


if __name__ == "__main__":
    unittest.main()
