"""The documented Module 3 demo remains runnable end to end."""

from module3.examples.demo_runtime import main


async def test_demo_runtime_completes(capsys):
    await main()
    output = capsys.readouterr().out
    assert "Demo complete" in output
    assert "[PASSED]" in output
