#!/usr/bin/env python3
"""
scripts/audit_benchmark_leakage.py — pre-submission compliance scan.

The official rules say: "Don't hardcode, memorize, or fine-tune on benchmark test items.
FDB-v3 is public, so its answers are visible; any submission that pattern-matches test items
instead of solving them is disqualified, and we check."

A reviewer who "checks" will do roughly what this script does: take the public expected
answers (and the recorded user speech) and grep the submission's prompts and tool
descriptions for them.  Run this before every submission; it exits 1 if anything is found.

What it flags
  LEAK      an expected argument value (e.g. an ID, a search phrase, an address) that appears
            as a quoted literal inside a string of the scanned files, and is NOT vocabulary
            the organizers' own reference agent already ships.
  SPEECH    a run of >= N consecutive words copied from a recorded user utterance.
  (ok)      shown, not failed: values that the reference agent's tool template itself uses
            (tool/parameter names, its docstring examples such as 'BOB12', 'driving').

What it deliberately does NOT do
  It cannot tell you whether a *rule* is legitimate (e.g. "IDs have no hyphens" is general
  linguistic guidance; listing the exact filter keys the benchmark expects is a judgment
  call).  It only finds *verbatim overlap*, which is what is mechanically checkable.

Usage
  python scripts/audit_benchmark_leakage.py
  python scripts/audit_benchmark_leakage.py --benchmark external/FDB-v3/v3/benchmark_data_v2.json \\
      --reference external/FDB-v3/v3/lk_agent_tool.py --files agent/*.py --json report.json
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BENCHMARK = "external/FDB-v3/v3/benchmark_data_v2.json"
DEFAULT_REFERENCE = "external/FDB-v3/v3/lk_agent_tool.py"
# The submission's actual prompt/description text.  (The benchmark script only imports these
# and otherwise prints analysis prose, so it is not scanned by default - pass --files to add it.)
DEFAULT_FILES = [
    "agent/prism_agent.py",
    "agent/tool_specs.py",
]

MIN_VALUE_LEN = 2          # ignore 1-char values
MIN_STRING_LEN = 12        # only inspect string literals long enough to be prompt text
SPEECH_SHINGLE_WORDS = 5   # consecutive copied words that count as verbatim speech


@dataclass
class Finding:
    kind: str                      # LEAK | SPEECH | OK_VOCAB
    value: str
    file: str
    line: int
    scenarios: list[str] = field(default_factory=list)
    note: str = ""


# ------------------------------------------------------------------ loading
def load_benchmark(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["scenarios"] if isinstance(data, dict) and "scenarios" in data else data


def expected_values(scenarios: list[dict]) -> dict[str, set[str]]:
    """lower-cased expected argument value -> scenario ids.  Dynamic '$RESULT' refs skipped."""
    out: dict[str, set[str]] = {}
    for s in scenarios:
        for call in s.get("expected_tool_calls", []):
            for v in (call.get("args") or {}).values():
                if isinstance(v, bool):
                    continue
                text = str(v).strip()
                if not text or text.startswith("$") or len(text) < MIN_VALUE_LEN:
                    continue
                if isinstance(v, (int, float)) and len(text) < 3:
                    continue           # tiny numbers ("2") are not identifying
                out.setdefault(text.lower(), set()).add(s["id"])
    return out


def _norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def speech_shingles(scenarios: list[dict], n: int) -> dict[tuple[str, ...], str]:
    shingles: dict[tuple[str, ...], str] = {}
    for s in scenarios:
        for turn in s.get("dialogue", []):
            for key in ("user", "user_annotated"):
                words = _norm_words(turn.get(key) or "")
                for i in range(len(words) - n + 1):
                    shingles.setdefault(tuple(words[i:i + n]), s["id"])
    return shingles


def string_literals(path: Path) -> list[tuple[str, int]]:
    """(text, line) for every string constant in a Python file (prompts, descriptions,
    docstrings).  Non-Python files are scanned line by line."""
    src = path.read_text(encoding="utf-8")
    if path.suffix != ".py":
        return [(line, i + 1) for i, line in enumerate(src.splitlines())]
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append((node.value, node.lineno))
    return out


def reference_vocabulary(path: Path | None) -> set[str]:
    """Every quoted literal in the organizers' reference agent = legitimate public vocabulary."""
    if path is None or not path.exists():
        return set()
    vocab: set[str] = set()
    for text, _ in string_literals(path):
        for m in re.finditer(r"""['"]([^'"]{2,40})['"]""", text):
            vocab.add(m.group(1).strip().lower())
    # function and parameter identifiers are part of the public tool interface too
    # (e.g. `max_price`), even though they are not quoted literals
    if path.suffix == ".py":
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                vocab.add(node.name.lower())
                vocab.update(a.arg.lower() for a in node.args.args + node.args.kwonlyargs)
    return vocab


# ------------------------------------------------------------------ scanning
def _quoted_or_phrase(value: str, text_lower: str) -> bool:
    quoted = re.search(r"""['"]""" + re.escape(value) + r"""['"]""", text_lower)
    if quoted:
        return True
    # multi-word phrases are distinctive enough to match unquoted (whole words only)
    return " " in value and re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text_lower) is not None


def scan_file(path: Path, expected: dict[str, set[str]], vocab: set[str],
              shingles: dict[tuple[str, ...], str], n: int) -> list[Finding]:
    findings: list[Finding] = []
    rel = str(path.relative_to(ROOT)) if path.is_absolute() and ROOT in path.parents else str(path)
    seen_speech: set[tuple[int, str]] = set()
    for text, line in string_literals(path):
        if len(text) < MIN_STRING_LEN:
            continue
        low = text.lower()
        for value, ids in expected.items():
            if _quoted_or_phrase(value, low):
                kind = "OK_VOCAB" if value in vocab else "LEAK"
                note = ("shipped in the organizers' reference agent" if kind == "OK_VOCAB"
                        else "verbatim expected answer of a public scenario")
                findings.append(Finding(kind, value, rel, line, sorted(ids), note))
        words = _norm_words(text)
        for i in range(len(words) - n + 1):
            sc = shingles.get(tuple(words[i:i + n]))
            if sc and (line, sc) not in seen_speech:
                seen_speech.add((line, sc))
                findings.append(Finding("SPEECH", " ".join(words[i:i + n]), rel, line, [sc],
                                        "copied from a recorded user utterance"))
    return findings


def audit(benchmark: Path, reference: Path | None, files: Iterable[Path],
          n: int = SPEECH_SHINGLE_WORDS) -> list[Finding]:
    scenarios = load_benchmark(benchmark)
    expected = expected_values(scenarios)
    vocab = reference_vocabulary(reference)
    shingles = speech_shingles(scenarios, n)
    out: list[Finding] = []
    for f in files:
        out.extend(scan_file(f, expected, vocab, shingles, n))
    return out


# ------------------------------------------------------------------ CLI
def _resolve(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pat in patterns:
        matches = glob.glob(str(ROOT / pat)) or glob.glob(pat)
        paths.extend(Path(m) for m in sorted(matches))
    return [p for p in paths if p.is_file()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    ap.add_argument("--reference", default=DEFAULT_REFERENCE,
                    help="organizers' lk_agent_tool.py; its quoted literals are treated as public vocabulary")
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    ap.add_argument("--shingle", type=int, default=SPEECH_SHINGLE_WORDS)
    ap.add_argument("--json", metavar="PATH", help="also write findings as JSON")
    args = ap.parse_args(argv)

    bench = Path(args.benchmark) if Path(args.benchmark).is_absolute() else ROOT / args.benchmark
    if not bench.exists():
        print(f"benchmark data not found: {bench}\n"
              f"clone Full-Duplex-Bench into external/FDB-v3 first (see README).", file=sys.stderr)
        return 2
    ref = Path(args.reference) if Path(args.reference).is_absolute() else ROOT / args.reference
    if not ref.exists():
        print(f"note: reference agent not found ({ref}); nothing is whitelisted, expect extra hits",
              file=sys.stderr)
        ref = None

    files = _resolve(args.files)
    if not files:
        print("no files to scan", file=sys.stderr)
        return 2
    findings = audit(bench, ref, files, args.shingle)

    leaks = [f for f in findings if f.kind in ("LEAK", "SPEECH")]
    ok = [f for f in findings if f.kind == "OK_VOCAB"]
    print(f"scanned {len(files)} file(s) against {bench.name}")
    for f in leaks:
        print(f"  {f.kind:6} {f.file}:{f.line}  {f.value!r}  <- {', '.join(f.scenarios[:4])}  ({f.note})")
    if ok:
        vals = sorted({f.value for f in ok})
        print(f"  (ok) {len(vals)} value(s) are public interface vocabulary: {', '.join(vals[:12])}"
              f"{' …' if len(vals) > 12 else ''}")
    print(f"RESULT: {len(leaks)} finding(s)" + ("  — FIX BEFORE SUBMITTING" if leaks else "  — clean"))
    if args.json:
        Path(args.json).write_text(json.dumps([asdict(f) for f in findings], indent=2), encoding="utf-8")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
