#!/usr/bin/env python3
# c_task_factory.py · v0.1.0
"""
Build a self-verifying instruction-tuning dataset for C.

Each JSONL record looks like:
{
  "instruction": "You are a C programming assistant.",
  "question":    "Write a C function that returns the greatest common divisor ...",
  "answer":      "<full C source with unit tests>"
}

Usage
-----
# 1 000 tasks, deterministic, write to file
python c_task_factory.py 1000 --seed 123 --out c_train.jsonl

# quick sanity-print 5 tasks to console
python c_task_factory.py 5
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# ──────────────────────────────────────────────────────────────
#  TASK DEFINITIONS
# ──────────────────────────────────────────────────────────────
TaskGen = Callable[[random.Random], Tuple[str, str]]  # -> (question, answer)

def _wrap(code_body: str) -> str:
    """Add common #includes once per answer."""
    return (
        "#include <assert.h>\n"
        "#include <stdio.h>\n\n"
        f"{code_body.strip()}\n"
    )

def task_gcd(rng: random.Random) -> Tuple[str, str]:
    pairs = [(rng.randint(10, 200), rng.randint(10, 200)) for _ in range(3)]
    q = "Write a C function `int gcd(int a, int b)` that returns the greatest common divisor of `a` and `b` using Euclid's algorithm. Include a `main` that asserts the function on a few cases."
    fn = (
        "int gcd(int a, int b) {\n"
        "    return b == 0 ? a : gcd(b, a % b);\n"
        "}\n"
    )
    tests = "\n".join(
        f"    assert(gcd({a},{b}) == {math.gcd(a,b)});"
        for a, b in pairs
    )
    main = (
        "int main(void) {\n"
        f"{tests}\n"
        "    puts(\"gcd tests passed\");\n"
        "    return 0;\n"
        "}\n"
    )
    return q, _wrap(fn + "\n" + main)

def task_factorial(rng: random.Random) -> Tuple[str, str]:
    nums = [rng.randint(0, 10) for _ in range(3)]
    q = "Write a C function `unsigned long factorial(unsigned int n)` that returns `n!` recursively. Provide a `main` with asserts."
    fn = (
        "unsigned long factorial(unsigned int n) {\n"
        "    return n == 0 ? 1UL : n * factorial(n - 1);\n"
        "}\n"
    )
    tests = "\n".join(
        f"    assert(factorial({n}) == {math.factorial(n)}UL);"
        for n in nums
    )
    main = (
        "int main(void) {\n"
        f"{tests}\n"
        "    puts(\"factorial tests passed\");\n"
        "    return 0;\n"
        "}\n"
    )
    return q, _wrap(fn + "\n" + main)

def task_is_prime(rng: random.Random) -> Tuple[str, str]:
    nums = [rng.randint(2, 50) for _ in range(5)]
    q = "Write a C function `int is_prime(int n)` that returns 1 if `n` is prime, else 0. Add a `main` that asserts several inputs."
    fn = (
        "int is_prime(int n) {\n"
        "    if (n < 2) return 0;\n"
        "    for (int i = 2; i * i <= n; ++i) {\n"
        "        if (n % i == 0) return 0;\n"
        "    }\n"
        "    return 1;\n"
        "}\n"
    )
    tests = "\n".join(
        f"    assert(is_prime({n}) == {1 if all(n%d for d in range(2,int(n**0.5)+1)) else 0});"
        for n in nums
    )
    main = (
        "int main(void) {\n"
        f"{tests}\n"
        "    puts(\"prime tests passed\");\n"
        "    return 0;\n"
        "}\n"
    )
    return q, _wrap(fn + "\n" + main)

TASK_TABLE: Dict[str, TaskGen] = {
    "gcd": task_gcd,
    "factorial": task_factorial,
    "is_prime": task_is_prime,
}

# ──────────────────────────────────────────────────────────────
#  RECORD FACTORY
# ──────────────────────────────────────────────────────────────
INSTRUCTION = "You are a C programming assistant."

def make_record(rng: random.Random) -> dict:
    name, gen = rng.choice(list(TASK_TABLE.items()))
    question, answer = gen(rng)
    return {"instruction": INSTRUCTION, "question": question, "answer": answer}

# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────
def _cli() -> None:
    ap = argparse.ArgumentParser(description="Generate self-verifying C instruction-tuning data.")
    ap.add_argument("n", type=int, help="Number of examples to generate")
    ap.add_argument("--seed", type=int, help="Random seed")
    ap.add_argument("--out", type=Path, help="Path to JSONL output")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sink = args.out.open("w", encoding="utf-8") if args.out else sys.stdout

    for _ in range(args.n):
        rec = make_record(rng)
        json.dump(rec, sink, ensure_ascii=False)
        sink.write("\n")

    if args.out:
        print(f"✔ wrote {args.n:,} records → {args.out}")

if __name__ == "__main__":
    _cli()
