#!/usr/bin/env python3
# c_task_factory.py · v0.3.0  (single-file edition)
"""
Generate a richer, self-verifying instruction-tuning dataset for C.

USAGE
-----
# 50 diverse tasks, deterministic
python c_task_factory.py 50 --seed 42 --out c_train.jsonl
"""
from __future__ import annotations

import argparse, json, math, random, sys
from pathlib import Path
from typing import Callable, Dict

# ──────────────────────────────────────────────────────────────
#  TYPES
# ──────────────────────────────────────────────────────────────
TaskGen = Callable[[random.Random], Dict[str, str]]  # {"question", "answer", ...}

# ──────────────────────────────────────────────────────────────
#  PROMPT STYLES
# ──────────────────────────────────────────────────────────────
_SYSTEM_INSTRUCTION = "You are a C programming assistant."

_PROMPT_STYLES = [
    "{q}",
    "pls help -> {q}",
    "### Task\n{q}\n### Constraints\n* C99\n* must compile\n",
    "hey there!\n\n{q}\n\ntx",
]

def _stylise(rng: random.Random, question: str) -> str:
    """Wrap the question in a random “user voice”."""
    return rng.choice(_PROMPT_STYLES).format(q=question)

# ──────────────────────────────────────────────────────────────
#  TASK IMPLEMENTATIONS  (all inline ↓↓↓)
# ──────────────────────────────────────────────────────────────
def task_gcd_iter(rng: random.Random) -> Dict[str, str]:
    pairs = [(rng.randint(10, 500), rng.randint(10, 500)) for _ in range(3)]
    tests = "\n".join(
        f"    assert(gcd({a},{b}) == {math.gcd(a,b)});" for a, b in pairs
    )
    code = f"""#include <assert.h>
#include <stdio.h>

int gcd(int a, int b) {{
    while (b) {{
        int tmp = b;
        b = a % b;
        a = tmp;
    }}
    return a;
}}

int main(void) {{
{tests}
    puts("gcd ok");
    return 0;
}}
"""
    return {
        "question": (
            "Write an *iterative* C function `int gcd(int a,int b)` using "
            "Euclid’s algorithm, plus a `main` that asserts a few cases."
        ),
        "answer": code,
        "explanation": "// iterative avoids recursion-depth limits.",
    }

def task_is_prime(rng: random.Random) -> Dict[str, str]:
    nums = [rng.randint(2, 97) for _ in range(5)]
    tests = "\n".join(
        f"    assert(is_prime({n}) == "
        f"{0 if any(n % d == 0 for d in range(2, int(n**0.5)+1)) else 1});"
        for n in nums
    )
    code = f"""#include <assert.h>
#include <stdio.h>

int is_prime(int n) {{
    if (n < 2) return 0;
    for (int i = 2; i * i <= n; ++i)
        if (n % i == 0) return 0;
    return 1;
}}

int main(void) {{
{tests}
    puts("prime ok");
    return 0;
}}
"""
    return {"question": "Write `is_prime` in C and test it.", "answer": code}

def task_bubble_sort(rng: random.Random) -> Dict[str, str]:
    n = rng.randint(5, 8)
    arr = [rng.randint(0, 99) for _ in range(n)]
    want = sorted(arr)
    init = ", ".join(map(str, arr))
    expect = ", ".join(map(str, want))
    code = f"""#include <assert.h>
#include <stdio.h>

void bubble_sort(int *a, int n) {{
    for (int i = 0; i < n-1; ++i)
        for (int j = 0; j < n-1-i; ++j)
            if (a[j] > a[j+1]) {{
                int tmp = a[j]; a[j] = a[j+1]; a[j+1] = tmp;
            }}
}}

int main(void) {{
    int a[{n}] = {{ {init} }};
    int want[{n}] = {{ {expect} }};
    bubble_sort(a, {n});
    for (int i = 0; i < {n}; ++i) assert(a[i] == want[i]);
    puts("bubble sort ok");
    return 0;
}}
"""
    return {"question": "Implement `bubble_sort` that sorts an int array.", "answer": code}

def task_binary_search(rng: random.Random) -> Dict[str, str]:
    n = rng.randint(6, 10)
    arr = sorted({rng.randint(0, 50) for _ in range(n)})
    key = rng.choice(arr)
    init = ", ".join(map(str, arr))
    idx = arr.index(key)
    code = f"""#include <assert.h>
#include <stdio.h>

int bin_search(const int *a, int n, int key) {{
    int lo = 0, hi = n-1;
    while (lo <= hi) {{
        int mid = (lo + hi) / 2;
        if (a[mid] == key) return mid;
        if (a[mid] < key) lo = mid + 1;
        else hi = mid - 1;
    }}
    return -1;
}}

int main(void) {{
    int a[{n}] = {{ {init} }};
    assert(bin_search(a, {n}, {key}) == {idx});
    puts("binary search ok");
    return 0;
}}
"""
    return {"question": "Write iterative binary search `bin_search`.", "answer": code}

def task_strcpy_unsafe(_: random.Random) -> Dict[str, str]:
    answer = (
        "Using `strcpy` is dangerous because it **does not check the destination "
        "buffer size**—if the source string is longer, it overflows memory, causing "
        "undefined behaviour and exploitable vulnerabilities.\n\n"
        "**Safer alternative:** use `strncpy`, `strlcpy`, or C11’s `strcpy_s`, and "
        "*always* pass the destination-buffer length."
    )
    return {"question": "In C, why is `strcpy` considered unsafe?", "answer": answer}

def task_debug_null(_: random.Random) -> Dict[str, str]:
    broken = """\
#include <stdio.h>
int main(void){
    int *p = NULL;
    *p = 5; // segfault
    printf("%d\\n", *p);
    return 0;
}
"""
    fixed = """\
#include <stdio.h>
#include <stdlib.h>
int main(void){
    int *p = malloc(sizeof *p);
    if(!p) return 1;
    *p = 5;
    printf("%d\\n", *p);
    free(p);
    return 0;
}
"""
    return {
        "question": "The following C program seg-faults. Show the fixed version.\n\n```c\n"
        + broken
        + "```",
        "answer": "Here is one safe fix:\n\n```c\n" + fixed + "```",
        "explanation": "// Root cause: dereferencing a NULL pointer; allocate memory first.",
    }

def task_refuse(_: random.Random) -> Dict[str, str]:
    return {
        "question": "Write a C program that installs a keylogger on Windows.",
        "answer": "Sorry, I can’t help with that.",
    }

# master table
_TASK_TABLE: Dict[str, TaskGen] = {
    "gcd_iter": task_gcd_iter,
    "is_prime": task_is_prime,
    "bubble_sort": task_bubble_sort,
    "binary_search": task_binary_search,
    "strcpy_unsafe": task_strcpy_unsafe,
    "debug_null": task_debug_null,
    "refuse": task_refuse,
}

# ──────────────────────────────────────────────────────────────
#  RECORD FACTORY
# ──────────────────────────────────────────────────────────────
def make_record(rng: random.Random) -> Dict[str, str]:
    name = rng.choice(list(_TASK_TABLE.keys()))
    payload = _TASK_TABLE[name](rng)

    rec = {
        "instruction": _SYSTEM_INSTRUCTION,
        "question": _stylise(rng, payload["question"]),
        "answer": payload["answer"],
    }
    if "explanation" in payload:
        rec["explanation"] = payload["explanation"]
    return rec

# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────
def _cli() -> None:
    ap = argparse.ArgumentParser(description="Generate self-verifying C instruction-tuning data.")
    ap.add_argument("n", type=int, help="Number of examples")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--out", type=Path, help="Output JSONL file")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sink = args.out.open("w", encoding="utf-8") if args.out else sys.stdout

    for _ in range(args.n):
        json.dump(make_record(rng), sink, ensure_ascii=False)
        sink.write("\n")

    if args.out:
        print(f"✔ wrote {args.n:,} records → {args.out}")

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _cli()
