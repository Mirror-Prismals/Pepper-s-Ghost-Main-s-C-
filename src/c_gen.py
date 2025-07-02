#!/usr/bin/env python3
# synthetic_c.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—C source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* Tracks typedefs and structs to reference in functions
* --out to save directly to disk

Usage
-----
python synthetic_c.py 200
python synthetic_c.py 300 --seed 42 --out fake.c
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

__version__ = "0.1.0"

@dataclass(frozen=True)
class CConfig:
    loc: int = 200                 # approximate number of lines
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":       0.10,
        "include":       0.10,
        "define_macro":  0.05,
        "typedef":       0.10,
        "struct":        0.10,
        "var_decl":      0.15,
        "func_decl":     0.10,
        "func_def":      0.10,
        "main":          0.10,
        "conditional":   0.05,
        "loop":          0.05,
    })

GeneratorFn = Callable[[Dict], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def inner(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return inner

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

LETTERS = "abcdefghijklmnopqrstuvwxyz"
C_TYPES = ["int", "long", "float", "double", "char"]

def fresh_name(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choice(LETTERS) for _ in range(length))

def random_value(rng: random.Random, ctype: str) -> str:
    if ctype == "char":
        return f"'{rng.choice(LETTERS)}'"
    if ctype in ("int", "long"):
        v = rng.randint(0, 100)
        return f"{v}{'L' if ctype=='long' else ''}"
    # float or double
    return f"{rng.uniform(0,100):.2f}"

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["// TODO", "// FIXME", "// NOTE", "// HACK"]
    text = fresh_name(rng, rng.randint(3,8))
    return f"{rng.choice(tags)}: {text}\n"

@register("include")
def gen_include(state: Dict) -> str:
    rng = state["rng"]
    hdr = rng.choice(["<stdio.h>", "<stdlib.h>", "<string.h>", "<math.h>"])
    return f"#include {hdr}\n"

@register("define_macro")
def gen_define_macro(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng).upper()
    val = rng.randint(1, 100)
    return f"#define {name} {val}\n"

@register("typedef")
def gen_typedef(state: Dict) -> str:
    rng = state["rng"]
    base = rng.choice(C_TYPES)
    alias = fresh_name(rng, rng.randint(3,6))
    state["typedefs"].add(alias)
    return f"typedef {base} {alias};\n"

@register("struct")
def gen_struct(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng, rng.randint(3,6)).capitalize()
    field_count = rng.randint(1,3)
    fields = []
    for _ in range(field_count):
        t = rng.choice(C_TYPES)
        fn = fresh_name(rng, rng.randint(3,6))
        fields.append(f"    {t} {fn};")
    state["structs"].add(name)
    body = "\n".join(fields)
    return f"typedef struct {name} {{\n{body}\n}} {name};\n"

@register("var_decl")
def gen_var_decl(state: Dict) -> str:
    rng = state["rng"]
    # choose a type from basic or typedefs or structs
    types = C_TYPES + list(state["typedefs"]) + list(state["structs"])
    ctype = rng.choice(types)
    name = fresh_name(rng)
    val = random_value(rng, rng.choice(C_TYPES)) if rng.random() < 0.5 else ""
    init = f" = {val}" if val else ""
    return f"{ctype} {name}{init};\n"

@register("func_decl")
def gen_func_decl(state: Dict) -> str:
    rng = state["rng"]
    ret = rng.choice(C_TYPES + list(state["typedefs"]))
    name = fresh_name(rng)
    # parameters
    n = rng.randint(0,2)
    params = []
    for _ in range(n):
        ptype = rng.choice(C_TYPES + list(state["typedefs"]))
        pname = fresh_name(rng)
        params.append(f"{ptype} {pname}")
    params_str = ", ".join(params) if params else "void"
    state["funcs"].add((ret, name, params_str))
    return f"{ret} {name}({params_str});\n"

@register("func_def")
def gen_func_def(state: Dict) -> str:
    rng = state["rng"]
    if not state["funcs"]:
        return ""
    ret, name, params_str = rng.choice(list(state["funcs"]))
    lines = [f"{ret} {name}({params_str}) {{\n"]
    # simple body: return or variable
    if ret != "void":
        val = random_value(rng, rng.choice(C_TYPES))
        lines.append(f"    return {val};\n")
    else:
        lines.append("    // function body\n")
    lines.append("}\n\n")
    return "".join(lines)

@register("main")
def gen_main(state: Dict) -> str:
    if state["main_written"]:
        return ""
    state["main_written"] = True
    lines = ["int main(void) {\n"]
    # call up to 2 functions or declare vars
    rng = state["rng"]
    for _ in range(rng.randint(1,3)):
        if state["funcs"] and rng.random() < 0.5:
            _, fname, pstr = rng.choice(list(state["funcs"]))
            args = ", ".join("0" for _ in pstr.split(",")) if pstr != "void" else ""
            lines.append(f"    {fname}({args});\n")
        else:
            # simple printf
            lines.append("    printf(\"Hello, world!\\n\");\n")
    lines.append("    return 0;\n")
    lines.append("}\n")
    return "".join(lines)

@register("conditional")
def gen_conditional(state: Dict) -> str:
    rng = state["rng"]
    var = fresh_name(rng)
    cmp_val = rng.randint(0,10)
    return (
        f"if ({var} > {cmp_val}) {{\n"
        f"    {var} = {cmp_val};\n"
        f"}} else {{\n"
        f"    {var} += {cmp_val};\n"
        f"}}\n"
    )

@register("loop")
def gen_loop(state: Dict) -> str:
    rng = state["rng"]
    var = fresh_name(rng)
    count = rng.randint(1,5)
    return (
        f"for (int {var} = 0; {var} < {count}; ++{var}) {{\n"
        f"    // loop body\n"
        f"}}\n"
    )

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_c(cfg: CConfig) -> str:
    rng = random.Random(cfg.seed)
    state = {
        "rng": rng,
        "typedefs": set(),     # alias names
        "structs": set(),      # struct names
        "funcs": set(),        # (ret, name, params)
        "main_written": False,
    }
    parts: List[str] = ["/* Auto-generated C code */\n\n"]
    lines = parts[0].count("\n")
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        snippet = _REGISTRY[kind](state)
        if not snippet:
            continue
        parts.append(snippet)
        lines += snippet.count("\n")

    # ensure main exists
    if not state["main_written"]:
        parts.append(gen_main(state))
    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic C source file.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx. number of lines")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .c")
    args = p.parse_args()

    cfg = CConfig(loc=args.loc, seed=args.seed)
    code = build_c(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated C code to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
