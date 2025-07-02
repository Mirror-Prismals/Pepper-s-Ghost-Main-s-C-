#!/usr/bin/env python3
# c_gen.py · v0.2.1
"""
Generate synthetic—yet *compilable-ish*—C source files for LM pre-training.

Highlights
----------
* Deterministic output with --seed
* Rich C surface: enums, unions, pointers, switch, function-like macros
* Per-file style randomisation (K&R / Allman / GNU)
* --weights to tweak construct distribution on the fly
* Optional --check to run a compile smoke-test (gcc/clang)

Usage
-----
python c_gen.py 300
python c_gen.py 400 --seed 123 --style allman --weights switch=0.08,enum=0.05 --check
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

__version__ = "0.2.1"

# ──────────────────────────────────────────────────────────────
# Config & registry
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CConfig:
    loc: int = 200
    seed: Optional[int] = None
    style: str = "auto"          # auto|kr|allman|gnu
    check: bool = False
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":        0.07,
        "include":        0.07,
        "define_macro":   0.04,
        "define_macro_f": 0.04,
        "typedef":        0.08,
        "enum":           0.06,
        "union":          0.05,
        "struct":         0.07,
        "var_decl":       0.12,
        "func_decl":      0.08,
        "func_def":       0.08,
        "switch":         0.05,
        "main":           0.07,
        "conditional":    0.06,
        "loop":           0.06,
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
BASE_CTYPES = ["int", "long", "float", "double", "char"]
POINTER_CHANCE = 0.25           # probability a type is emitted as a pointer

STYLE_TABLE = {
    "kr":     {"indent": "    ", "brace_same": True},
    "allman": {"indent": "    ", "brace_same": False},
    "gnu":    {"indent": "  ",   "brace_same": True},
}

def fresh_name(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choice(LETTERS) for _ in range(length))

def choose_ctype(rng: random.Random, extra: List[str]) -> str:
    base = rng.choice(BASE_CTYPES + extra)
    if rng.random() < POINTER_CHANCE and not base.endswith("*"):
        return base + "*"
    return base

def random_value(rng: random.Random, ctype: str) -> str:
    if ctype.endswith("*"):
        return "NULL"
    if ctype == "char":
        return f"'{rng.choice(LETTERS)}'"
    if ctype in ("int", "long"):
        v = rng.randint(0, 100)
        return f"{v}{'L' if ctype == 'long' else ''}"
    return f"{rng.uniform(0, 100):.2f}"

def brace_line(state: Dict, header: str) -> str:
    style = state["style"]
    if STYLE_TABLE[style]["brace_same"]:
        return f"{header} {{\n"
    return f"{header}\n{{\n"

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state):
    rng = state["rng"]
    tags = ["// TODO", "// FIXME", "// NOTE", "// HACK"]
    return f"{rng.choice(tags)}: {fresh_name(rng, rng.randint(3, 8))}\n"

@register("include")
def gen_include(state):
    rng = state["rng"]
    hdrs = ["<stdio.h>", "<stdlib.h>", "<string.h>", "<math.h>", "<stddef.h>"]
    # allow repeats after we've used every header once
    available = [h for h in hdrs if h not in state["headers"]]
    if not available:
        state["headers"].clear()
        available = hdrs
    hdr = rng.choice(available)
    state["headers"].add(hdr)
    return f"#include {hdr}\n"

@register("define_macro")
def gen_define_macro(state):
    rng = state["rng"]
    name = fresh_name(rng).upper()
    return f"#define {name} {rng.randint(1, 100)}\n"

@register("define_macro_f")
def gen_define_macro_func(state):
    rng = state["rng"]
    name = fresh_name(rng).upper()
    param = fresh_name(rng, 1)
    return f"#define {name}({param}) (({param}) * ({param}))\n"

@register("typedef")
def gen_typedef(state):
    rng = state["rng"]
    alias = fresh_name(rng, rng.randint(3, 6))
    state["typedefs"].add(alias)
    return f"typedef {rng.choice(BASE_CTYPES)} {alias};\n"

@register("enum")
def gen_enum(state):
    rng = state["rng"]
    name = fresh_name(rng, rng.randint(3, 6)).capitalize()
    items = ", ".join(f"{name.upper()}_{i}" for i in range(rng.randint(2, 4)))
    state["typedefs"].add(name)
    return f"typedef enum {{ {items} }} {name};\n"

@register("union")
def gen_union(state):
    rng = state["rng"]
    name = fresh_name(rng, rng.randint(3, 6)).capitalize()
    fields = [f"    {t} {fresh_name(rng)};" for t in rng.sample(BASE_CTYPES, 2)]
    state["structs"].add(name)
    return f"typedef union {name} {{\n" + "\n".join(fields) + f"\n}} {name};\n"

@register("struct")
def gen_struct(state):
    rng = state["rng"]
    name = fresh_name(rng, rng.randint(3, 6)).capitalize()
    lines = [
        f"    {rng.choice(BASE_CTYPES)} {fresh_name(rng, rng.randint(3, 6))};"
        for _ in range(rng.randint(1, 3))
    ]
    state["structs"].add(name)
    return f"typedef struct {name} {{\n" + "\n".join(lines) + f"\n}} {name};\n"

@register("var_decl")
def gen_var_decl(state):
    rng = state["rng"]
    ctype = choose_ctype(rng, list(state["typedefs"]))
    name = fresh_name(rng)
    init = ""
    if not ctype.endswith("*") and rng.random() < 0.5:
        init = f" = {random_value(rng, rng.choice(BASE_CTYPES))}"
    return f"{ctype} {name}{init};\n"

@register("func_decl")
def gen_func_decl(state):
    rng = state["rng"]
    ret = choose_ctype(rng, list(state["typedefs"]))
    name = fresh_name(rng)
    params = [
        f"{choose_ctype(rng, list(state['typedefs']))} {fresh_name(rng)}"
        for _ in range(rng.randint(0, 2))
    ]
    params_str = ", ".join(params) if params else "void"
    state["funcs"].add((ret, name, params_str))
    return f"{ret} {name}({params_str});\n"

@register("func_def")
def gen_func_def(state):
    rng = state["rng"]
    if not state["funcs"]:
        return ""
    ret, name, params_str = rng.choice(list(state["funcs"]))
    indent = STYLE_TABLE[state["style"]]["indent"]
    body = f"{indent}// function body\n" if ret == "void" else f"{indent}return {random_value(rng, rng.choice(BASE_CTYPES))};\n"
    return brace_line(state, f"{ret} {name}({params_str})") + body + "}\n\n"

@register("switch")
def gen_switch(state):
    rng = state["rng"]
    var = fresh_name(rng)
    indent = STYLE_TABLE[state["style"]]["indent"]
    cases = [
        f"{indent}case {i}:\n{indent*2}{var} += {i};\n{indent*2}break;\n"
        for i in range(rng.randint(2, 4))
    ]
    default = f"{indent}default:\n{indent*2}break;\n}}\n"
    return brace_line(state, f"switch ({var})") + "".join(cases) + default

@register("conditional")
def gen_conditional(state):
    rng = state["rng"]
    var = fresh_name(rng)
    cmp_val = rng.randint(0, 10)
    indent = STYLE_TABLE[state["style"]]["indent"]
    part1 = brace_line(state, f"if ({var} > {cmp_val})") + f"{indent}{var} = {cmp_val};\n}}\n"
    part2 = brace_line(state, f"else") + f"{indent}{var} += {cmp_val};\n}}\n"
    return part1 + part2

@register("loop")
def gen_loop(state):
    rng = state["rng"]
    var = fresh_name(rng)
    indent = STYLE_TABLE[state["style"]]["indent"]
    return brace_line(state, f"for (int {var} = 0; {var} < {rng.randint(1,5)}; ++{var})") + f"{indent}// loop body\n}}\n"

@register("main")
def gen_main(state):
    if state["main_written"]:
        return ""
    state["main_written"] = True
    rng = state["rng"]
    indent = STYLE_TABLE[state["style"]]["indent"]
    body = []
    for _ in range(rng.randint(1, 3)):
        if state["funcs"] and rng.random() < 0.5:
            _, fname, pstr = rng.choice(list(state["funcs"]))
            args = ", ".join("0" for _ in pstr.split(",")) if pstr != "void" else ""
            body.append(f"{indent}{fname}({args});\n")
        else:
            body.append(f'{indent}printf("Hello, world!\\n");\n')
    body.append(f"{indent}return 0;\n")
    return brace_line(state, "int main(void)") + "".join(body) + "}\n"

# ──────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────

def build_c(cfg: CConfig) -> str:
    rng = random.Random(cfg.seed)
    style = rng.choice(list(STYLE_TABLE.keys())) if cfg.style == "auto" else cfg.style
    state = {
        "rng": rng,
        "style": style,
        "typedefs": set(),
        "structs": set(),
        "funcs": set(),
        "headers": set(),
        "main_written": False,
    }

    parts = ["/* Auto-generated C code */\n\n"]
    lines = parts[0].count("\n")
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        snippet = _REGISTRY[rng.choices(kinds, weights=weights)[0]](state)
        if snippet:
            parts.append(snippet)
            lines += snippet.count("\n")

    if not state["main_written"]:
        parts.append(gen_main(state))

    return "".join(parts)

# ──────────────────────────────────────────────────────────────
# CLI helpers
# ──────────────────────────────────────────────────────────────

def _compile_check(code: str) -> None:
    for compiler in ("clang", "gcc"):
        if subprocess.run(["which", compiler], capture_output=True).returncode == 0:
            cmd = [compiler, "-x", "c", "-", "-std=c17", "-Werror", "-o", "/dev/null"]
            proc = subprocess.run(cmd, input=code, text=True, capture_output=True)
            msg = "passed" if proc.returncode == 0 else f"failed:\n{proc.stderr}"
            print(f"[*] {compiler} smoke-test {msg}", file=sys.stderr)
            return
    print("[*] No C compiler found for --check", file=sys.stderr)

def _parse_weights(arg: Optional[str]) -> Dict[str, float]:
    base = CConfig().weights.copy()
    if not arg:
        return base
    for pair in arg.split(","):
        try:
            key, val = pair.split("=")
            base[key.strip()] = float(val)
        except ValueError:
            sys.exit("✖ Bad --weights syntax (use key=val[,key=val])")
    return base

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic C source file.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx. number of lines")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .c")
    p.add_argument("--style", choices=["auto", "kr", "allman", "gnu"], default="auto",
                   help="Brace/indent style")
    p.add_argument("--weights", type=str, help="Override weights: key=val[,key=val...]")
    p.add_argument("--check", action="store_true", help="Compile smoke-test via gcc/clang")
    args = p.parse_args()

    cfg = CConfig(
        loc=args.loc,
        seed=args.seed,
        style=args.style,
        check=args.check,
        weights=_parse_weights(args.weights),
    )

    code = build_c(cfg)

    if args.check:
        _compile_check(code)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated C code to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
