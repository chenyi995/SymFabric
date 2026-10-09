#!/usr/bin/env python3
"""Check or populate the general README's evidence blocks, without benchmarks.

Sources: workload AST CATEGORY/NAME fields; the historical large/seed-0
workloads.json; exact teaching fixtures in documentation_examples.py.
Accounting: ratio_mul = algebra_mul_ops/native_ops;
ratio_cand = algebra_candidates/native_ops. No timing/model values are copied.
Default is read-only; --write updates only declared generated Markdown blocks.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "eval/results/20260925-203019-final6-large/workloads.json"
DOCUMENTS = (ROOT / "README.md", ROOT / "local/general-readme.zh.md")


# codex: decision start — mechanically transfer historical counts and checked
# teaching fixtures; preserve source accounting and verify the written blocks.
def source_blocks():
    categories = defaultdict(list)
    for source in sorted((ROOT / "softwareworkload").glob("*/workload.py")):
        fields = {}
        for node in ast.parse(source.read_text()).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"NAME", "CATEGORY"}:
                        fields[target.id] = ast.literal_eval(node.value)
        assert fields["NAME"] == source.parent.name, source
        categories[fields["CATEGORY"]].append(fields["NAME"])
    inventory = ["| CATEGORY | Modules | Count |", "|---|---|---|"]
    for category, names in sorted(categories.items()):
        inventory.append(f"| {category} | {', '.join(f'`{n}`' for n in names)} | {len(names)} |")
    names = {name for group in categories.values() for name in group}
    inventory.append(f"\nSource inventory: {len(names)} modules, {len(categories)} categories.")

    raw = SOURCE.read_bytes()
    rows = json.loads(raw)
    assert {r["name"] for r in rows} == names
    assert len(rows) == len(names)
    table = ["| Workload | Golden agrees | Native events | ratio_mul | ratio_cand |",
             "|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda r: r["name"]):
        assert row["scale"] == "large" and row["seed"] == 0, row["name"]
        native = row["native_ops"]
        assert native > 0
        mul = row["algebra_mul_ops"] / native
        candidates = row["algebra_candidates"] / native
        assert mul == row["ratio_mul"] and candidates == row["ratio_cand"], row["name"]
        table.append(f"| `{row['name']}` | {'yes' if row['ok'] else 'NO'} | "
                     f"{native:,} | {mul:.3f} | {candidates:.3f} |")
    table.append(f"\nSource SHA-256: `{hashlib.sha256(raw).hexdigest()}`.")

    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "readme_teaching_examples", ROOT / "sym1/scripts/documentation_examples.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    examples = {fn.__name__: f"```text\n{fn()}\n```" for fn in module.EXAMPLES}
    assert set(examples) == names
    blocks = {"INVENTORY": "\n".join(inventory), "WORK_TABLE": "\n".join(table)}
    blocks.update({f"CASE:{name}": value for name, value in examples.items()})
    return blocks, len(names)


def check_document(path, text, blocks):
    for name, value in blocks.items():
        expected = f"<!-- {name} -->\n{value}\n<!-- /{name} -->"
        assert text.count(expected) == 1, (path, name, "missing, duplicated or stale block")
    assert len(re.findall(r"^```", text, flags=re.M)) % 2 == 0, (path, "code fences")
    ids = re.findall(r'<a id="([^"]+)"></a>', text)
    assert len(ids) == len(set(ids)), (path, "duplicate anchor")
    for _, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text):
        if target.startswith(("https://", "http://")):
            continue
        rel, _, anchor = target.partition("#")
        if not rel:
            assert anchor in ids, (path, target)
        else:
            assert (path.parent / rel).exists(), (path, target, "missing local path")
    if path == DOCUMENTS[0]:
        assert not re.search(r"[\u4e00-\u9fff]", text), "main README must be English"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    blocks, count = source_blocks()
    for path in DOCUMENTS:
        text = path.read_text()
        if args.write:
            for name, value in blocks.items():
                pattern = re.escape(f"<!-- {name} -->") + r".*?" + re.escape(f"<!-- /{name} -->")
                assert len(re.findall(pattern, text, flags=re.S)) == 1, (path, name)
                replacement = f"<!-- {name} -->\n{value}\n<!-- /{name} -->"
                text = re.sub(pattern, lambda _: replacement, text, flags=re.S)
            path.write_text(text)
        check_document(path, path.read_text(), blocks)
    print(f"PASS: {count} workload inventories, historical count ratios, teaching traces, "
          "bilingual generated-block parity, local paths and README language.")


if __name__ == "__main__":
    main()
# codex: decision end
