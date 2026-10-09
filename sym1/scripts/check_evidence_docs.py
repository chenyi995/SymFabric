#!/usr/bin/env python3
"""Read-only structural/source audit for the bilingual evidence handbook.

Sources: handbook Markdown, current workload ASTs, and upstream_lock.json.
Checks paths, source-line/function links, pinned upstream URLs, evidence slots,
mathematical-block parity and example parity. It does not verify a paper's
theorems or performance. Numerical teaching checks live in documentation_examples.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def check():
    # claude: decision start — lock-based link validation is read-only; a clone
    # or syntactically valid citation is never recorded as a completed baseline.
    lock = json.loads((ROOT / "sym1/upstream_lock.json").read_text())
    repo_prefixes = {
        r["url"].removesuffix(".git") + "/blob/" + r["commit"] + "/": r
        for r in lock["repositories"]
    }
    workloads = {p.parent.name: p for p in (ROOT / "softwareworkload").glob("*/workload.py")}
    paths = [ROOT / "sym1/workload_operator_evidence.md", ROOT / "local/sym1-workload-operator-evidence.zh.md"]
    texts = [p.read_text() for p in paths]
    heading_groups = [
        ["Workload literature and mathematical task", "Native key code and work", "Workload example",
         "Decomposition literature and justification", "Key lowering code and work",
         "Decomposition example: derived intermediate values"],
        ["Workload 文献与数学任务", "原生关键代码及其工作", "Workload 例子", "算子展开文献与理由",
         "展开关键代码及其工作", "算子展开例子：推导出的中间值"],
    ]
    for path, text, headings in zip(paths, texts, heading_groups):
        ids = re.findall(r'<a id="([^"]+)"></a>', text)
        assert len(ids) == len(set(ids)), (path, "duplicate anchor")
        blocks = re.findall(r"<!-- EXAMPLE:([^ ]+) -->\n(.*?)\n<!-- /EXAMPLE:\1 -->", text, flags=re.S)
        assert {name for name, _ in blocks} == set(workloads), (path, "inventory mismatch")
        assert len(blocks) == len(workloads), (path, "duplicate example")
        for heading in headings:
            assert text.count("#### " + heading + "\n") == len(workloads), (path, heading)
        assert text.count("\n$$\n") % 2 == 0, (path, "display math delimiter")
        assert text.count("\n```\n") == len(re.findall(r"\n```(?:text|python|bash)\n", text)), (path, "code fence")
        for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text):
            if target.startswith("#"):
                assert target[1:] in ids, (path, target)
            elif not target.startswith(("https://", "http://")):
                rel, _, fragment = target.partition("#")
                source = (path.parent / rel).resolve()
                assert source.exists(), (path, target)
                if fragment.startswith("L"):
                    line = int(fragment[1:])
                    content = source.read_text().splitlines()
                    assert 0 < line <= len(content), (target, "line out of range")
                    if source.name == "workload.py":
                        fn = label.removeprefix("Local ").removeprefix("本地 ").split(" / ")[0]
                        tree = ast.parse(source.read_text())
                        actual = {node.name: node.lineno for node in tree.body if isinstance(node, ast.FunctionDef)}
                        assert actual[fn] == line, (path, target, fn)
            elif "github.com" in target and "/blob/" in target:
                matching = [(prefix, r) for prefix, r in repo_prefixes.items() if target.startswith(prefix)]
                assert len(matching) == 1, (target, "unpinned or wrong revision")
                prefix, repo = matching[0]
                rel = target[len(prefix):].split("#")[0]
                assert (ROOT / repo["path"] / rel).is_file(), (target, "missing local upstream file")
    math = [re.findall(r"\n\$\$\n(.*?)\n\$\$\n", t, flags=re.S) for t in texts]
    assert math[0] == math[1], "bilingual equation mismatch"
    examples = [re.findall(r"<!-- EXAMPLE:([^ ]+) -->\n(.*?)\n<!-- /EXAMPLE:\1 -->", t, flags=re.S) for t in texts]
    assert examples[0] == examples[1], "bilingual example mismatch"
    urls = [{u for u in re.findall(r"\]\((https://[^)]+)\)", t)} for t in texts]
    assert urls[0] == urls[1], "bilingual external reference mismatch"
    print(f"PASS: {len(workloads)} workloads, six evidence slots each; local function lines, pinned upstream paths, math and bilingual parity.")
    # claude: decision end


if __name__ == "__main__":
    check()
