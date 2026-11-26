"""Порівняння версій системного промпту TestGen (ітеративне вдосконалення v1 -> v2 -> v3).

v1 — наївний промпт без ролі й формату;
v2 — роль + Few-Shot приклад стилю, без самоперевірки;
v3 — CoT-аналіз вимог + Few-Shot + агентний цикл з run_pytest (фінальна версія).
Для v1/v2 код тестів витягується з відповіді й запускається тим самим pytest-пісочницею.
"""
import os
import re
import sys

from testgen import MODEL, PytestSandbox, client, generate_tests, load_prompt


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.S)
    return max(blocks, key=len) if blocks else text


def one_shot(module_path: str, version: str) -> dict:
    sandbox = PytestSandbox(module_path)
    source = open(module_path, encoding="utf-8").read()
    prompt = (load_prompt(version, sandbox.module_name) +
              f"\n\nМодуль `{sandbox.module_name}.py`:\n```python\n{source}\n```")
    text = client.models.generate_content(model=MODEL, contents=prompt).text
    res = sandbox.run_pytest(extract_code(text))
    return {"counts": res["counts"], "coverage": res["coverage_percent"], "exit": res["exit_code"],
            "bugs": "—", "error": "ImportError" in res["output"] or "SyntaxError" in res["output"]}


def main(module_path: str):
    rows = []
    for v in ("v1", "v2"):
        print(f"Промпт {v}...")
        rows.append((v, one_shot(module_path, v)))
    print("Промпт v3 (агент)...")
    r3 = generate_tests(module_path, "v3")
    last = r3["runs"][-1] if r3["runs"] else {"counts": {}, "coverage_percent": None, "exit_code": None}
    rows.append(("v3", {"counts": last["counts"], "coverage": last["coverage_percent"], "exit": last["exit_code"],
                        "bugs": len(r3.get("bugs", [])), "error": False}))
    print(f"\nМодуль: {module_path}")
    print(f"{'Версія':8}{'passed':>8}{'failed':>8}{'xfailed':>9}{'error':>7}{'покриття':>10}{'багів':>7}")
    for v, r in rows:
        c = r["counts"]
        print(f"{v:8}{c.get('passed', 0):>8}{c.get('failed', 0):>8}{c.get('xfailed', 0):>9}"
              f"{c.get('error', 0):>7}{str(r['coverage']) + '%':>10}{str(r['bugs']):>7}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "examples", "grades.py"))
