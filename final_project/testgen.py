"""Підсумковий проєкт: TestGen — генератор unit-тестів pytest на базі Gemini API.

Архітектура: агент (Function Calling) + промпт-інженерія (CoT-аналіз вимог, Few-Shot стиль тестів).
Інструмент run_pytest запускає згенеровані тести в ізольованій тимчасовій теці й повертає результат,
тож модель сама перевіряє свої тести, виправляє помилки в них і відрізняє їх від багів у коді.

Запуск: python final_project/testgen.py <шлях до модуля .py> [--out tests_dir] [--prompt v3]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from google.genai import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gemini_setup import MODEL, get_client  # noqa: E402

client = get_client()
MAX_RUNS = 3


class PytestSandbox:
    """Запускає pytest для одного модуля в окремій тимчасовій теці з таймаутом."""

    def __init__(self, module_path: str):
        self.module_path = module_path
        self.module_name = os.path.splitext(os.path.basename(module_path))[0]
        self.runs = []

    def run_pytest(self, test_code: str) -> dict:
        """Runs the given pytest test file against the student's module and returns the results."""
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(self.module_path, tmp)
            test_path = os.path.join(tmp, f"test_{self.module_name}.py")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write(test_code)
            cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider",
                   f"--cov={self.module_name}", "--cov-report=term-missing", test_path]
            try:
                proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=60,
                                      env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                out = (proc.stdout + proc.stderr)[-3500:]
            except subprocess.TimeoutExpired:
                return {"status": "timeout", "output": "pytest перевищив ліміт 60 с"}
        summary = re.findall(r"(\d+) (passed|failed|xfailed|xpassed|error)", out)
        counts = {k: int(v) for v, k in summary}
        cov = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", out) or re.search(rf"{self.module_name}\.py\s+\d+\s+\d+\s+(\d+)%", out)
        result = {"exit_code": proc.returncode, "counts": counts,
                  "coverage_percent": int(cov.group(1)) if cov else None, "output": out}
        self.runs.append(result)
        return result


def load_prompt(version: str, module_name: str) -> str:
    with open(os.path.join(HERE, "prompts", f"system_{version}.txt"), encoding="utf-8") as f:
        return f.read().replace("{module}", module_name)


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def generate_tests(module_path: str, prompt_version: str = "v3", verbose: bool = True) -> dict:
    sandbox = PytestSandbox(module_path)
    with open(module_path, encoding="utf-8") as f:
        source = f.read()
    config = types.GenerateContentConfig(
        system_instruction=load_prompt(prompt_version, sandbox.module_name),
        tools=[sandbox.run_pytest],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    chat = client.chats.create(model=MODEL, config=config)
    user_msg = (f"Модуль `{sandbox.module_name}.py` (імпортуй як `{sandbox.module_name}`):\n"
                f"<module_code>\n{source}\n</module_code>")
    t0 = time.time()
    response = chat.send_message(user_msg)
    while response.function_calls and len(sandbox.runs) < MAX_RUNS:
        parts = []
        for fc in response.function_calls:
            res = sandbox.run_pytest(**dict(fc.args))
            if verbose:
                print(f"  [run_pytest #{len(sandbox.runs)}] exit={res['exit_code']} counts={res['counts']} "
                      f"coverage={res['coverage_percent']}%")
            parts.append(types.Part.from_function_response(
                name=fc.name, response={k: v for k, v in res.items()}))
        response = chat.send_message(parts)
    if response.function_calls:  # ліміт запусків вичерпано — просимо фінальну відповідь
        response = chat.send_message("Ліміт запусків вичерпано. Поверни фінальний JSON.")
    elapsed = time.time() - t0
    try:
        result = extract_json(response.text)
    except (ValueError, json.JSONDecodeError):
        result = {"test_file": "", "bugs": [], "summary": response.text.strip()[:500], "parse_error": True}
    result["runs"] = [{k: r[k] for k in ("exit_code", "counts", "coverage_percent")} for r in sandbox.runs]
    result["elapsed_s"] = round(elapsed, 1)
    return result


def main():
    ap = argparse.ArgumentParser(description="TestGen: генерація pytest-тестів за допомогою Gemini")
    ap.add_argument("module")
    ap.add_argument("--out", default=os.path.join(HERE, "generated_tests"))
    ap.add_argument("--prompt", default="v3")
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.module))[0]
    print(f"TestGen | модель {MODEL} | промпт {args.prompt} | модуль {args.module}")
    result = generate_tests(args.module, args.prompt)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"test_{name}.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result.get("test_file", ""))
    with open(os.path.join(args.out, f"report_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    last = result["runs"][-1] if result["runs"] else {}
    print("\n=== Підсумок ===")
    print(f"Запусків pytest: {len(result['runs'])}; останній: {last.get('counts')}; "
          f"покриття: {last.get('coverage_percent')}%; час: {result['elapsed_s']} с")
    print(f"Тести збережено: {out_path}")
    print("Знайдені баги:" if result.get("bugs") else "Багів у коді не виявлено.")
    for b in result.get("bugs", []):
        print(f"  - {b.get('function')}: {b.get('description')} (вхід {b.get('failing_input')}, "
              f"очікувано {b.get('expected')}, отримано {b.get('actual')})")
    print("Висновок моделі:", result.get("summary"))
    print("\n--- Згенерований файл тестів (початок) ---")
    print("\n".join(result.get("test_file", "").splitlines()[:40]))


if __name__ == "__main__":
    main()
