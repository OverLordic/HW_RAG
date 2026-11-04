import argparse
import copy
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

from predict import check_answer


NAMES = ("model_only", "baseline", "candidate", "no_dense", "no_context_extension")


def read_rows(path, content=None, strict=True):
    rows = {}
    content = Path(path).read_text(encoding="utf-8") if content is None else content
    for num, line in enumerate(content.splitlines(), 1):
        try:
            row = json.loads(line)
            key = row["id"]
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Пустой id")
            if key in rows:
                if strict:
                    raise ValueError("Повторный id в результатах")
                row = {"id": key, "_error": "Повторный id"}
            rows[key] = row
        except (ValueError, KeyError, TypeError):
            if strict:
                raise
            rows["__invalid_line_" + str(num)] = {"_error": "Некорректная строка"}
    return rows


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_config(config, name):
    result = copy.deepcopy(config)
    result["use_context"] = name != "model_only"
    result["retrieval"]["mode"] = "bm25" if name in ("model_only", "baseline", "no_dense") else "hybrid"
    result["retrieval"]["neighbors"] = name not in ("model_only", "baseline", "no_neighbors", "no_context_extension")
    return result


def evaluate(gold, answers, traces, corpus, model_only=False, reviews=None):
    rows = []
    reviews = reviews or {}
    for key, expected in gold.items():
        answer = answers.get(key, {})
        trace = traces.get(key, {})
        errors = []
        counts = [trace.get(field) for field in ("input_tokens", "output_tokens", "llm_calls")]
        bad_usage = any(n is not None and (type(n) is not int or n < 0) for n in counts)
        known = all(type(n) is int and n >= 0 for n in counts)
        cost = counts[0] + 4 * counts[1] + 256 * counts[2] if known else None
        limit_error = False
        if known:
            limit_error = counts[0] > 16384 or counts[1] > 4096 or counts[2] > 4
            calls = trace.get("calls")
            if counts[2] == 1:
                generation = trace.get("generation", {})
                if not isinstance(generation, dict):
                    generation = {}
                    bad_usage = True
                calls = [dict(input_tokens=counts[0], output_tokens=counts[1],
                              max_tokens=generation.get("max_tokens", 512))]
            if counts[2] > 0:
                if not isinstance(calls, list) or len(calls) != counts[2]:
                    known = False
                else:
                    for call in calls:
                        values = [call.get(k) for k in ("input_tokens", "output_tokens", "max_tokens")]
                        if not all(type(n) is int and n >= 0 for n in values):
                            known = False
                            bad_usage = True
                            break
                        limit_error |= values[0] + values[2] > 8192 or values[1] > values[2]
                    if known:
                        known = (sum(c["input_tokens"] for c in calls) == counts[0]
                                 and sum(c["output_tokens"] for c in calls) == counts[1])
        if limit_error:
            errors.append("Превышение вычислительного лимита")
        if not known:
            errors.append("Непроверенный расход")
            cost = None
        valid = trace.get("valid") is True and not limit_error and not bad_usage
        try:
            if answer.get("id") != key:
                raise ValueError("Потерян ID ответа")
            ids = trace.get("context_block_ids", [])
            if not set(ids) <= corpus:
                raise ValueError("Неизвестный источник в trace")
            check_answer({k: v for k, v in answer.items() if k != "id"},
                         {"id": key, "options": expected["options"]}, ids, not model_only)
        except (ValueError, KeyError, TypeError) as error:
            valid = False
            errors.append(str(error))
        if known and counts[2] == 0 and trace.get("valid") is True:
            valid = False
            errors.append("Ответ без вызова модели")
        exact = (valid and answer["abstain"] == expected["abstain"]
                 and set(answer["answer_ids"]) == set(expected["answer_ids"]))
        score = float(exact)
        if exact and not expected["abstain"]:
            found = set(answer["citations"])
            f1 = max((2 * len(found & set(ids)) / (len(found) + len(ids))
                      for ids in expected["citation_sets"]), default=0)
            score *= 0.8 + 0.2 * f1
        if reviews.get(key, {}).get("reject", False):
            score = 0
        rows.append({"id": key, "category": expected["category"], "exact": bool(exact),
                     "score": score, "cost": cost, "valid": valid, "errors": errors,
                     "efficiency_eligible": known and not limit_error,
                     "review": reviews.get(key), "answer": answer})
    if not rows:
        raise ValueError("Эталоны пусты")
    categories = sorted({row["category"] for row in rows})
    quality = statistics.mean(statistics.mean(row["score"] for row in rows if row["category"] == category)
                              for category in categories)

    def mean(field):
        values = [traces.get(key, {}).get(field) for key in gold]
        return statistics.mean(values) if all(type(n) is int and n >= 0 for n in values) else ""

    latency = [traces.get(key, {}).get("elapsed_ms") for key in gold]
    metrics = {"answer_accuracy": sum(row["exact"] for row in rows) / len(rows),
               "Q": "" if model_only else quality, "mean_input_tokens": mean("input_tokens"),
               "mean_output_tokens": mean("output_tokens"), "mean_llm_calls": mean("llm_calls"),
               "mean_cost": statistics.mean(row["cost"] for row in rows) if all(row["cost"] is not None for row in rows) else "",
               "median_latency_ms": statistics.median(latency) if all(type(n) is int and n >= 0 for n in latency) else ""}
    return metrics, rows


def prepare_run(root, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Каталог результатов не пуст: задайте новый --output")
    output.mkdir(parents=True, exist_ok=True)
    files = list(root.glob("*.py")) + list((root / "tests").glob("*.py"))
    files += [root / p for p in ("config.json", "config_hybrid.json", "runtime.json", "requirements.txt",
                                "dev_questions.jsonl", "dev_gold.jsonl", "data/corpus.jsonl")]
    hashes = {}
    with zipfile.ZipFile(output / "source_bundle.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            name = str(file.relative_to(root))
            hashes[name] = file_hash(file)
            archive.write(file, name)
    manifest = {"run_id": uuid.uuid4().hex[:12], "status": "prepared", "names": list(NAMES),
                "run_source_sha256": hashes, "source_bundle_sha256": file_hash(output / "source_bundle.zip"), "variants": {}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def summarize(root, output):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(output / "source_bundle.zip") as archive:
        gold_text = archive.read("dev_gold.jsonl").decode("utf-8")
        corpus_text = archive.read("data/corpus.jsonl").decode("utf-8")
    gold = read_rows(None, gold_text)
    if len({r["category"] for r in gold.values()}) != 4:
        raise ValueError("Ожидаются четыре категории открытого набора")
    corpus = {json.loads(line)["block_id"] for line in corpus_text.splitlines()}
    names = manifest.get("names", ("model_only", "baseline", "hybrid", "no_dense", "no_neighbors"))
    review_path = output / "rationale_review.json"
    reviews = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    basis = {"gold": hashlib.sha256(gold_text.encode()).hexdigest(),
             "answers": {name: file_hash(output / name / "answers.jsonl") for name in names
                         if (output / name / "answers.jsonl").exists()}}
    if reviews:
        if "_hashes" in reviews:
            if reviews["_hashes"] != basis:
                raise ValueError("Ручная оценка относится к другим ответам или эталонам")
        else:
            # Старый неизменяемый прогон имел хэши review и ответов в manifest.
            expected = manifest.get("result_sha256", {})
            files = [review_path] + [output / name / "answers.jsonl" for name in names]
            if (manifest.get("names") is not None or basis["gold"] != manifest["run_source_sha256"]["dev_gold.jsonl"]
                    or any(expected.get(str(p.relative_to(output))) != file_hash(p) for p in files)):
                raise ValueError("Добавьте _hashes из manifest.review_basis в новую ручную оценку")
    metrics, details, statuses = [], {}, {}
    for name in names:
        path = output / name
        error = None
        try:
            answers = read_rows(path / "answers.jsonl", strict=False)
            traces = read_rows(path / "trace.jsonl", strict=False)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            answers, traces = {}, {}
            error = str(exc)
        values, rows = evaluate(gold, answers, traces, corpus, name == "model_only", reviews.get(name))
        ids_ok = set(answers) == set(gold) == set(traces)
        metrics.append(dict(name=name, **values))
        details[name] = rows
        statuses[name] = {"complete": ids_ok and all(row["valid"] for row in rows),
                          "efficiency_eligible": ids_ok and all(row["efficiency_eligible"] for row in rows),
                          "error": error}
    with (output / "experiments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (output / "scores.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest.update(review_basis=basis, evaluation=statuses)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return all(s["complete"] and s["efficiency_eligible"] for s in statuses.values())


def run_all(root, output, index, env=None):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, sha in manifest["run_source_sha256"].items():
        if file_hash(root / name) != sha:
            raise ValueError("Исходники изменены после подготовки: " + name)
    config = json.loads((root / "config_hybrid.json").read_text(encoding="utf-8"))
    failed = False
    manifest["status"] = "running"
    try:
        for name in NAMES:
            dest = output / name
            dest.mkdir()
            (dest / "config.json").write_text(json.dumps(make_config(config, name), ensure_ascii=False, indent=2), encoding="utf-8")
            settings = dict(os.environ if env is None else env,
                            SUBMISSION_ID=manifest["run_id"] + "-" + name,
                            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            result = subprocess.run([sys.executable, str(root / "predict.py"), "--config", str(dest / "config.json"),
                                     "--questions", str(root / "dev_questions.jsonl"), "--index", str(index),
                                     "--output", str(dest / "answers.jsonl"), "--trace", str(dest / "trace.jsonl")], env=settings)
            manifest["variants"][name] = {"exit_code": result.returncode}
            failed |= result.returncode != 0
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print("EXPERIMENT", name, "exit_code", result.returncode, flush=True)
    except Exception as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
        failed = True
    finally:
        failed |= len(manifest["variants"]) != len(NAMES)
        manifest["status"] = "failed" if failed else "completed"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        failed |= not summarize(root, output)
    except Exception as error:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["summary_error"] = f"{type(error).__name__}: {error}"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        failed = True
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "failed" if failed else "completed"
    manifest["result_sha256"] = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*"))
                                 if p.is_file() and p != manifest_path}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return int(failed)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--output", type=Path, default=root / "outputs/experiments_new")
    parser.add_argument("--index", type=Path, default=root / "index")
    args = parser.parse_args()
    if args.summarize:
        code = int(not summarize(root, args.output))
    else:
        prepare_run(root, args.output)
        code = run_all(root, args.output, args.index)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
