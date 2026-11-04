import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from time import perf_counter


def tokenize(text):
    parts = text.split("[")
    for num in range(1, len(parts)):
        tag, end, rest = parts[num].partition("]")
        if end and ":U+" in tag:
            parts[num] = rest
    text = " ".join(parts).lower().replace("ё", "е")
    words = []
    word = ""
    for char in text:
        if char.isalnum() or (word and unicodedata.category(char).startswith("M")):
            word += char
        else:
            if word:
                words.append(word)
                word = ""
    if word:
        words.append(word)
    return words


def read_corpus(path):
    blocks = []
    ids = set()
    with Path(path).open(encoding="utf-8") as stream:
        for num, line in enumerate(stream, 1):
            try:
                block = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Строка {num}: неверный JSON") from error
            
            if not isinstance(block, dict) or not {"block_id", "page", "section", "text"} <= block.keys():
                raise ValueError(f"Строка {num}: отсутствуют поля корпуса")
            key = block["block_id"]
            if not isinstance(key, str) or not key.strip() or key in ids:
                raise ValueError(f"Строка {num}: пустой или повторный block_id")
            if type(block["page"]) is not int or block["page"] < 1:
                raise ValueError(f"Строка {num}: неверный номер страницы")
            if not isinstance(block["text"], str) or not block["text"].strip():
                raise ValueError(f"Строка {num}: пустой text")
            if block["section"] is not None and not isinstance(block["section"], str):
                raise ValueError(f"Строка {num}: неверный section")
            
            ids.add(key)

            blocks.append({field: block[field] for field in ("block_id", "page", "section", "text")})

    if not blocks:
        raise ValueError("Корпус пуст")
    return blocks


def read_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    k1 = config["bm25"]["k1"]
    b = config["bm25"]["b"]

    for name, value in (("k1", k1), ("b", b)):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name}: требуется конечное число")
        
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("Требуется k1 > 0 и 0 <= b <= 1")
    if type(config["top_k"]) is not int or config["top_k"] < 1:
        raise ValueError("top_k должен быть положительным целым числом")
    if config.get("retrieval", {}).get("mode", "bm25") not in ("bm25", "hybrid"):
        raise ValueError("retrieval.mode: требуется bm25 или hybrid")
    if "retrieval" in config:
        settings = config["retrieval"]
        if settings.get("fusion", "rrf") not in ("rrf", "guarded"):
            raise ValueError("fusion: требуется rrf или guarded")
        if type(settings.get("neighbors", False)) is not bool:
            raise ValueError("neighbors должен быть boolean")
        if type(settings.get("candidates", 20)) is not int or settings.get("candidates", 20) < config["top_k"]:
            raise ValueError("candidates должен быть целым и не меньше top_k")
    if config.get("retrieval", {}).get("mode") == "hybrid":
        embedding = config["embedding"]
        revision = embedding["revision"]
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError("Зафиксируйте embedding.revision полным commit SHA")
        if type(embedding["max_length"]) is not int or not 1 <= embedding["max_length"] <= 512:
            raise ValueError("embedding.max_length должен быть от 1 до 512")
    config.setdefault("context", {"max_chars": 8000, "max_blocks": 5})

    config["context"].setdefault("max_input_tokens", 4096)
    if type(config["context"]["max_input_tokens"]) is not int or not 1 <= config["context"]["max_input_tokens"] <= 7680:
        raise ValueError("max_input_tokens должен быть от 1 до 7680")
    for name in ("max_chars", "max_blocks"):
        value = config["context"][name]
        if type(value) is not int or value < 1:
            raise ValueError(f"context.{name} должен быть положительным целым числом")
        
    return config


def build_index(path, config):
    blocks = read_corpus(path)
    lengths = []
    terms = {}

    for num, block in enumerate(blocks):
        words = tokenize(block["text"])
        lengths.append(len(words))
        for word, count in Counter(words).items():
            terms.setdefault(word, []).append([num, count])

    if not sum(lengths):
        raise ValueError("В корпусе нет слов для поиска")
    
    return {"version": 3, "tokenizer": "words-v1",
            "corpus_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "bm25": config["bm25"], "blocks": blocks, "lengths": lengths, "terms": terms}


def save_index(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "index.json"
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_index(output):
    with (Path(output) / "index.json").open(encoding="utf-8") as stream:
        data = json.load(stream)

    if data["version"] != 3 or data["tokenizer"] != "words-v1":
        raise ValueError("Несовместимый индекс: пересоберите build_index.py")
    if not data["blocks"] or len(data["blocks"]) != len(data["lengths"]) or sum(data["lengths"]) <= 0:
        raise ValueError("Неверный индекс: пересоберите build_index.py")
    return data


def check_index(data, config):
    if data["bm25"] != config["bm25"]:
        raise ValueError("Параметры поиска изменились: сначала запустите build_index.py")


def search(data, query, top_k=5):
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k должен быть положительным целым числом")
    
    size = len(data["blocks"])
    avg = sum(data["lengths"]) / size
    k1 = data["bm25"]["k1"]
    b = data["bm25"]["b"]
    scores = {}

    for word in sorted(set(tokenize(query))):
        hits = data["terms"].get(word, [])
        idf = math.log(1 + (size - len(hits) + 0.5) / (len(hits) + 0.5))

        for num, count in hits:
            length = data["lengths"][num]
            norm = k1 * (1 - b + b * length / avg)
            score = idf * count * (k1 + 1) / (count + norm)
            scores[num] = scores.get(num, 0) + score
            
    order = sorted(scores, key=lambda num: (-scores[num], num))[:top_k]
    return [dict(data["blocks"][num], score=scores[num]) for num in order]


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Сборка локального индекса BM25")
    parser.add_argument("--corpus", type=Path, default=root / "data/corpus.jsonl")
    parser.add_argument("--output", type=Path, default=root / "index")
    parser.add_argument("--config", type=Path, default=root / "config.json")
    args = parser.parse_args()
    dest = (args.output / "index.json").resolve()
    if dest in (args.corpus.resolve(), args.config.resolve()):
        raise ValueError("Нельзя записывать индекс поверх корпуса или конфигурации")
    start = perf_counter()
    config = read_config(args.config)
    data = build_index(args.corpus, config)
    path = save_index(data, args.output)
    if "tokenizer" in config:
        from tokenizer import save_tokenizer
        save_tokenizer(config["tokenizer"], args.output / "tokenizer")
    if config.get("retrieval", {}).get("mode") == "hybrid":
        from retrieval import build_dense
        build_dense(data, config, args.output)
    if load_index(args.output) != data:
        raise ValueError("Сохранённый индекс отличается от построенного")
    print(f"Блоков: {len(data['blocks'])}; словарь: {len(data['terms'])}; токенов: {sum(data['lengths'])}")
    print(f"Размер: {path.stat().st_size} байт; время: {perf_counter() - start:.3f} с")
    print(f"Корпус SHA256: {data['corpus_sha256']}")
    print(f"Индекс: {path}")


if __name__ == "__main__":
    main()
