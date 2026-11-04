import json
from pathlib import Path

from build_index import search


def make_query(question):
    return question["question"] + "\n" + "\n".join(option["text"] for option in question["options"])


def format_block(block):
    text = f"[Источник: {block['block_id']}; страница PDF: {block['page']}]\n"
    if block["section"] is not None:
        text += f"Раздел: {block['section']}\n"
    return text + "Текст:\n" + block["text"]


def make_context(blocks, max_chars=8000, max_blocks=5, max_input_tokens=4096):
    for name, value in (("max_chars", max_chars), ("max_blocks", max_blocks)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} должен быть положительным целым числом")
    parts = []
    ids = []
    pages = []
    seen = {}
    texts = set()
    dropped = []
    size = 0
    for block in blocks:
        key = block["block_id"]
        text = block["text"]
        if not isinstance(key, str) or not key.strip() or not isinstance(text, str) or not text.strip():
            raise ValueError("У источника должны быть непустые block_id и text")
        if type(block["page"]) is not int or block["page"] < 1:
            raise ValueError("У источника должен быть положительный номер страницы PDF")
        if key in seen:
            if seen[key] != (text, block["page"], block["section"]):
                raise ValueError(f"Один block_id обозначает разные источники: {key}")
            dropped.append({"block_id": key, "reason": "duplicate_id"})
            continue
        seen[key] = (text, block["page"], block["section"])
        if text in texts:
            dropped.append({"block_id": key, "reason": "duplicate_text"})
            continue
        if block.get("via") == "continuation" and block.get("parent") not in ids:
            dropped.append({"block_id": key, "reason": "missing_parent"})
            continue
        if len(ids) >= max_blocks:
            dropped.append({"block_id": key, "reason": "block_limit"})
            continue
        part = format_block(block)
        extra = len(part) + (2 if parts else 0)
        if size + extra > max_chars:
            dropped.append({"block_id": key, "reason": "size_limit"})
            continue
        parts.append(part)
        ids.append(key)
        pages.append(block["page"])
        texts.add(text)
        size += extra
    return {"text": "\n\n".join(parts), "block_ids": ids, "pages": pages,
            "chars": size, "max_chars": max_chars, "max_blocks": max_blocks,
            "max_input_tokens": max_input_tokens,
            "dropped": dropped}


def load_encoder(settings, path=None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(4)
    source = path if path is not None else settings["model"]
    options = {"local_files_only": True} if path is not None else {"revision": settings["revision"]}
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **options)
    model = AutoModel.from_pretrained(source, trust_remote_code=False, use_safetensors=True, **options).to("cpu").eval()
    if sum(param.numel() for param in model.parameters()) > 1_000_000_000:
        raise ValueError("Embedding-модель превышает лимит 1 млрд параметров")
    return {"tokenizer": tokenizer, "model": model}


def encode(encoder, texts, prefix, max_length):
    import torch

    vectors = []
    for start in range(0, len(texts), 8):
        inputs = encoder["tokenizer"]([prefix + text for text in texts[start:start + 8]],
                                      padding=True, truncation=True, max_length=max_length,
                                      return_tensors="pt")
        with torch.inference_mode():
            states = encoder["model"](**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            mean = (states * mask).sum(1) / mask.sum(1)
            vectors.extend(torch.nn.functional.normalize(mean, dim=1).tolist())
    return vectors


def build_dense(data, config, output):
    settings = config["embedding"]
    encoder = load_encoder(settings)
    path = Path(output)
    model_path = path / "encoder"
    model_path.mkdir(parents=True, exist_ok=True)
    encoder["tokenizer"].save_pretrained(model_path)
    encoder["model"].save_pretrained(model_path)
    texts = [(block["section"] or "") + "\n" + block["text"] for block in data["blocks"]]
    vectors = encode(encoder, texts, "passage: ", settings["max_length"])
    result = {"embedding": settings, "corpus_sha256": data["corpus_sha256"],
              "block_ids": [block["block_id"] for block in data["blocks"]], "vectors": vectors}
    (path / "dense.json").write_text(json.dumps(result) + "\n", encoding="utf-8")
    print(f"E5: {len(vectors)} векторов, размерность {len(vectors[0])}; CPU")


def open_search(data, config, path):
    if config.get("retrieval", {}).get("mode", "bm25") == "bm25":
        return None
    import torch

    dense = json.loads((Path(path) / "dense.json").read_text(encoding="utf-8"))
    if (dense["embedding"] != config["embedding"] or dense["corpus_sha256"] != data["corpus_sha256"]
            or dense["block_ids"] != [block["block_id"] for block in data["blocks"]]):
        raise ValueError("Семантический индекс устарел: пересоберите build_index.py")
    vectors = torch.tensor(dense["vectors"], dtype=torch.float32)
    if vectors.shape != (len(data["blocks"]), 384) or not torch.isfinite(vectors).all():
        raise ValueError("Неверные векторы E5")
    return {"encoder": load_encoder(config["embedding"], Path(path) / "encoder"), "vectors": vectors}


def add_neighbors(data, blocks, extra=()):
    from extract_corpus import LABELS

    positions = {block["block_id"]: num for num, block in enumerate(data["blocks"])}
    result = [dict(block, via="search") for block in blocks]
    seen = {block["block_id"] for block in blocks}
    for block in list(blocks) + list(extra):
        if len(result) >= len(blocks) + 2:
            break
        pos = positions[block["block_id"]]
        if pos + 1 >= len(data["blocks"]) or not block["text"].startswith(("Теорема ", "Следствие ", "Лемма ")):
            continue
        other = data["blocks"][pos + 1]
        text = other["text"].lstrip()
        if (other["block_id"] in seen or other["page"] != block["page"] + 1
                or other["section"] != block["section"] or other["section"] is None
                or text.startswith(LABELS) or not text or not text[0].islower()):
            continue
        if block["block_id"] not in seen:
            if len(result) + 2 > len(blocks) + 2:
                continue
            result.append(dict(block, via="continuation_start"))
            seen.add(block["block_id"])
        result.append(dict(other, score=block["score"], via="continuation", parent=block["block_id"]))
        seen.add(other["block_id"])
    return result


def find_blocks(data, query, config, engine=None):
    if not query.strip():
        return []
    settings = config.get("retrieval", {})
    if settings.get("mode", "bm25") == "bm25":
        blocks = search(data, query, config["top_k"])
    else:
        if engine is None:
            raise ValueError("Сначала загрузите семантический индекс")
        count = settings["candidates"]
        lexical = search(data, query, count)
        vector = encode(engine["encoder"], [query], "query: ", config["embedding"]["max_length"])[0]
        import torch

        scores = (engine["vectors"] @ torch.tensor(vector)).tolist()
        order = sorted(range(len(scores)), key=lambda num: (-scores[num], num))[:count]
        semantic = [data["blocks"][num] for num in order]
        ranks = {}
        found = {}
        for group in (lexical, semantic):
            for rank, block in enumerate(group, 1):
                key = block["block_id"]
                ranks[key] = ranks.get(key, 0) + 1 / (60 + rank)
                found[key] = block
        keys = sorted(ranks, key=lambda key: (-ranks[key], key))[:config["top_k"]]
        if settings.get("fusion", "rrf") == "guarded":
            anchors = [b["block_id"] for b in lexical[:min(3, config["top_k"])]]
            keys = (anchors + [key for key in keys if key not in anchors])[:config["top_k"]]
        blocks = [dict(found[key], score=ranks[key]) for key in keys]
    if settings.get("neighbors", False):
        extra = lexical[:config["top_k"]] if settings.get("mode") == "hybrid" else []
        return add_neighbors(data, blocks, extra)
    return blocks
