import hashlib
import json
from pathlib import Path


def save_tokenizer(settings, path):
    from transformers import AutoTokenizer

    path = Path(path)
    tokenizer = AutoTokenizer.from_pretrained(
        settings["model"], revision=settings["revision"], trust_remote_code=False
    )
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(path)
    hashes = {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
              for file in sorted(path.iterdir()) if file.is_file() and file.name != "manifest.json"}
    (path / "manifest.json").write_text(
        json.dumps(dict(settings, files=hashes), indent=2) + "\n", encoding="utf-8"
    )


def load_tokenizer(settings, path):
    from transformers import AutoTokenizer

    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if any(manifest[key] != settings[key] for key in ("model", "revision")):
        raise ValueError("Токенизатор не соответствует config: пересоберите индекс")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Изменён файл токенизатора: {name}")
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def count_tokens(tokenizer, messages):
    data = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=True
    )
    return len(data["input_ids"])
