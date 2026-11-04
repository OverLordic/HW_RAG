import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median
import pymupdf


LABELS = ("Теорема ", "Определение ", "Следствие ", "Замечание ",
          "Пример ", "Примеры ", "Лемма ", "Предложение ", "Доказательство.")
TEXT_SIGNS = {"\x15": "–", "\x16": "—", "\x9f": "§", "¼": "ё", "½": "„"}
GLYPHS = {"summationdisplay": "∑", "summationtext": "∑",
          "integraldisplay": "∫", "integraltext": "∫",
          "productdisplay": "∏", "producttext": "∏", "uniondisplay": "∪"}


def is_cyr(char):
    return "а" <= char.lower() <= "я" or char.lower() == "ё"


def read_fonts(doc):
    types = {}
    maps = {}
    seen = set()
    for page in doc:
        for font in page.get_fonts():
            xref = font[0]
            kind = font[2]
            name = font[3]
            name = name.split("+")[-1]
            types[name] = kind
            if xref in seen or name != "CMEX10":
                continue
            seen.add(xref)
            data = doc.extract_font(xref)[3].split(b"currentfile eexec")[0]
            for line in data.decode("latin1").splitlines():
                parts = line.split()
                if len(parts) == 4 and parts[0] == "dup" and parts[1].isdigit():
                    glyph = parts[2].lstrip("/")
                    if glyph in GLYPHS:
                        maps[(name, chr(int(parts[1])))] = GLYPHS[glyph]
                    elif glyph.startswith(("parenleftbig", "parenleftBig")):
                        maps[(name, chr(int(parts[1])))] = "("
                    elif glyph.startswith(("parenrightbig", "parenrightBig")):
                        maps[(name, chr(int(parts[1])))] = ")"

    counts = Counter()
    for page in doc:
        for block in page.get_text("dict", sort=False)["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    font = span["font"]
                    if types.get(font) == "Type3":
                        words = [word.strip(".,:;()!?") for word in span["text"].split()]
                        counts[font] += sum(
                            len(word) >= 5 and all(192 <= ord(c) <= 255 for c in word)
                            for word in words
                        )
    return {font for font, count in counts.items() if count >= 1}, maps


def fix_text(text, font, cyr, maps):
    chars = []
    for char in text:
        if font in cyr:
            if 192 <= ord(char) <= 255:
                char = char.encode("latin1").decode("cp1251")
            else:
                char = TEXT_SIGNS.get(char, char)
        else:
            if font == "CMEX10" and not char.isspace() and (font, char) not in maps:
                char = f"[{font}:U+{ord(char):04X}]"
            else:
                char = maps.get((font, char), char)
        if len(char) == 1 and (ord(char) < 32 or 127 <= ord(char) < 160 or 0xE000 <= ord(char) <= 0xF8FF):
            char = f"[{font}:U+{ord(char):04X}]"
        chars.append(char)
    return "".join(chars)


def read_bars(page):
    bars = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                a, b = item[1:]
                if abs(a.y - b.y) < 0.1:
                    bars.append((min(a.x, b.x), max(a.x, b.x), a.y))
    return bars


def has_bar(char, size, bars):
    x, y = char["origin"]
    width = char["bbox"][2] - char["bbox"][0]
    return any(abs(left - x) < 0.25 and abs(right - x - width) < 0.7
               and 0.52 * size < y - top < 0.62 * size
               for left, right, top in bars)


def read_line(line, cyr, maps, bars):
    parts = []
    prose = False
    bold = False
    for span in line["spans"]:
        font = span["font"]
        prose = prose or font in cyr
        bold = bold or font == "F49"
        text = ""
        for char in span["chars"]:
            value = fix_text(char["c"], font, cyr, maps)
            if char["c"] == "z" and font.startswith("CMMI"):
                if has_bar(char, span["size"], bars):
                    value += "\u0305"
            text += value
        parts.append(text)
    return "".join(parts).strip(), prose, bold


def join_lines(lines):
    text = ""
    prev_prose = False
    for line, prose in lines:
        if text and prev_prose and prose:
            if len(text) > 1 and text.endswith("-") and is_cyr(text[-2]) and is_cyr(line[0]):
                text = text[:-1] + line
            else:
                text += " " + line
        else:
            text += ("\n" if text else "") + line
        prev_prose = prose
    return text


def read_page(page, cyr, maps):
    bars = read_bars(page)
    units = []
    for block in page.get_text("rawdict", sort=False)["blocks"]:
        lines = []
        start = False
        heading = None
        for line in block.get("lines", []):
            # Колонтитулы этого учебника находятся выше y=40.
            if line["bbox"][3] < 40:
                continue
            text, prose, bold = read_line(line, cyr, maps, bars)
            if not text:
                continue
            label = text.startswith(LABELS) and not text.startswith(("Теорема доказана", "Лемма доказана"))
            title = bold or text in ("Содержание", "Список литературы")
            boundary = label or (title and not (heading and lines))
            if boundary and lines:
                units.append((join_lines(lines), start, heading))
                lines = []
            if not lines:
                start = boundary
                heading = text if title else None
            lines.append((text, prose))
        if lines:
            units.append((join_lines(lines), start, heading))
    return units


def get_section(heading, page, toc):
    if not heading:
        return None
    text = heading.removeprefix("§").strip()
    number = text.split()[0].rstrip(".")
    for item in toc:
        title = item[1]
        num = item[2]
        if num == page and (title.split()[0] == number or title.strip() == text):
            return title.strip()
    return None


def split_page(units, page, section, toc, target):
    chunks = []
    parts = []
    size = 0
    pending_heading = False
    for text, start, heading in units:
        new_section = get_section(heading, page, toc)
        # Мягкий размер: не отрываем формулу от предшествующего текста.
        paragraph = len(text) > 100 and is_cyr(text[0])
        if parts and not pending_heading and (start or new_section or (size >= target and paragraph)):
            chunks.append((section, "\n\n".join(parts)))
            parts, size = [], 0
        if new_section:
            section = new_section
        parts.append(text)
        size += len(text) + 2
        pending_heading = bool(heading and len(text) < 150)
    if parts:
        chunks.append((section, "\n\n".join(parts)))
    return chunks, section


def format_math(chars):
    size = max(c["size"] for c in chars)
    base = median(c["origin"][1] for c in chars if c["size"] >= 0.85 * size)
    result = ""
    script = ""
    level = ""
    for char in chars:
        delta = char["origin"][1] - base
        mark = ""
        if char["size"] < 0.85 * size:
            mark = "^" if delta < -0.18 * size else "_" if delta > 0.08 * size else ""
        if mark != level:
            if script:
                result += level + "{" + script + "}"
                script = ""
            level = mark
        if mark:
            script += char["text"]
        else:
            result += char["text"]
    return result + (level + "{" + script + "}" if script else "")


def math_parts(page, cyr, maps):
    bars = read_bars(page)
    chars = []
    for block in page.get_text("rawdict", sort=False)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["font"] in cyr:
                    continue
                for char in span["chars"]:
                    if not char["c"].isspace():
                        text = fix_text(char["c"], span["font"], cyr, maps)
                        if char["c"] == "z" and span["font"].startswith("CMMI"):
                            if has_bar(char, span["size"], bars):
                                text += "\u0305"
                        chars.append(dict(char, size=span["size"],
                                          text=text))
    parts = []
    for left, right, top in bars:
        if not 3 < right - left < 120:
            continue
        above, below = [], []
        for num, char in enumerate(chars):
            center = (char["bbox"][0] + char["bbox"][2]) / 2
            delta = (char["origin"][1] - top) / char["size"]
            if left - 0.3 <= center <= right + 0.3:
                if -1.4 < delta < -0.05:
                    above.append(num)
                elif 0.2 < delta < 1.4:
                    below.append(num)
        if not above or not below:
            continue
        ids = set(above + below)
        if any(":U+" in chars[i]["text"] for i in ids):
            continue
        numerator = [chars[i] for i in above]
        denominator = [chars[i] for i in below]
        raw = "".join(c["text"] for c in numerator + denominator)
        text = "(" + format_math(numerator) + ")/(" + format_math(denominator) + ")"
        parts.append((ids, raw, text))
    # Пересекающиеся группы могут быть вложенной дробью: их не угадываем.
    return [(raw, text) for i, (ids, raw, text) in enumerate(parts)
            if not any(ids & other[0] for j, other in enumerate(parts) if j != i)]


def fix_math(chunks, parts):
    groups = {}
    for raw, formula in parts:
        groups.setdefault(raw, []).append(formula)
    for raw, formulas in groups.items():
        if len(set(formulas)) != 1:
            continue
        matches = []
        for num, (section, text) in enumerate(chunks):
            positions = [i for i, char in enumerate(text) if not char.isspace()]
            compact = "".join(text[i] for i in positions)
            start = compact.find(raw)
            while start >= 0:
                matches.append((num, positions[start], positions[start + len(raw) - 1] + 1))
                start = compact.find(raw, start + len(raw))
        if len(matches) == len(formulas):
            for num, start, end in reversed(matches):
                section, text = chunks[num]
                chunks[num] = (section, text[:start] + formulas[0] + text[end:])
    return chunks


def read_pdf(path, target=1800):
    blocks = []
    with pymupdf.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("PDF защищён паролем")
        cyr, maps = read_fonts(doc)
        toc = doc.get_toc()
        section = None
        for page in doc:
            units = read_page(page, cyr, maps)
            if not units:
                raise ValueError(f"Страница {page.number + 1}: нет текстового слоя; нужен отдельный разбор")
            chunks, section = split_page(units, page.number + 1, section, toc, target)
            chunks = fix_math(chunks, math_parts(page, cyr, maps))
            for num, (title, text) in enumerate(chunks, 1):
                blocks.append({"block_id": f"pdf_p{page.number + 1:03d}_b{num:03d}",
                               "page": page.number + 1, "section": title, "text": text})
        return blocks, len(doc)


def check_data(blocks, pages):
    ids = set()
    for num, block in enumerate(blocks, 1):
        if set(block) != {"block_id", "page", "section", "text"}:
            raise ValueError(f"Строка {num}: неверные поля")
        key = block["block_id"]
        if not isinstance(key, str) or not key.strip() or key in ids:
            raise ValueError(f"Строка {num}: пустой или повторный block_id")
        ids.add(key)
        if type(block["page"]) is not int or not 1 <= block["page"] <= pages:
            raise ValueError(f"Строка {num}: page вне PDF")
        if not isinstance(block["text"], str) or not block["text"].strip():
            raise ValueError(f"Строка {num}: пустой text")
        if block["section"] is not None and not isinstance(block["section"], str):
            raise ValueError(f"Строка {num}: неверный section")
    if not blocks:
        raise ValueError("Корпус пуст")
    if {b["page"] for b in blocks} != set(range(1, pages + 1)):
        raise ValueError("В корпусе отсутствуют страницы PDF")


def save_data(blocks, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for block in blocks:
            stream.write(json.dumps(block, ensure_ascii=False) + "\n")


def main():
    root = Path(__file__).resolve().parent
    path = root / "data/source.pdf"
    output = root / "data/corpus.jsonl"
    target = 1800
    if path.resolve() == output.resolve():
        raise ValueError("PDF и JSONL должны иметь разные пути")
    blocks, pages = read_pdf(path, target)
    check_data(blocks, pages)
    save_data(blocks, output)
    with output.open(encoding="utf-8") as stream:
        saved = [json.loads(line) for line in stream]
    check_data(saved, pages)
    if saved != blocks:
        raise ValueError("Сохранённый корпус отличается от извлечённого")
    sizes = sorted(len(b["text"]) for b in blocks)
    counts = Counter(b["page"] for b in blocks)
    print(f"PDF: {pages} страниц; JSONL: {len(blocks)} блоков; проверки пройдены")
    print(f"Символов в блоке: min={sizes[0]}, median={sizes[len(sizes)//2]}, max={sizes[-1]}")
    print(f"Блоков на странице: min={min(counts.values())}, max={max(counts.values())}")
    print(f"Раздел не определён: {sum(b['section'] is None for b in blocks)} блоков")
    print(f"PDF SHA256: {hashlib.sha256(path.read_bytes()).hexdigest()}")
    print(f"JSONL SHA256: {hashlib.sha256(output.read_bytes()).hexdigest()}")
    print("Формулы: линейный текст PDF; дроби, индексы и составные знаки требуют сверки с PDF.")


if __name__ == "__main__":
    main()
