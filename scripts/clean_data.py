import json
import re
import html
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional


INPUT = Path("data/cards.json")
OUT_JSON = Path("data/cleaned_cards.json")


TYPO_MAP = {
    "Altering": "Alerting",
    "bath-only": "batch-only",
}

#words that are commonly duplicated by mistake (e.g., "want want") but it only removes immediate duplicates.
DUPLICATE_WORD_RE = re.compile(r"\b(\w+)(\s+\1\b)+", flags=re.IGNORECASE)



def strip_html_and_unescape(s: str) -> str:
    """Unescape HTML entities and strip HTML tags."""
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", "", s)
    return s


def apply_typos(s: str) -> str:
    """Apply known typo replacements (case-insensitive)."""
    if not s:
        return ""
    for wrong, right in TYPO_MAP.items():
        s = re.sub(re.escape(wrong), right, s, flags=re.IGNORECASE)
    return s


def fix_soft_hyphen_linebreaks(s: str) -> str:
    """
    Fix cases like:
      'auto-\\n    sent' -> 'auto-sent'
    i.e., hyphen at end of line used to split a word.
    """
    if not s:
        return ""
    return re.sub(r"-\s*\n\s*", "-", s)


def dedupe_adjacent_words(s: str) -> str:
    """
    Fix immediate word duplication like:
      'want want' -> 'want'
    while leaving legitimate repetitions farther apart intact.
    """
    if not s:
        return ""
    # Replace any run of the same word repeated with single instance
    return DUPLICATE_WORD_RE.sub(lambda m: m.group(1), s)


def normalize_whitespace_keep_structure(s: str) -> str:
    """
    Normalize whitespace while preserving readability:
    - convert newlines into " | " separators
    - collapse multiple spaces
    - keep bullet-like structure interpretable
    """
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)  # cap excessive blank lines
    # Replace newlines with a visible separator to preserve structure for embeddings
    s = re.sub(r"\s*\n\s*", " | ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Clean up separators at ends or duplicates
    s = re.sub(r"(\|\s*){2,}", "| ", s).strip()
    s = re.sub(r"^\|\s*", "", s)
    s = re.sub(r"\s*\|$", "", s)
    return s.strip()


def extract_assigned(text: str) -> Tuple[str, List[str]]:
    """
    Extract assignee(s) from either:
      - a standalone line: "Assigned to: NAME" / "Assigned: NAME"
      - or inline: " ... Assigned to: NAME" / " ... Assigned: NAME"
    Supports multiple names separated by comma or 'and'.

    Returns:
      (text_without_assigned_clause, [names...])
    """
    if not text:
        return text, []

    assigned: List[str] = []

    # Capture assignment anywhere, not just at start-of-string or after newline.
    # Stops at a separator, end, or another common section marker.
    pattern = re.compile(
        r"(?:\bAssigned\s*(?:to)?\s*:\s*)([^|]+?)(?=$|\||\bSteps\b|\bStep\b)",
        flags=re.IGNORECASE,
    )

    def repl(m: re.Match) -> str:
        names = m.group(1).strip()
        parts = re.split(r",\s*| and ", names, flags=re.IGNORECASE)
        for p in parts:
            p = p.strip(" .")
            if p:
                assigned.append(p)
        return ""  # remove the assigned clause

    new_text = pattern.sub(repl, text)

    # Remove leftover double spaces / dangling punctuation after removal
    new_text = re.sub(r"\s{2,}", " ", new_text).strip()
    new_text = re.sub(r"\s+\|", " |", new_text).strip()
    new_text = re.sub(r"\|\s+\|", "|", new_text).strip(" |")

    # Deduplicate names while preserving order
    seen = set()
    deduped = []
    for a in assigned:
        key = a.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(a)

    return new_text, deduped


def clean_text_pipeline(s: str) -> str:
    """
    Full text pipeline for name/desc fields:
    1) unescape + strip html
    2) fix hyphen line-break word splits
    3) apply known typos
    4) de-dupe accidental repeated words
    5) normalize whitespace but keep structure via separators
    """
    if not s:
        return ""
    s = strip_html_and_unescape(s)
    s = fix_soft_hyphen_linebreaks(s)
    s = apply_typos(s)
    s = dedupe_adjacent_words(s)
    s = normalize_whitespace_keep_structure(s)
    return s


def normalize_labels(labels):
    """
    Normalize Trello labels, keeping only:
      - id
      - idBoard
      - name
      - nodeId
      - color
      - uses
    """
    if not labels:
        return []

    allowed_keys = {"id", "idBoard", "name", "nodeId", "color", "uses"}
    out = []

    for l in labels:
        if isinstance(l, dict):
            cleaned = {k: l.get(k) for k in allowed_keys if k in l}
            # Drop labels that have no meaningful content
            if cleaned.get("id") or cleaned.get("name"):
                out.append(cleaned)

        elif isinstance(l, str):
            # If labels come in as strings, preserve name only
            name = l.strip()
            if name:
                out.append({"name": name})

    return out



def normalize_members(members: Any) -> List[Dict[str, str]]:
    """Normalize members to [{fullName: ...}] and drop empties."""
    if not members:
        return []
    out: List[Dict[str, str]] = []
    for m in members:
        if isinstance(m, dict):
            name = (m.get("fullName") or m.get("name") or "").strip()
            if name:
                out.append({"fullName": name})
        elif isinstance(m, str):
            nm = m.strip()
            if nm:
                out.append({"fullName": nm})
    # Deduplicate by name (case-insensitive)
    seen = set()
    deduped = []
    for m in out:
        key = m["fullName"].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(m)
    return deduped


def add_assigned_to_members(members: List[Dict[str, str]], assigned: List[str]) -> List[Dict[str, str]]:
    """Ensure any extracted assigned names appear in members (without duplicates)."""
    existing = {m["fullName"].lower() for m in members}
    for a in assigned:
        if a.lower() not in existing:
            members.append({"fullName": a})
            existing.add(a.lower())
    return members


def normalize_card(c: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize a Trello card into a stable, cleaned schema for downstream embedding."""
    if not isinstance(c, dict):
        return None
    if not c.get("id") and not c.get("name"):
        return None

    raw_name = str(c.get("name") or "")
    raw_desc = str(c.get("desc") or "")

    # Clean *first*, but preserve structure; then extract assignments from the cleaned text
    name_clean = clean_text_pipeline(raw_name)
    name_clean, assigned_from_name = extract_assigned(name_clean)

    desc_clean = clean_text_pipeline(raw_desc)
    desc_clean, assigned_from_desc = extract_assigned(desc_clean)

    assigned = assigned_from_name + assigned_from_desc
    # Final dedupe for assigned (case-insensitive) while preserving order
    seen = set()
    assigned_deduped = []
    for a in assigned:
        k = a.lower()
        if k not in seen:
            seen.add(k)
            assigned_deduped.append(a)
    assigned = assigned_deduped

    members = normalize_members(c.get("members", []))
    members = add_assigned_to_members(members, assigned)

    labels = normalize_labels(c.get("labels", []))

    cleaned: Dict[str, Any] = {
        "id": c.get("id"),
        "name": name_clean,
        "desc": desc_clean,
        "labels": labels,
        "idList": c.get("idList"),
        "due": c.get("due"),
        "shortUrl": c.get("shortUrl"),
        "dateLastActivity": c.get("dateLastActivity"),
        "attachments": c.get("attachments") or [],
        "checklists": c.get("checklists") or [],
        "members": members,
    }

    if assigned:
        cleaned["assigned"] = assigned if len(assigned) > 1 else assigned[0]

    return cleaned


def load_cards(path: Path = INPUT) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_out(cards: List[Dict[str, Any]], out: Path = OUT_JSON) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(cards)} cards to {out}")




def main() -> None:
    cards = load_cards()

    seen_ids = set()
    cleaned_cards: List[Dict[str, Any]] = []
    skipped = 0

    for c in cards:
        nc = normalize_card(c)
        if not nc:
            skipped += 1
            continue

        cid = nc.get("id")
        if cid in seen_ids:
            skipped += 1
            continue
        seen_ids.add(cid)

        cleaned_cards.append(nc)

    write_out(cleaned_cards)
    print(f"Wrote {len(cleaned_cards)} cleaned cards to {OUT_JSON} (skipped {skipped})")


if __name__ == "__main__":
    main()
