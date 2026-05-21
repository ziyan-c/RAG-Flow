from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz


@dataclass(frozen=True)
class PdfInfo:
    pdf_id: str
    path: Path
    product_group: str
    document_type: str
    target_questions: int
    visual_priority: bool = False


PDFS: list[PdfInfo] = [
    PdfInfo(
        "dss_pc_manual",
        Path("source-pdfs/DSS/Dahua_DSS_Ultimate_PC_Client_User_Manual_V8.8.0.pdf"),
        "DSS",
        "pc_client_user_manual",
        58,
    ),
    PdfInfo(
        "dss_web_manual",
        Path("source-pdfs/DSS/Dahua_DSS_Ultimate_Web_Client_User_Manual_V8.8.0.pdf"),
        "DSS",
        "web_client_user_manual",
        48,
    ),
    PdfInfo(
        "dss_quick_deployment",
        Path("source-pdfs/DSS/Dahua_DSS_Ultimate_Quick_Deployment_Manual_V8.8.0.pdf"),
        "DSS",
        "quick_deployment_manual",
        15,
    ),
    PdfInfo(
        "hdcvi_camera_manual",
        Path("source-pdfs/HAC-HF3805G/HDCVI_Camera_User_Manual_V1.0.7.pdf"),
        "HAC-HF3805G",
        "camera_user_manual",
        21,
    ),
    PdfInfo(
        "switch_user_manual",
        Path("source-pdfs/S3006-4ET-60/Ethernet_Switch_(4_8-Port_Unmanaged_Desktop_Switch)_User_s_Manual_V1.0.1.pdf"),
        "S3006-4ET-60",
        "switch_user_manual",
        13,
    ),
    PdfInfo(
        "s3006_datasheet",
        Path("source-pdfs/S3006-4ET-60/S3006-4ET-60_datasheet_20251106.pdf"),
        "S3006-4ET-60",
        "datasheet",
        8,
    ),
    PdfInfo(
        "hac_hf3805g_datasheet",
        Path("source-pdfs/HAC-HF3805G/DH-HAC-HF3805G-datasheet3.pdf"),
        "HAC-HF3805G",
        "datasheet",
        8,
    ),
    PdfInfo(
        "hac_open_source_notice",
        Path("source-pdfs/HAC-HF3805G/DAHUA_HDCVI_CAMERA_OPEN_SOURCE_SOFTWARE_NOTICE__VERSION_A_V1.0.1-Eng.pdf"),
        "HAC-HF3805G",
        "open_source_notice",
        4,
    ),
    PdfInfo(
        "hdcvi_box_install_guide",
        Path("source-pdfs/HAC-HF3805G/HDCVI_Box_Camera_Installation_Guide_V1.0.0-Eng.pdf"),
        "HAC-HF3805G",
        "visual_installation_guide",
        5,
        visual_priority=True,
    ),
    PdfInfo(
        "hac_hf3231e_installation",
        Path("source-pdfs/HAC-HF3805G/HAC-HF3231E_Installation_20171121.pdf"),
        "HAC-HF3805G",
        "visual_mounting_compatibility",
        4,
        visual_priority=True,
    ),
    PdfInfo(
        "hac_hf3805g_dimension",
        Path("source-pdfs/HAC-HF3805G/HAC-HF3805G_Dimention_20171128.pdf"),
        "HAC-HF3805G",
        "visual_dimension_drawing",
        4,
        visual_priority=True,
    ),
    PdfInfo(
        "s3006_dimensions",
        Path("source-pdfs/S3006-4ET-60/S3006-4ET-36_DIMENSIONS.pdf"),
        "S3006-4ET-60",
        "visual_dimension_drawing",
        4,
        visual_priority=True,
    ),
    PdfInfo(
        "s3006_installation_method",
        Path("source-pdfs/S3006-4ET-60/S3006-4ET-36_INSTALLATION_METHOD_EN.pdf"),
        "S3006-4ET-60",
        "visual_installation_method",
        4,
        visual_priority=True,
    ),
    PdfInfo(
        "s3006_port_diagram",
        Path("source-pdfs/S3006-4ET-60/S3006-4ET-36_PORT.pdf"),
        "S3006-4ET-60",
        "visual_port_diagram",
        4,
        visual_priority=True,
    ),
]


VISUAL_NOTES: dict[str, dict[int, dict[str, Any]]] = {
    "s3006_dimensions": {
        0: {
            "summary": "Dimension drawing for the S3006-4ET-36/S3006 desktop switch, showing width 115.5 mm [4.55 in], depth 84.7 mm [3.33 in], body depth 88.2 mm [3.47 in], and height 27.0 mm [1.06 in].",
            "key_facts": [
                "Width is marked as 115.5 mm [4.55 in].",
                "Depth is marked as 84.7 mm [3.33 in], with another body-depth dimension of 88.2 mm [3.47 in].",
                "Height is marked as 27.0 mm [1.06 in].",
            ],
        }
    },
    "s3006_installation_method": {
        0: {
            "summary": "Five-step visual installation method for mounting the S3006 switch, shown as numbered diagrams rather than explanatory prose.",
            "key_facts": [
                "The installation method is represented as five numbered visual steps.",
                "The document relies on diagrams, so the step order must be read from the rendered page.",
            ],
        }
    },
    "s3006_port_diagram": {
        0: {
            "summary": "Port layout diagram for the S3006 switch. The front panel shows a Default Extend DIP area at the left, PoE ports 1-4 labeled Port 1-4: 10/100Mbps (PoE), uplink ports 5 and 6 labeled Port 5-6: 10/100Mbps, Link/Act and PoE indicators above the ports, and a PWR indicator at the far right. The rear view shows a grounding point and a DC IN power jack.",
            "key_facts": [
                "PoE ports 1-4 are labeled Port 1-4: 10/100Mbps (PoE).",
                "Ports 5 and 6 are marked as uplink ports.",
                "A PWR indicator appears at the far right of the front panel.",
                "The rear view shows a grounding point and a DC IN jack.",
            ],
        }
    },
    "hac_hf3231e_installation": {
        0: {
            "summary": "Mounting accessory compatibility sheet for HAC-HF3231E/HAC-HF3805G-style cameras, showing wall mount and ceiling mount options.",
            "key_facts": [
                "Wall Mount is associated with PFB121W.",
                "Ceiling Mount is associated with PFB110W.",
            ],
        }
    },
    "hac_hf3805g_dimension": {
        0: {
            "summary": "Dimension drawing for HAC-HF3805G. The top/side views show a main length of 144.5 mm [5.69 in], body length of 134 mm [5.28 in], front width of 82 mm [3.23 in], front height of 73.7 mm [2.9 in], and mounting-hole spacing marks of 54 mm [2.13 in] and 31 mm [1.22 in]. The screw note reads 4-1/4-20X6UCN.",
            "key_facts": [
                "Main length is marked 144.5 mm [5.69 in].",
                "Body length is marked 134 mm [5.28 in].",
                "The front view marks 82 mm [3.23 in] by 73.7 mm [2.9 in].",
                "Mounting-hole spacing marks include 54 mm [2.13 in] and 31 mm [1.22 in].",
                "The screw note reads 4-1/4-20X6UCN.",
            ],
        }
    },
    "hdcvi_box_install_guide": {
        0: {
            "summary": "First page of a visual HDCVI box camera installation guide. It shows lens attachment steps 1A, 1B, 1C; wall-mount positioning as 2.1; drilling/marking as 2.1.1; fastening the bracket to the wall as 2.1.2; mounting the camera on the wall bracket as 2.1.3; and the completed wall-mounted view as 2.1.4. It also starts ceiling-mount steps 2.2 and 2.2.1.",
            "key_facts": [
                "Steps 1A, 1B, and 1C show lens attachment/removal sequence details.",
                "Step 2.1 is the wall-mount branch.",
                "Step 2.1.1 shows drilling/marking for the wall bracket.",
                "Step 2.1.2 shows fastening the wall bracket.",
                "Step 2.1.3 shows mounting the camera onto the bracket.",
                "Step 2.1.4 shows the completed wall-mounted camera.",
                "Step 2.2 begins the ceiling-mount branch.",
            ],
        },
        1: {
            "summary": "Second page of the visual HDCVI box camera installation guide. It continues the ceiling-mount branch with 2.2.2 and 2.2.3, then shows enclosure/cover installation steps 3.1, 3.2, and 3.3, and camera angle/image adjustment steps 4.1 through 4.4. The page footer shows document code 1.2.51.32.16704-000.",
            "key_facts": [
                "Steps 2.2.2 and 2.2.3 continue the ceiling-mount branch.",
                "Steps 3.1, 3.2, and 3.3 show camera enclosure/cover installation.",
                "Steps 4.1 through 4.4 show camera angle/image adjustment.",
                "The document code on the second page is 1.2.51.32.16704-000.",
            ],
        },
    },
}


QUESTION_TYPE_TARGETS = Counter(
    {
        "fact": 50,
        "procedure": 45,
        "table_or_spec": 35,
        "visual": 25,
        "cross_page": 25,
        "multi_pdf_comparison": 15,
        "safety_or_notice": 5,
    }
)


KEYWORD_GROUPS: dict[str, list[str]] = {
    "login": ["login", "log in", "password", "account"],
    "live_view": ["live view", "view", "video", "window", "channel"],
    "alarm": ["alarm", "event", "alert", "trigger"],
    "map": ["map", "gis", "emergency", "location"],
    "access_control": ["access control", "door", "card", "visitor"],
    "video_wall": ["video wall", "wall"],
    "license_plate": ["license plate", "vehicle", "parking"],
    "deployment": ["deploy", "deployment", "install", "server", "service"],
    "safety": ["warning", "caution", "privacy", "notice"],
    "network": ["ip address", "network", "port", "poe", "ethernet"],
    "storage": ["storage", "record", "recording", "disk"],
    "camera_menu": ["exposure", "white balance", "osd", "backlight", "privacy mask"],
    "spec": ["technical specification", "specification", "power", "dimension", "effective pixels"],
}


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "will",
    "manual",
    "user",
    "system",
    "device",
    "page",
    "figure",
    "table",
    "click",
    "select",
    "section",
    "dss",
    "ultimate",
    "client",
}


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return text or "item"


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sentence_split(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", clean_text(text))
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9●])", text)
    result = []
    for chunk in chunks:
        sentence = clean_text(chunk)
        if is_good_support(sentence):
            result.append(sentence)
    return result


def is_good_support(sentence: str) -> bool:
    sentence = clean_text(sentence)
    if not 45 <= len(sentence) <= 520:
        return False
    if re.search(r"\.{6,}", sentence):
        return False
    if re.match(r"^[,;:)\]}]", sentence):
        return False
    if sentence.count(" ") < 5:
        return False
    if len(re.findall(r"[A-Za-z]", sentence)) < 25:
        return False
    return True


def compact_support(text: str, limit: int = 420) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    cut = text[: limit - 3].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + "..."


def support_score(sentence: str) -> int:
    sentence = clean_text(sentence)
    low = sentence.lower()
    score = min(len(sentence), 260) // 40
    if re.match(r"^(figure|table)\s+\S+", low):
        score -= 3
    if "click ," in low or "select >" in low:
        score -= 1
    if any(token in low for token in ["configure", "select", "click", "enter", "must", "can", "supports", "introduces"]):
        score += 2
    if any(token in low for token in ["foreword general", "copyright", "revision history"]):
        score -= 1
    return score


def first_good_sentence(text: str, *, keywords: list[str] | None = None) -> str:
    sentences = sentence_split(text)
    if keywords:
        lowered = [kw.lower() for kw in keywords]
        for sentence in sentences:
            low = sentence.lower()
            if any(kw in low for kw in lowered):
                return sentence
    return sentences[0] if sentences else compact_support(text)


def best_support(card: dict[str, Any], *, keywords: list[str] | None = None) -> str:
    candidates = [str(item) for item in card.get("key_facts", []) if str(item).strip()]
    candidates.extend(sentence_split(card.get("text_excerpt", "")))
    if keywords:
        lowered = [kw.lower() for kw in keywords]
        matches = [
            candidate
            for candidate in candidates
            if is_good_support(candidate) and any(kw in candidate.lower() for kw in lowered)
        ]
        if matches:
            return compact_support(max(matches, key=support_score))
    good_candidates = [candidate for candidate in candidates if is_good_support(candidate)]
    if good_candidates:
        return compact_support(max(good_candidates, key=support_score))
    return compact_support(card.get("visual_summary") or card.get("text_excerpt") or card.get("title_hint", ""))


def visual_support(card: dict[str, Any], question: str, answer: str) -> str:
    facts = [str(item).strip() for item in card.get("key_facts", []) if str(item).strip()]
    if not facts:
        return compact_support(card.get("visual_summary") or card.get("text_excerpt") or card.get("title_hint", ""))

    query = f"{question} {answer}".lower()
    broad_visual_phrases = [
        "physical dimensions",
        "dimension questions",
        "size questions",
        "visual evidence",
        "rendered",
        "extracted text",
        "best evidence",
        "source pdf",
        "purpose of",
    ]
    if card.get("visual_summary") and any(phrase in query for phrase in broad_visual_phrases):
        return compact_support(card["visual_summary"], 520)

    terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}|[0-9]+(?:\.[0-9]+)?", query))
    selected = []
    for fact in facts:
        fact_low = fact.lower()
        overlap = {term for term in terms if term in fact_low and term not in STOPWORDS}
        has_number = any(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", term) for term in overlap)
        has_identifier = any(any(char.isalpha() for char in term) and any(char.isdigit() for char in term) for term in overlap)
        if len(overlap) >= 2 or has_number or has_identifier:
            selected.append(fact)

    if selected:
        return compact_support(" ".join(selected), 520)
    return compact_support(card.get("visual_summary") or " ".join(facts), 520)


def comparison_modality(card: dict[str, Any], *, requires_visual: bool) -> str:
    if requires_visual and card.get("visual_summary"):
        return "diagram"
    return "text"


def cross_pdf_text_support(card: dict[str, Any], question: str, answer: str) -> str:
    low_question = question.lower()
    if "privacy categories" in low_question:
        return (
            "Privacy Protection Notice As the device user or data controller, you might collect the personal data of "
            "others such as their face, audio, fingerprints, and license plate number."
        )
    if "safety-warning convention" in low_question:
        return "Safety Instructions The following signal words might appear in the manual."
    keywords = extract_keywords(f"{question} {answer}", limit=10)
    if card["pdf_id"] == "s3006_datasheet" and "document purpose" in question.lower():
        keywords = ["Series Overview", "unmanaged", "PoE switch"]
    elif card["pdf_id"] == "hac_hf3805g_datasheet" and "differ in focus" in question.lower():
        return (
            "System Overview DH-HAC-HFW3805G adopts high performance ISP and advanced 4/3\" 8MP image sensor, "
            "which can bring vaster coverage and superiorer image details in 7/24 with the 4K resolution."
        )
    return best_support(card, keywords=keywords)


def extract_title(text: str, fallback: str) -> str:
    lines = [clean_text(line) for line in text.splitlines() if clean_text(line)]
    for line in lines[:8]:
        if 8 <= len(line) <= 130 and not re.fullmatch(r"[IVXLC0-9 .-]+", line):
            return line
    return fallback


def clean_toc_title(title: str) -> str:
    title = clean_text(title)
    title = re.sub(r"^[0-9.]+\s*", "", title)
    return title.strip() or "Overview"


def toc_path_for_page(toc: list[list[Any]], page_number: int) -> list[str]:
    active: dict[int, str] = {}
    for level, title, toc_page_number in toc:
        if toc_page_number > page_number:
            break
        active[int(level)] = clean_toc_title(str(title))
        for stale_level in list(active):
            if stale_level > int(level):
                active.pop(stale_level, None)
    return [active[level] for level in sorted(active) if active[level]]


def extract_keywords(text: str, limit: int = 12) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text.lower())
    counts = Counter(w for w in words if w not in STOPWORDS)
    return [word for word, _count in counts.most_common(limit)]


def page_modality(text_chars: int, image_count: int, drawing_count: int, visual_priority: bool) -> list[str]:
    modalities = []
    if text_chars:
        modalities.append("text")
    if image_count:
        modalities.append("image")
    if drawing_count:
        modalities.append("diagram")
    if visual_priority and "diagram" not in modalities and "image" not in modalities:
        modalities.append("diagram")
    return modalities or ["visual"]


def is_visual_modality(modality: str) -> bool:
    return modality in {"image", "diagram", "screenshot", "table"}


def scan_pdf(info: PdfInfo) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    doc = fitz.open(info.path)
    toc = doc.get_toc(simple=True)
    page_cards = []
    total_text_chars = 0
    image_pages = 0
    drawing_pages = 0
    for page_idx, page in enumerate(doc):
        raw_text = page.get_text("text") or ""
        text = clean_text(raw_text)
        text_chars = len(text)
        total_text_chars += text_chars
        image_count = len(page.get_images(full=True))
        drawing_count = len(page.get_drawings())
        if image_count:
            image_pages += 1
        if drawing_count:
            drawing_pages += 1
        note = VISUAL_NOTES.get(info.pdf_id, {}).get(page_idx, {})
        section_path = toc_path_for_page(toc, page_idx + 1)
        title_hint = section_path[-1] if section_path else extract_title(text, info.path.stem)
        key_facts = list(note.get("key_facts", []))
        if not key_facts and text:
            key_facts = [compact_support(sentence) for sentence in sentence_split(text)[:4]]
        page_cards.append(
            {
                "pdf_id": info.pdf_id,
                "pdf": str(info.path),
                "file_name": info.path.name,
                "product_group": info.product_group,
                "document_type": info.document_type,
                "page_idx": page_idx,
                "page_number": page_idx + 1,
                "title_hint": title_hint,
                "section_path": section_path,
                "text_chars": text_chars,
                "text_excerpt": compact_support(text, 1200),
                "visual_summary": note.get("summary", ""),
                "key_facts": key_facts[:8],
                "keywords": extract_keywords(text),
                "image_count": image_count,
                "drawing_count": drawing_count,
                "modality": page_modality(text_chars, image_count, drawing_count, info.visual_priority),
                "needs_visual_review": bool(info.visual_priority or text_chars < 120),
            }
        )
    inventory = {
        "pdf_id": info.pdf_id,
        "path": str(info.path),
        "file_name": info.path.name,
        "product_group": info.product_group,
        "document_type": info.document_type,
        "target_questions": info.target_questions,
        "page_count": len(doc),
        "text_chars": total_text_chars,
        "avg_text_chars_per_page": round(total_text_chars / max(1, len(doc)), 1),
        "image_pages": image_pages,
        "drawing_pages": drawing_pages,
        "visual_priority": info.visual_priority,
        "toc_count": len(toc),
        "toc_sample": [{"level": item[0], "title": item[1], "page_number": item[2]} for item in toc[:40]],
    }
    return inventory, page_cards


def grouped_pages(cards: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        text = " ".join([card.get("title_hint", ""), card.get("text_excerpt", "")]).lower()
        for group, keywords in KEYWORD_GROUPS.items():
            if any(keyword in text for keyword in keywords):
                grouped[group].append(card)
    return grouped


def evidence_from_card(card: dict[str, Any], *, support: str | None = None, modality: str | None = None) -> dict[str, Any]:
    chosen_modality = modality or ("diagram" if card.get("needs_visual_review") else card["modality"][0])
    return {
        "pdf_id": card["pdf_id"],
        "pdf": card["pdf"],
        "page_idx": card["page_idx"],
        "page_number": card["page_number"],
        "modality": chosen_modality,
        "support": compact_support(support or (card.get("key_facts") or [card.get("text_excerpt", "")])[0]),
    }


def make_question(
    *,
    question: str,
    answer: str,
    evidence: list[dict[str, Any]],
    question_type: str,
    difficulty: str,
    requires_visual: bool | None = None,
) -> dict[str, Any]:
    source_pdfs = sorted({item["pdf"] for item in evidence})
    page_keys = {(item["pdf"], item["page_idx"]) for item in evidence}
    visual = any(is_visual_modality(item["modality"]) for item in evidence) if requires_visual is None else requires_visual
    return {
        "question": question,
        "answer": answer,
        "source_pdfs": source_pdfs,
        "evidence": evidence,
        "question_type": question_type,
        "requires_visual": visual,
        "requires_multiple_pages": len(page_keys) > 1,
        "requires_multiple_pdfs": len(source_pdfs) > 1,
        "difficulty": difficulty,
    }


def card_for(cards_by_pdf: dict[str, list[dict[str, Any]]], pdf_id: str, page_idx: int) -> dict[str, Any]:
    for card in cards_by_pdf[pdf_id]:
        if card["page_idx"] == page_idx:
            return card
    raise KeyError(f"Missing card {pdf_id} page {page_idx}")


def representative_cards(cards: list[dict[str, Any]], limit: int, *, min_text: int = 120) -> list[dict[str, Any]]:
    candidates = [card for card in cards if is_content_card(card, min_text=min_text)]
    if not candidates:
        return cards[:limit]
    if len(candidates) <= limit:
        return candidates
    step = max(1, len(candidates) // limit)
    selected = candidates[::step][:limit]
    seen = {card["page_idx"] for card in selected}
    for card in candidates:
        if len(selected) >= limit:
            break
        if card["page_idx"] not in seen:
            selected.append(card)
            seen.add(card["page_idx"])
    return selected[:limit]


def is_content_card(card: dict[str, Any], *, min_text: int = 120) -> bool:
    text = card.get("text_excerpt", "")
    if card["text_chars"] < min_text:
        return False
    if re.search(r"\.{6,}", text):
        return False
    if "Table of Contents" in text[:260]:
        return False
    return True


def make_visual_questions(cards_by_pdf: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    qas: list[dict[str, Any]] = []
    visual_specs = [
        (
            "s3006_dimensions",
            0,
            [
                (
                    "What are the main external dimensions shown for the S3006 switch in the dimensions drawing?",
                    "The drawing marks the switch at 115.5 mm [4.55 in] wide, 84.7 mm [3.33 in] deep, with an additional 88.2 mm [3.47 in] body-depth mark, and 27.0 mm [1.06 in] high.",
                    "visual",
                    "medium",
                ),
                (
                    "Which dimension in the S3006 dimensions drawing gives the device height?",
                    "The height is shown as 27.0 mm, or 1.06 inches.",
                    "visual",
                    "easy",
                ),
                (
                    "Why should a RAG answerer inspect the rendered S3006 dimensions PDF instead of relying only on extracted text?",
                    "The useful information is the dimension drawing itself: the page consists mainly of visual measurement arrows and labels, so layout and labels from the rendered page are needed.",
                    "visual",
                    "medium",
                ),
                (
                    "What unit pair is used in the S3006 dimensions drawing?",
                    "The drawing gives dimensions in millimeters with inch equivalents in brackets.",
                    "visual",
                    "easy",
                ),
            ],
        ),
        (
            "s3006_installation_method",
            0,
            [
                (
                    "How is the S3006 installation method represented in its installation-method PDF?",
                    "It is represented as a five-step visual sequence with numbered diagrams rather than a prose procedure.",
                    "visual",
                    "easy",
                ),
                (
                    "How many numbered visual steps are shown in the S3006 installation-method sheet?",
                    "The sheet shows five numbered visual steps.",
                    "visual",
                    "easy",
                ),
                (
                    "What kind of evidence is needed to answer questions about the order of the S3006 installation method?",
                    "The rendered installation diagram is needed, because the order is conveyed by numbered visuals.",
                    "visual",
                    "medium",
                ),
                (
                    "Is the S3006 installation-method sheet mainly a text instruction page or a visual instruction page?",
                    "It is mainly a visual instruction page.",
                    "visual",
                    "easy",
                ),
            ],
        ),
        (
            "s3006_port_diagram",
            0,
            [
                (
                    "What is the purpose of the S3006-4ET-36 PORT PDF in the source set?",
                    "It is a visual port-layout reference for the S3006 switch, used to identify physical port and connector arrangement.",
                    "visual",
                    "medium",
                ),
                (
                    "Why is the S3006 port diagram marked as requiring visual evidence?",
                    "The page has no useful extractable text; the relevant information is the rendered port layout.",
                    "visual",
                    "medium",
                ),
                (
                    "What should a retrieval system return for a question about the physical arrangement of S3006 ports?",
                    "It should return the port diagram page, because the front panel shows PoE ports 1-4, uplink ports 5 and 6, status indicators, and the rear DC IN/grounding layout.",
                    "visual",
                    "medium",
                ),
                (
                    "Which S3006 source PDF is the best evidence for physical port-layout questions?",
                    "The S3006-4ET-36_PORT.pdf file is the best evidence for physical port-layout questions.",
                    "visual",
                    "easy",
                ),
                (
                    "Which S3006 ports are marked as uplink ports in the port diagram?",
                    "Ports 5 and 6 are marked as uplink ports.",
                    "visual",
                    "easy",
                ),
                (
                    "What connectors or indicators are shown on the rear view of the S3006 port diagram?",
                    "The rear view shows a grounding point and a DC IN power jack.",
                    "visual",
                    "medium",
                ),
            ],
        ),
        (
            "hac_hf3231e_installation",
            0,
            [
                (
                    "Which wall-mount accessory is shown for the HAC-HF3231E/HAC-HF3805G-style installation sheet?",
                    "The wall-mount accessory shown is PFB121W.",
                    "visual",
                    "easy",
                ),
                (
                    "Which ceiling-mount accessory is shown in the HAC-HF3231E installation sheet?",
                    "The ceiling-mount accessory shown is PFB110W.",
                    "visual",
                    "easy",
                ),
                (
                    "What two mounting categories does the HAC-HF3231E installation sheet compare?",
                    "It compares Wall Mount and Ceiling Mount options.",
                    "visual",
                    "easy",
                ),
                (
                    "Why is the HAC-HF3231E installation sheet useful for visual retrieval tests?",
                    "It is a compact visual compatibility sheet where the answer depends on reading labels from the rendered page.",
                    "visual",
                    "medium",
                ),
            ],
        ),
        (
            "hac_hf3805g_dimension",
            0,
            [
                (
                    "What type of document is HAC-HF3805G_Dimention_20171128.pdf?",
                    "It is a dimension drawing for the HAC-HF3805G camera.",
                    "visual",
                    "easy",
                ),
                (
                    "Why should dimension questions for HAC-HF3805G_Dimention_20171128.pdf use rendered-page evidence?",
                    "The page has effectively no extractable text, so the dimensional information is carried by the visual/vector drawing.",
                    "visual",
                    "medium",
                ),
                (
                    "What should be retrieved for a question asking about HAC-HF3805G physical dimensions?",
                    "The HAC-HF3805G dimension drawing page should be retrieved; it contains the 144.5 mm main length, 134 mm body length, 82 mm front width, and 73.7 mm front height markings.",
                    "visual",
                    "medium",
                ),
                (
                    "Is HAC-HF3805G_Dimention_20171128.pdf primarily a prose manual or a visual drawing?",
                    "It is primarily a visual dimension drawing.",
                    "visual",
                    "easy",
                ),
                (
                    "What main length values are shown in the HAC-HF3805G dimension drawing?",
                    "The drawing shows a main length of 144.5 mm [5.69 in] and a body length of 134 mm [5.28 in].",
                    "visual",
                    "medium",
                ),
                (
                    "What front-view width and height are marked in the HAC-HF3805G dimension drawing?",
                    "The front view marks 82 mm [3.23 in] wide and 73.7 mm [2.9 in] high.",
                    "visual",
                    "medium",
                ),
            ],
        ),
        (
            "hdcvi_box_install_guide",
            0,
            [
                (
                    "Which early labels appear on the first page of the HDCVI Box Camera Installation Guide?",
                    "The first page includes visual labels such as 1A, 1B, 1C, 2.1, 2.1.1, 2.1.2, 2.1.3, and 2.1.4.",
                    "visual",
                    "medium",
                ),
                (
                    "In the HDCVI Box Camera Installation Guide, which branch begins with step 2.1?",
                    "Step 2.1 begins the wall-mount branch.",
                    "visual",
                    "medium",
                )
            ],
        ),
        (
            "hdcvi_box_install_guide",
            1,
            [
                (
                    "What document code is visible on the second page of the HDCVI Box Camera Installation Guide?",
                    "The visible document code is 1.2.51.32.16704-000.",
                    "visual",
                    "easy",
                ),
                (
                    "Which later step groups appear on the second page of the HDCVI Box Camera Installation Guide?",
                    "The second page continues with 2.2.x steps and later 3.x and 4.x steps.",
                    "visual",
                    "medium",
                ),
                (
                    "Why does the HDCVI Box Camera Installation Guide need visual QA coverage?",
                    "Its procedure is mostly diagrammatic, so the ordering and labels are on the rendered pages rather than in normal prose.",
                    "visual",
                    "medium",
                ),
                (
                    "How many pages of visual evidence does the HDCVI Box Camera Installation Guide provide in this source set?",
                    "It provides two pages of diagrammatic installation evidence.",
                    "visual",
                    "easy",
                ),
            ],
        ),
    ]
    for pdf_id, page_idx, questions in visual_specs:
        card = card_for(cards_by_pdf, pdf_id, page_idx)
        for question, answer, question_type, difficulty in questions:
            qas.append(
                make_question(
                    question=question,
                    answer=answer,
                    evidence=[evidence_from_card(card, support=visual_support(card, question, answer), modality="diagram")],
                    question_type=question_type,
                    difficulty=difficulty,
                    requires_visual=True,
                )
            )
    return qas


def make_small_doc_questions(cards_by_pdf: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    qas: list[dict[str, Any]] = []
    card = card_for(cards_by_pdf, "hac_open_source_notice", 0)
    qas.extend(
        [
            make_question(
                question="What is the purpose of the Dahua HDCVI camera open-source software notice?",
                answer="It provides open-source software notices and warranty-disclaimer information for software components associated with the product.",
                evidence=[
                    evidence_from_card(
                        card,
                        support=(
                            "OPEN SOURCE SOFTWARE NOTICE This document contains open source software notice for the product "
                            "which includes this file does not necessarily use all of the third party software components "
                            "referred to below. Warranty Disclaimer THE OPEN SOURCE SOFTWARE IN THIS PRODUCT IS DISTRIBUTED "
                            "IN THE HOPE THAT IT WILL BE USEFUL, BUT WITHOUT ANY WARRANTY."
                        ),
                    )
                ],
                question_type="safety_or_notice",
                difficulty="easy",
                requires_visual=False,
            ),
            make_question(
                question="According to the open-source software notice, does every listed third-party component necessarily apply to the product?",
                answer="No. The notice says the product that includes the file does not necessarily use all of the third-party software components referred to in the document.",
                evidence=[evidence_from_card(card, support=best_support(card, keywords=["does not necessarily use"]))],
                question_type="safety_or_notice",
                difficulty="medium",
                requires_visual=False,
            ),
        ]
    )
    card2 = card_for(cards_by_pdf, "hac_open_source_notice", 1)
    qas.append(
        make_question(
            question="What point does the open-source notice make about linking this file with other works?",
            answer="It states that linking this file with other works does not by itself cause the resulting work to be covered by the GNU General Public License, while the source code for this file must still be made available.",
            evidence=[
                evidence_from_card(
                    card2,
                    support=(
                        "and link it with other works to produce a work based on this file, this file does not by itself "
                        "cause the resulting work to be covered by the GNU General Public License. However the source code "
                        "for this file must still be made available in accordance with section (3) of the GNU General Public License v2."
                    ),
                )
            ],
            question_type="safety_or_notice",
            difficulty="hard",
            requires_visual=False,
        )
    )
    card3 = card_for(cards_by_pdf, "hac_open_source_notice", 2)
    qas.append(
        make_question(
            question="How does the notice characterize BSD-style licenses?",
            answer="It characterizes BSD-style licenses as having extremely minimal restrictions and allowing software to be modified and used in proprietary software with source code kept secret.",
            evidence=[evidence_from_card(card3, support=best_support(card3, keywords=["BSD-style licenses"]))],
            question_type="safety_or_notice",
            difficulty="medium",
            requires_visual=False,
        )
    )
    return qas


def make_spec_questions(cards_by_pdf: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    qas: list[dict[str, Any]] = []
    specs = [
        (
            "s3006_datasheet",
            [
                (
                    0,
                    "What is the high-level product positioning of the S3006-4ET-60 datasheet?",
                    "It describes a highly reliable unmanaged PoE switch with a high-performance switching engine, large buffer memory, and optimized transmission performance.",
                    ["highly reliable", "unmanaged", "poe switch"],
                ),
                (
                    1,
                    "What PoE standards are listed for the S3006-4ET-60?",
                    "The datasheet lists IEEE 802.3af and IEEE 802.3at as the PoE protocols.",
                    ["IEEE 802.3af", "IEEE 802.3at"],
                ),
                (
                    1,
                    "What PoE power budget is stated for the S3006-4ET-60 ports?",
                    "Ports 1-4 are listed at up to 30 W each, with a total PoE power budget of up to 60 W.",
                    ["PoE Power", "total"],
                ),
                (
                    1,
                    "What PoE pin assignment is stated for the S3006-4ET-60?",
                    "The datasheet states 1,2,4,5 as V+ and 3,6,7,8 as V-.",
                    ["PoE Pin Assignment"],
                ),
                (
                    1,
                    "Does the S3006-4ET-60 datasheet indicate support for long-distance PoE transmission?",
                    "Yes. It lists Long Distance PoE Transmission as supported.",
                    ["Long Distance PoE Transmission"],
                ),
                (
                    2,
                    "Where should ordering-panel questions for the S3006-4ET-60 datasheet look?",
                    "They should look at the third page, where the datasheet includes panels and revision/copyright information.",
                    ["Panels", "Rev"],
                ),
            ],
        ),
        (
            "hac_hf3805g_datasheet",
            [
                (
                    0,
                    "What sensor and resolution positioning does the DH-HAC-HF3805G datasheet emphasize?",
                    "It emphasizes a high-performance ISP, an advanced 4/3-inch 8MP image sensor, and 4K-resolution image detail.",
                    ["4/3", "8MP", "4K"],
                ),
                (
                    1,
                    "What effective pixel count is listed for the DH-HAC-HF3805G?",
                    "The datasheet lists 3840(H) × 2160(V), 8MP effective pixels.",
                    ["Effective Pixels"],
                ),
                (
                    1,
                    "What scanning system is specified for the DH-HAC-HF3805G?",
                    "The scanning system is Progressive.",
                    ["Scanning System"],
                ),
                (
                    1,
                    "What electronic shutter-speed range is listed for the DH-HAC-HF3805G?",
                    "The electronic shutter speed is listed as 1/3 s to 1/100,000 s.",
                    ["Electronic Shutter Speed"],
                ),
                (
                    2,
                    "Which ordering information entry identifies the PAL version of the 8MP boxed camera?",
                    "The PAL version is DH-HAC-HF3805GP, described as an 8Megapixel Ultra WDR Boxed Camera, PAL.",
                    ["DH-HAC-HF3805GP", "PAL"],
                ),
                (
                    2,
                    "Where does the DH-HAC-HF3805G datasheet place ordering information?",
                    "Ordering information is on the third page of the datasheet.",
                    ["Ordering Information"],
                ),
            ],
        ),
    ]
    for pdf_id, entries in specs:
        for page_idx, question, answer, keywords in entries:
            card = card_for(cards_by_pdf, pdf_id, page_idx)
            support = best_support(card, keywords=keywords)
            if question == "Where should ordering-panel questions for the S3006-4ET-60 datasheet look?":
                support = (
                    "Panels Desktop PoE Switch | DH-S3006-4ET-60 Rev 002.000 © 2025 Dahua. "
                    "All rights reserved. Design and specifications are subject to change without notice."
                )
            elif question == "What sensor and resolution positioning does the DH-HAC-HF3805G datasheet emphasize?":
                support = (
                    "System Overview DH-HAC-HFW3805G adopts high performance ISP and advanced 4/3\" 8MP image sensor, "
                    "which can bring vaster coverage and superiorer image details in 7/24 with the 4K resolution."
                )
            qas.append(
                make_question(
                    question=question,
                    answer=answer,
                    evidence=[
                        evidence_from_card(
                            card,
                            support=support,
                            modality="table",
                        )
                    ],
                    question_type="table_or_spec",
                    difficulty="medium",
                    requires_visual=False,
                )
            )
    return qas


def make_cross_pdf_questions(cards_by_pdf: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    qas: list[dict[str, Any]] = []
    pairs = [
        (
            "dss_pc_manual",
            0,
            "dss_web_manual",
            0,
            "What DSS version is shared by both the PC Client User Manual and the Web Client User Manual?",
            "Both manuals are for DSS Ultimate V8.8.0.",
            "multi_pdf_comparison",
            False,
        ),
        (
            "dss_pc_manual",
            1,
            "dss_web_manual",
            1,
            "What general product scope do the DSS PC and Web client manuals share?",
            "Both introduce functions and operations of DSS Ultimate, referred to as the system or platform.",
            "multi_pdf_comparison",
            False,
        ),
        (
            "dss_pc_manual",
            2,
            "dss_web_manual",
            2,
            "What privacy categories are mentioned in both DSS PC and Web client manuals?",
            "Both mention personal data such as face, audio, fingerprints, and license plate number.",
            "multi_pdf_comparison",
            False,
        ),
        (
            "dss_quick_deployment",
            0,
            "dss_web_manual",
            0,
            "Which DSS documents in the source set are explicitly tied to V8.8.0?",
            "The DSS Ultimate Quick Deployment Manual, Web Client User Manual, and PC Client User Manual are all tied to V8.8.0.",
            "multi_pdf_comparison",
            False,
        ),
        (
            "switch_user_manual",
            1,
            "s3006_datasheet",
            0,
            "How do the S3006 user manual and datasheet differ in document purpose?",
            "The user manual introduces installation, functions, and operations, while the datasheet summarizes product positioning and specifications.",
            "multi_pdf_comparison",
            False,
        ),
        (
            "hdcvi_camera_manual",
            1,
            "hac_hf3805g_datasheet",
            0,
            "How do the HDCVI camera user manual and DH-HAC-HF3805G datasheet differ in focus?",
            "The user manual explains camera functions and operations, while the datasheet emphasizes the specific DH-HAC-HF3805G hardware features and specifications.",
            "multi_pdf_comparison",
            False,
        ),
        (
            "hac_hf3231e_installation",
            0,
            "hac_hf3805g_dimension",
            0,
            "Which HAC-HF3805G-related sources should be used for mounting compatibility versus physical dimensions?",
            "Use the HAC-HF3231E installation sheet for wall/ceiling mount compatibility and the HAC-HF3805G dimension drawing for physical dimensions.",
            "multi_pdf_comparison",
            True,
        ),
        (
            "s3006_dimensions",
            0,
            "s3006_port_diagram",
            0,
            "Which S3006 visual source should answer size questions and which should answer port-layout questions?",
            "The S3006 dimensions PDF should answer size questions, while the S3006 port diagram should answer port-layout questions.",
            "multi_pdf_comparison",
            True,
        ),
        (
            "s3006_installation_method",
            0,
            "switch_user_manual",
            1,
            "Which S3006 source is better for installation sequence diagrams and which is better for prose installation and operation context?",
            "The installation-method sheet is better for visual step sequence diagrams, while the switch user manual is better for prose installation and operation context.",
            "multi_pdf_comparison",
            True,
        ),
        (
            "hdcvi_box_install_guide",
            0,
            "hdcvi_camera_manual",
            1,
            "Which HDCVI source should answer diagrammatic installation-step questions and which should answer general camera operation questions?",
            "The HDCVI Box Camera Installation Guide should answer diagrammatic installation-step questions, while the HDCVI Camera User Manual should answer general operation questions.",
            "multi_pdf_comparison",
            True,
        ),
        (
            "hac_open_source_notice",
            0,
            "hdcvi_camera_manual",
            1,
            "Which source handles legal open-source notices and which handles general camera safety and operation guidance?",
            "The open-source software notice handles legal software notices, while the HDCVI Camera User Manual handles safety and operation guidance.",
            "multi_pdf_comparison",
            False,
        ),
        (
            "dss_quick_deployment",
            1,
            "dss_pc_manual",
            1,
            "What safety-warning convention appears across DSS manuals?",
            "The DSS manuals use signal words to indicate safety instruction severity and meaning.",
            "multi_pdf_comparison",
            False,
        ),
        (
            "s3006_datasheet",
            1,
            "switch_user_manual",
            1,
            "Which S3006 source is best for exact PoE protocol values and which gives broader switch usage guidance?",
            "The datasheet is best for exact PoE protocol values such as IEEE 802.3af/at, while the user manual gives broader switch usage guidance.",
            "multi_pdf_comparison",
            False,
        ),
        (
            "hac_hf3805g_datasheet",
            1,
            "hdcvi_camera_manual",
            1,
            "Which HDCVI source should be cited for exact sensor/effective-pixel specifications and which for general manual guidance?",
            "The DH-HAC-HF3805G datasheet should be cited for exact sensor and effective-pixel specifications, while the camera user manual should be cited for general guidance.",
            "multi_pdf_comparison",
            False,
        ),
        (
            "dss_web_manual",
            0,
            "dss_quick_deployment",
            0,
            "Which DSS source is for web-client operation and which is for deployment?",
            "The Web Client User Manual is for web-client operation, while the Quick Deployment Manual is for deployment.",
            "multi_pdf_comparison",
            False,
        ),
    ]
    for left_id, left_page, right_id, right_page, question, answer, qtype, requires_visual in pairs:
        left = card_for(cards_by_pdf, left_id, left_page)
        right = card_for(cards_by_pdf, right_id, right_page)
        if requires_visual:
            left_support = visual_support(left, question, answer)
            right_support = visual_support(right, question, answer)
        else:
            left_support = cross_pdf_text_support(left, question, answer)
            right_support = cross_pdf_text_support(right, question, answer)
        qas.append(
            make_question(
                question=question,
                answer=answer,
                evidence=[
                    evidence_from_card(
                        left,
                        support=left_support,
                        modality=comparison_modality(left, requires_visual=requires_visual),
                    ),
                    evidence_from_card(
                        right,
                        support=right_support,
                        modality=comparison_modality(right, requires_visual=requires_visual),
                    ),
                ],
                question_type=qtype,
                difficulty="medium",
                requires_visual=requires_visual,
            )
        )
    return qas


def make_supplemental_questions(cards_by_pdf: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    qas: list[dict[str, Any]] = []
    specs = [
        ("dss_pc_manual", "login", "live_view", "PC client account access and live-view use"),
        ("dss_pc_manual", "alarm", "storage", "PC client alarm handling and recording context"),
        ("dss_web_manual", "login", "alarm", "Web client login and alarm workflow context"),
        ("dss_web_manual", "map", "access_control", "Web client map and access-control context"),
        ("dss_quick_deployment", "deployment", "network", "quick deployment and network preparation"),
        ("hdcvi_camera_manual", "safety", "camera_menu", "camera safety and menu configuration"),
        ("switch_user_manual", "safety", "network", "switch safety and network/PoE usage"),
        ("hdcvi_camera_manual", "camera_menu", "spec", "camera menu and specification-related setup"),
    ]
    for pdf_id, left_group, right_group, topic in specs:
        grouped = grouped_pages(cards_by_pdf[pdf_id])
        left_candidates = representative_cards(grouped.get(left_group, []), 3, min_text=120)
        right_candidates = representative_cards(grouped.get(right_group, []), 3, min_text=120)
        if not left_candidates or not right_candidates:
            continue
        left = left_candidates[0]
        right = next((card for card in right_candidates if card["page_idx"] != left["page_idx"]), right_candidates[0])
        if left["page_idx"] == right["page_idx"]:
            continue
        left_support = best_support(left, keywords=KEYWORD_GROUPS.get(left_group, []))
        right_support = best_support(right, keywords=KEYWORD_GROUPS.get(right_group, []))
        qas.append(
            make_question(
                question=(
                    f"For {topic}, what evidence should be combined from pages "
                    f"{left['page_number']} and {right['page_number']} of {left['file_name']}?"
                ),
                answer=f"Page {left['page_number']}: {left_support} Page {right['page_number']}: {right_support}",
                evidence=[
                    evidence_from_card(left, support=left_support),
                    evidence_from_card(right, support=right_support),
                ],
                question_type="cross_page",
                difficulty="hard",
                requires_visual=False,
            )
        )
    return qas


def make_doc_questions(info: PdfInfo, cards: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
    qas: list[dict[str, Any]] = []
    groups = grouped_pages(cards)
    used_pages: set[int] = set()
    group_order = [
        "login",
        "live_view",
        "alarm",
        "map",
        "access_control",
        "video_wall",
        "license_plate",
        "deployment",
        "safety",
        "network",
        "storage",
        "camera_menu",
        "spec",
    ]
    type_cycle = ["fact", "procedure", "cross_page", "fact", "procedure", "table_or_spec"]

    def add_from_card(card: dict[str, Any], group: str, qtype: str) -> None:
        support = best_support(card, keywords=KEYWORD_GROUPS.get(group, []))
        page_number = card["page_number"]
        file_name = info.path.name
        if qtype == "procedure":
            question = f"In {file_name}, what operational instruction or note is documented on page {page_number}?"
            answer = support
            difficulty = "medium"
        elif qtype == "table_or_spec":
            question = f"What structured detail or configuration point is documented on page {page_number} of {file_name}?"
            answer = support
            difficulty = "medium"
        elif qtype == "cross_page":
            next_card = None
            for candidate in cards:
                if candidate["page_idx"] > card["page_idx"] and is_content_card(candidate, min_text=120):
                    next_card = candidate
                    break
            if next_card is None:
                return
            next_support = best_support(next_card)
            question = (
                f"Which two evidence snippets should be combined from pages {page_number} "
                f"and {next_card['page_number']} of {file_name} for a cross-page retrieval check?"
            )
            answer = f"Page {page_number}: {support} Page {next_card['page_number']}: {next_support}"
            evidence = [
                evidence_from_card(card, support=support),
                evidence_from_card(next_card, support=next_support),
            ]
            qas.append(
                make_question(
                    question=question,
                    answer=answer,
                    evidence=evidence,
                    question_type="cross_page",
                    difficulty="hard",
                    requires_visual=False,
                )
            )
            used_pages.add(card["page_idx"])
            used_pages.add(next_card["page_idx"])
            return
        else:
            question = f"According to page {page_number} of {file_name}, what key point is stated?"
            answer = support
            difficulty = "easy"
        qas.append(
            make_question(
                question=question,
                answer=answer,
                evidence=[evidence_from_card(card, support=support)],
                question_type=qtype,
                difficulty=difficulty,
                requires_visual=False,
            )
        )
        used_pages.add(card["page_idx"])

    for group in group_order:
        if len(qas) >= quota:
            break
        candidates = representative_cards(groups.get(group, []), max(1, quota // 10), min_text=160)
        for idx, card in enumerate(candidates):
            if len(qas) >= quota:
                break
            if card["page_idx"] in used_pages and len(cards) > quota:
                continue
            add_from_card(card, group, type_cycle[(len(qas) + idx) % len(type_cycle)])

    if len(qas) < quota:
        candidates = representative_cards(cards, quota * 2, min_text=180)
        for idx, card in enumerate(candidates):
            if len(qas) >= quota:
                break
            if card["page_idx"] in used_pages and len(candidates) > quota:
                continue
            qtype = type_cycle[(len(qas) + idx) % len(type_cycle)]
            add_from_card(card, "general", qtype)

    return qas[:quota]


def build_qa(inventories: list[dict[str, Any]], cards_by_pdf: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    del inventories
    qas: list[dict[str, Any]] = []
    qas.extend(make_visual_questions(cards_by_pdf))
    qas.extend(make_small_doc_questions(cards_by_pdf))
    qas.extend(make_spec_questions(cards_by_pdf))
    qas.extend(make_cross_pdf_questions(cards_by_pdf))

    existing_counts = Counter()
    for qa in qas:
        for pdf in qa["source_pdfs"]:
            existing_counts[pdf] += 1

    for info in PDFS:
        already = existing_counts[str(info.path)]
        remaining = max(0, info.target_questions - already)
        if remaining <= 0:
            continue
        if info.visual_priority:
            continue
        qas.extend(make_doc_questions(info, cards_by_pdf[info.pdf_id], remaining))

    if len(qas) < 200:
        qas.extend(make_supplemental_questions(cards_by_pdf)[: 200 - len(qas)])

    for idx, qa in enumerate(qas, start=1):
        qa["id"] = f"qa-{idx:04d}"
    return qas


def validate(qas: list[dict[str, Any]], inventories: list[dict[str, Any]]) -> dict[str, Any]:
    path_pages = {item["path"]: item["page_count"] for item in inventories}
    errors = []
    ids = [qa.get("id") for qa in qas]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate QA ids.")
    if len(qas) != 200:
        errors.append(f"Expected 200 QA items, found {len(qas)}.")
    coverage = Counter()
    question_types = Counter()
    visual_count = 0
    multi_page_count = 0
    multi_pdf_count = 0
    for qa in qas:
        if not qa.get("question") or not qa.get("answer"):
            errors.append(f"{qa.get('id')} is missing question or answer.")
        source_pdfs = set(qa.get("source_pdfs", []))
        evidence = qa.get("evidence", [])
        evidence_pdfs = {item.get("pdf") for item in evidence}
        if source_pdfs != evidence_pdfs:
            errors.append(f"{qa.get('id')} source_pdfs do not match evidence pdfs.")
        page_keys = set()
        has_visual_evidence = False
        for item in evidence:
            pdf = item.get("pdf")
            page_idx = item.get("page_idx")
            support = str(item.get("support", ""))
            if re.search(r"\.{6,}", support) or "Table of Contents" in support:
                errors.append(f"{qa.get('id')} has table-of-contents style support.")
            if pdf not in path_pages:
                errors.append(f"{qa.get('id')} has unknown pdf {pdf}.")
                continue
            if not isinstance(page_idx, int) or page_idx < 0 or page_idx >= path_pages[pdf]:
                errors.append(f"{qa.get('id')} has out-of-range page {page_idx} for {pdf}.")
            page_keys.add((pdf, page_idx))
            if is_visual_modality(str(item.get("modality", ""))):
                has_visual_evidence = True
        if qa.get("requires_multiple_pages") != (len(page_keys) > 1):
            errors.append(f"{qa.get('id')} has inconsistent requires_multiple_pages.")
        if qa.get("requires_multiple_pdfs") != (len(source_pdfs) > 1):
            errors.append(f"{qa.get('id')} has inconsistent requires_multiple_pdfs.")
        if qa.get("requires_visual") and not has_visual_evidence:
            errors.append(f"{qa.get('id')} requires_visual but has no visual evidence.")
        for pdf in source_pdfs:
            coverage[pdf] += 1
        question_types[qa.get("question_type", "unknown")] += 1
        visual_count += int(bool(qa.get("requires_visual")))
        multi_page_count += int(bool(qa.get("requires_multiple_pages")))
        multi_pdf_count += int(bool(qa.get("requires_multiple_pdfs")))
    target_by_path = {str(info.path): info.target_questions for info in PDFS}
    for path, target in target_by_path.items():
        if coverage[path] < target:
            errors.append(f"{path} has {coverage[path]} QA items, target is {target}.")
    return {
        "total_questions": len(qas),
        "errors": errors,
        "coverage_by_pdf": dict(sorted(coverage.items())),
        "target_by_pdf": target_by_path,
        "question_type_counts": dict(sorted(question_types.items())),
        "requires_visual_count": visual_count,
        "requires_multiple_pages_count": multi_page_count,
        "requires_multiple_pdfs_count": multi_pdf_count,
        "question_type_targets": dict(QUESTION_TYPE_TARGETS),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the source-pdfs gold QA set.")
    parser.add_argument("--output-dir", default="qa-goldset")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    evidence_dir = output_dir / "evidence-cards"

    inventories = []
    cards_by_pdf = {}
    for info in PDFS:
        inventory, cards = scan_pdf(info)
        inventories.append(inventory)
        cards_by_pdf[info.pdf_id] = cards
        write_json(evidence_dir / f"{info.pdf_id}.json", cards)

    qas = build_qa(inventories, cards_by_pdf)
    report = validate(qas, inventories)
    write_json(output_dir / "source-pdfs-inventory.json", inventories)
    write_json(output_dir / "source-pdfs-qa-200.json", qas)
    write_json(output_dir / "qa-generation-report.json", report)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
