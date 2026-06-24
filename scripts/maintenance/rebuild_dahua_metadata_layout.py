from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / ".local" / "CUSTOM_DATA" / "pdfs" / "source"
OUTPUT_ROOT = REPO_ROOT / ".local" / "CUSTOM_DATA" / "pdfs" / "output"
REPORT_ROOT = REPO_ROOT / ".local" / "CUSTOM_DATA" / "reports" / "dahua_metadata_layout"

sys.path.insert(0, str(REPO_ROOT / "src"))
from rag_flow.source_paths import source_breadcrumb  # noqa: E402
from rag_flow.tagging import load_document_metadata  # noqa: E402


PRODUCT_FAMILIES = {
    "access_control",
    "accessory",
    "alarm",
    "attendance",
    "audio",
    "body_camera",
    "camera",
    "decoder",
    "desktop_computer",
    "display",
    "display_controller",
    "dmss",
    "dss",
    "ev_charger",
    "fire_alarm",
    "intercom",
    "mount",
    "networking",
    "recorder",
    "security",
    "software",
    "storage",
    "thermal",
    "traffic",
}

DOC_TYPES = {
    "datasheet",
    "installation_guide",
    "operation_manual",
    "other",
    "quick_start_guide",
    "security_guide",
    "selection_guide",
    "user_manual",
}

TOPIC_TAGS = {
    "access_control",
    "access_rule",
    "acupick",
    "add_device",
    "active_deterrence",
    "abnormality",
    "alarm_input",
    "alarm_output",
    "alarm_subscription",
    "ai_recognition",
    "anpr",
    "aperture",
    "attack_defense",
    "attendance",
    "attendance_period",
    "authentication",
    "aviation_connector",
    "auto_registration",
    "auto_tracking",
    "auto_maintenance",
    "audio",
    "audio_detection",
    "audio_encoding",
    "battery",
    "back_focus",
    "barrier_rod",
    "bit_rate",
    "bluetooth",
    "bracket",
    "buzzer",
    "call",
    "cellular_network",
    "camera_conditions",
    "card_enrollment",
    "card_reader",
    "carbon_monoxide_alarm",
    "cgi",
    "certificate_management",
    "cloud_update",
    "coaxial",
    "compatibility",
    "conference_endpoint",
    "configuration",
    "config_import_export",
    "configtool",
    "courseware",
    "cybersecurity",
    "backlight_compensation",
    "cvbs",
    "cvi",
    "day_night",
    "defocus_detection",
    "deep_ivs",
    "decoding_card",
    "demo_kit",
    "device_initialization",
    "device_management",
    "device_sharing",
    "digital_signage",
    "display",
    "dual_mode",
    "dip_switch",
    "digital_zoom",
    "disarming",
    "dmss",
    "dolynk",
    "door_station",
    "dst",
    "dss",
    "eas",
    "education",
    "electric_lock",
    "electronic_image_stabilization",
    "electronic_shelf_label",
    "eptz",
    "ev_charging",
    "event_alarm",
    "exit_button",
    "explosion_proof",
    "exposure",
    "face_detection",
    "face_recognition",
    "face_database",
    "flame_detection",
    "fingerprint",
    "fingerprint_enrollment",
    "file_management",
    "fire_alarm",
    "firmware_update",
    "factory_reset",
    "fisheye_dewarp",
    "ftp_storage",
    "general_operation",
    "gas_alarm",
    "google_edla",
    "defog",
    "hdd_installation",
    "hdcvi",
    "hdmi",
    "hdr",
    "high_refresh_rate",
    "housing",
    "heat_map",
    "hydrological_monitoring",
    "image_flicker",
    "image_quality",
    "image_mirror",
    "image_parameters",
    "indoor_monitor",
    "imou_account",
    "interactive_whiteboard",
    "infrared_illumination",
    "ir_cut_filter",
    "ir_reflection",
    "installation",
    "illuminator",
    "intrusion",
    "ivs",
    "firewall",
    "https",
    "idle_motion",
    "led_display",
    "license",
    "lens_distortion_correction",
    "live_view",
    "low_power_mode",
    "low_latency",
    "maintenance",
    "keyboard_controller",
    "language_setting",
    "lens",
    "local_storage",
    "mcu",
    "media_player",
    "mobile_app",
    "mobile_surveillance",
    "osd_menu",
    "motion_detection",
    "mounting",
    "mounting_accessory",
    "multi_screen_controller",
    "multicast",
    "mini_led",
    "nas_storage",
    "network_latency",
    "network_settings",
    "network_troubleshooting",
    "ntp",
    "oled_display",
    "onvif",
    "ops_pc",
    "optical_image_stabilization",
    "overlay",
    "plugin_installation",
    "p2p",
    "parking",
    "password_authentication",
    "password_reset",
    "passenger_counting",
    "people_counting",
    "perimeter_protection",
    "peripheral",
    "person_management",
    "plate_recognition",
    "playback",
    "poe",
    "polarlight",
    "power_supply",
    "presentation",
    "p_iris",
    "ppe_detection",
    "privacy_masking",
    "profile_management",
    "ptz",
    "ptz_default",
    "ptz_limit",
    "ptz_motion",
    "ptz_pan",
    "ptz_pattern",
    "ptz_powerup",
    "ptz_preset",
    "ptz_protocol",
    "ptz_restart",
    "ptz_scan",
    "ptz_speed",
    "ptz_tour",
    "pir_detection",
    "solar_power",
    "qr_code",
    "radio_equipment",
    "record_control",
    "recording_schedule",
    "remote_configuration",
    "remote_log",
    "regulatory_compliance",
    "reporting",
    "relay",
    "rfid",
    "resolution",
    "rtmp",
    "rtsp",
    "rs485",
    "sd_card",
    "security_recommendations",
    "server_requirements",
    "sip",
    "slip_ring",
    "smoke_alarm",
    "smoke_detection",
    "smb",
    "smtp_email",
    "sms_wake_up",
    "smart_dual_light",
    "smart_pss",
    "smart_motion_detection",
    "soundbar",
    "snapshot",
    "scene_change_detection",
    "screen_sharing",
    "splicing",
    "storage",
    "streaming",
    "system_log",
    "system_info",
    "system_service",
    "temperature_measurement",
    "thermal_alarm",
    "text_recognition",
    "time_schedule",
    "time_zone",
    "traffic",
    "troubleshooting",
    "touch_screen",
    "white_balance",
    "tripwire",
    "unlock",
    "utc_control",
    "usb_storage",
    "usb_camera",
    "user_management",
    "video_conferencing",
    "video_decoding",
    "video_encoding",
    "video_intercom",
    "video_loss",
    "video_metadata",
    "video_monitoring",
    "video_tampering",
    "video_wall",
    "video_standard",
    "voltage_detection",
    "water_level_monitoring",
    "web_interface",
    "whiteboard",
    "waterproof",
    "wiper",
    "wireless",
    "wiring",
    "access_reader",
    "adas",
    "bandwidth_management",
    "barrier",
    "blind_spot_detection",
    "body_camera",
    "bollard",
    "dash_camera",
    "desktop_computer",
    "digital_evidence",
    "display_controller",
    "driver_monitoring",
    "gps",
    "humidity_measurement",
    "intercom",
    "keypad",
    "magnetic_lock",
    "nfc",
    "ocpp",
    "palm_vein",
    "pppoe",
    "qos",
    "radar",
    "routing",
    "security_screening",
    "sfp",
    "starlight",
    "turnstile",
    "ups",
    "vlan",
    "water_leak_detection",
    "wiegand",
    "wizmind",
    "wizseek",
    "zoom_focus",
}

FAMILY_PATH = {
    "access_control": "access-control",
    "body_camera": "body-camera",
    "ev_charger": "ev-charger",
    "fire_alarm": "fire-alarm",
    "dmss": "mobile-dmss",
    "dss": "software-dss",
}
PATH_FAMILY = {value: key for key, value in FAMILY_PATH.items()}
for family in PRODUCT_FAMILIES:
    PATH_FAMILY.setdefault(family.replace("_", "-"), family)

DOC_TYPE_PATH = {
    "datasheet": "datasheets",
    "installation_guide": "installation-guides",
    "operation_manual": "operation-manuals",
    "other": "other",
    "quick_start_guide": "quick-start-guides",
    "security_guide": "security-guides",
    "selection_guide": "selection-guides",
    "user_manual": "user-manuals",
}

MODEL_PREFIXES = (
    "IPC",
    "HAC",
    "NVR",
    "XVR",
    "DVR",
    "IVSS",
    "EVS",
    "ESS",
    "VTO",
    "VTH",
    "VTS",
    "VTN",
    "VTNS",
    "VTA",
    "KTP",
    "KTX",
    "SD",
    "PTZ",
    "TPC",
    "ITC",
    "ITS",
    "IVD",
    "ASI",
    "ASA",
    "ASC",
    "DEE",
    "ASM",
    "ASF",
    "ARC",
    "ARA",
    "ARD",
    "ARM",
    "ARK",
    "HY",
    "ISC",
    "SSD",
    "HDD",
    "PFA",
    "PFB",
    "PFM",
    "PFS",
    "PFR",
    "NVD",
    "M70",
    "NKB",
    "LCH",
    "LUH",
    "LM",
)


@dataclass(frozen=True)
class DocumentPlan:
    current_relpath: str
    target_relpath: str
    current_pdf: Path
    target_pdf: Path
    metadata: dict[str, Any]
    output_dirs: tuple[Path, ...]
    chosen_output_dir: Path | None
    target_output_dir: Path | None


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _read_content_text(content_json: Path | None) -> str:
    if content_json is None or not content_json.exists():
        return ""
    try:
        data = json.loads(content_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    if not isinstance(data, list):
        return ""

    title_texts: list[str] = []
    heading_texts: list[str] = []
    body_texts: list[str] = []
    tail_texts: list[str] = []
    caption_texts: list[str] = []
    model_contexts: list[str] = []
    model_pattern = re.compile(r"\b(?:(?:DH|DHI)-)?[A-Z]{2,6}[A-Z0-9][A-Z0-9._()+/-]{2,45}\b")

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        raw = item.get("text") or item.get("caption") or item.get("image_caption") or ""
        if not raw:
            continue
        text = re.sub(r"\s+", " ", str(raw)).strip()
        if not text:
            continue
        text_level = item.get("text_level")
        item_type = str(item.get("type") or "")
        if text_level == 1 and len(title_texts) < 24:
            title_texts.append(text)
        elif isinstance(text_level, int) and text_level <= 3 and len(heading_texts) < 160:
            heading_texts.append(text)
        if item_type in {"image", "table", "equation"} and len(caption_texts) < 80:
            caption_texts.append(text)
        if len(body_texts) < 220:
            body_texts.append(text)
        if idx >= max(0, len(data) - 80):
            tail_texts.append(text)
        if model_pattern.search(text) and len(model_contexts) < 140:
            model_contexts.append(text)

    sections = [
        "TITLES:\n" + "\n".join(title_texts),
        "HEADINGS:\n" + "\n".join(heading_texts),
        "MODEL_CONTEXT:\n" + "\n".join(model_contexts),
        "CAPTIONS:\n" + "\n".join(caption_texts),
        "FRONT_TEXT:\n" + "\n".join(body_texts),
        "TAIL_TEXT:\n" + "\n".join(tail_texts[-80:]),
    ]
    # Keep classification evidence broad enough to reflect the document, while
    # bounding runtime and avoiding huge repeated OCR noise.
    return "\n\n".join(section[:120_000] for section in sections if section.strip())


def _content_json_for_docdir(docdir: Path) -> Path | None:
    for path in sorted(docdir.rglob("*_content_list.json")):
        if any(suffix in path.name for suffix in ("SECTIONED", "PATCHED", "CAPTIONED", "CHUNKED", "TAGGED")):
            continue
        return path
    return None


def _index_output_docdirs() -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for content_json in OUTPUT_ROOT.rglob("*_content_list.json"):
        if any(suffix in content_json.name for suffix in ("SECTIONED", "PATCHED", "CAPTIONED", "CHUNKED", "TAGGED")):
            continue
        docdir = content_json.parent.parent
        index[docdir.name.lower()].append(docdir)
    return index


def _output_dirs_for_pdf(pdf: Path, output_index: dict[str, list[Path]]) -> tuple[Path, ...]:
    stem = pdf.stem.lower()
    hits = list(output_index.get(stem, []))
    if hits:
        return tuple(sorted(hits))

    # Some earlier de-duplication appended a hash to the PDF basename while
    # leaving the original MinerU directory intact.
    without_hash = re.sub(r"__[0-9a-f]{10}$", "", stem)
    hits = list(output_index.get(without_hash, []))
    if hits:
        return tuple(sorted(hits))

    quoted = stem.replace('"', "%22").replace("'", "%27")
    return tuple(sorted(output_index.get(quoted, [])))


def _choose_output_dir(pdf_relpath: str, hits: tuple[Path, ...]) -> Path | None:
    if not hits:
        return None
    rel_parent = str(PurePosixPath(pdf_relpath).parent)
    scored: list[tuple[int, str, Path]] = []
    for hit in hits:
        rel = hit.relative_to(OUTPUT_ROOT).as_posix()
        score = 0
        if rel_parent in rel:
            score -= 100
        if "2026-06-19" in pdf_relpath and "2026-06-19" in rel:
            score -= 80
        if "official-supplement" in rel:
            score -= 30
        if any(hit.rglob("*TAGGED.json")):
            score -= 20
        score += len(rel)
        scored.append((score, rel, hit))
    return sorted(scored)[0][2]


def _infer_doc_type(relpath: str, filename: str, content: str, current: str | None = None) -> str:
    signal = f"{relpath}\n{filename}\n{content[:900]}"
    n = _norm(signal)
    file_path_n = _norm(f"{relpath}\n{filename}")
    if "security guide" in file_path_n or "security baseline" in file_path_n or "hardening guide" in file_path_n:
        return "security_guide"
    if "selection guide" in file_path_n or "selection manual" in file_path_n:
        return "selection_guide"
    if "quick start" in file_path_n or "qsg" in file_path_n:
        return "quick_start_guide"
    if (
        "installation" in file_path_n
        or "installazione" in file_path_n
        or "install method" in file_path_n
        or "install guide" in file_path_n
        or "installation method" in file_path_n
        or "installation procedure" in file_path_n
    ):
        return "installation_guide"
    if "operation manual" in file_path_n or "operating manual" in file_path_n or "operation guide" in file_path_n:
        return "operation_manual"
    if "configuration guide" in file_path_n or "how to configure" in file_path_n:
        return "operation_manual"
    if "user manual" in file_path_n or "user s manual" in file_path_n or "user guide" in file_path_n:
        return "user_manual"
    if "datasheet" in file_path_n or "data sheet" in file_path_n:
        return "datasheet"
    if "security recommendation" in n or "security baseline" in n or "hardening guide" in n:
        return "security_guide"
    if current in DOC_TYPES:
        return current
    return "other"


def _infer_version(filename: str, content: str, current: Any) -> str | None:
    for pattern in (
        r"(?:datasheet|data[_ -]?sheet)[_-]?(\d{8})",
        r"[_ -](\d{8})(?:\.pdf)?$",
        r"(?:^|[_ -])V(\d+(?:\.\d+){1,4})(?:[_ )\-.]|$)",
        r"version\s+(\d+(?:\.\d+){1,4})",
        r"\bV(\d+(?:\.\d+){1,4})\b",
    ):
        match = re.search(pattern, filename, re.IGNORECASE) or re.search(pattern, content[:4000], re.IGNORECASE)
        if match:
            return match.group(1)
    if isinstance(current, str) and current.strip():
        return current.removeprefix("V").removeprefix("v")
    return None


def _extract_models(filename: str, content: str, current: Any) -> list[str]:
    filename_for_models = str(PurePosixPath(filename).with_suffix(""))
    text = f"{filename_for_models}\n{content[:50000]}"
    models: list[str] = []
    if isinstance(current, list):
        models.extend(str(item) for item in current)

    prefix_re = "|".join(re.escape(prefix) for prefix in MODEL_PREFIXES)
    pattern = re.compile(
        rf"\b(?:(?:DH|DHI)-)?(?:{prefix_re})[A-Z0-9][A-Z0-9._()+/-]{{1,45}}\b",
        re.IGNORECASE,
    )
    stopword_models = {
        "ARABIC",
        "ARMING",
        "ARMING/DISARMING",
        "DATASHEET",
        "DHCP",
        "DHHOVERSEAS",
        "DHVISIONTECH.COM",
        "ESSENTIAL",
        "HTTP",
        "HTTPS",
        "MANUAL",
        "POE",
        "USER",
        "WIFI",
    }

    def normalize_model(value: str) -> str | None:
        model = str(value).upper().strip("._-() ")
        model = re.sub(r"[_ ]+", "-", model)
        model = model.replace("--", "-")
        model = re.sub(r"\)?[-_]*(?:USER|MANUAL|DATASHEET|QUICK|GUIDE)(?:[-_./].*)?$", "", model).strip("._-() ")
        if not model or model in stopword_models or model in {"DHI", "DH", "PDF"}:
            return None
        if any(token in model for token in ("HOVERSEAS", "VISIONTECH", "ESSENTIAL", "ARABIC")):
            return None
        if re.search(r"[A-Z0-9]-?DHI-", model):
            return None
        if re.fullmatch(r"V\d+(?:\.\d+)+", model):
            return None
        if any(token in model for token in ("DATASHEET", "INSTALLATION", "MANUAL", "USER")):
            return None
        if not re.search(r"\d", model) and model not in {"DEEPHUB", "DSS PROFESSIONAL", "DSS EXPRESS", "DMSS"}:
            return None
        return model

    for match in pattern.finditer(text):
        raw = match.group(0).strip("._-() ")
        value = normalize_model(raw)
        if not value:
            continue
        if value.startswith("DH-"):
            models.append(value[3:])
        if value.startswith("DHI-"):
            models.append(value[4:])
        models.append(value)

    # DSS/DMSS manuals often have no hardware model token.
    upper = text.upper()
    if "DSS PROFESSIONAL" in upper:
        models.append("DSS Professional")
    if "DSS EXPRESS" in upper:
        models.append("DSS Express")
    if "DMSS" in upper:
        models.append("DMSS")
    cleaned: list[str] = []
    for item in models:
        normalized = normalize_model(item)
        if normalized:
            cleaned.append(normalized)
    return _dedupe(cleaned)[:12]


def _path_family_hint(relpath: str) -> str | None:
    parts = PurePosixPath(relpath).parts
    if len(parts) >= 2 and parts[0] == "Dahua Italy":
        return PATH_FAMILY.get(parts[1])
    return None


def _classify_family_subfamilies(relpath: str, filename: str, content: str, current: dict[str, Any]) -> tuple[list[str], list[str]]:
    title = content.split("HEADINGS:", 1)[0][:2200]
    path_name_text = f"{relpath}\n{filename}"
    strong_text = f"{path_name_text}\n{title}"
    upper = strong_text.upper()
    filename_upper = filename.upper()
    path_upper = path_name_text.upper()
    norm = _norm(strong_text)
    families: list[str] = []
    subfamilies: list[str] = []

    def add_family(value: str) -> None:
        if value in PRODUCT_FAMILIES:
            families.append(value)

    def add_sub(value: str) -> None:
        subfamilies.append(value)

    hint = _path_family_hint(relpath)
    if hint:
        add_family(hint)

    path_n = _norm(relpath)
    if "network cameras" in path_n or "ptz cameras" in path_n or "hdcvi cameras" in path_n:
        add_family("camera")
        add_sub("network_camera")
    if "accessories" in path_n or "accessory" in path_n:
        add_family("accessory")
    if "storage products" in path_n or "network recorders" in path_n or "hdcvi recorders" in path_n:
        add_family("recorder")
    if "intelligent traffic" in path_n or "traffic" in path_n:
        add_family("traffic")
        add_sub("traffic_camera")
    if "video intercom" in path_n or "intercom" in path_n:
        add_family("intercom")
    if "access control" in path_n:
        add_family("access_control")
    if "time attendance" in path_n:
        add_family("attendance")
    if "dmss" in path_n:
        add_family("dmss")
        add_sub("dmss_app")
    elif "software products" in path_n or "smartpss" in path_n:
        add_family("dss")
    if "alarm" in path_n:
        add_family("alarm")
    if "thermal cameras" in path_n:
        add_family("thermal")
        add_sub("thermal_camera")

    if filename_upper.startswith(("ASI", "ASA", "ASC", "ASR", "ASM", "ASF", "ASG", "DEE")) or re.match(
        r"^(?:DH|DHI)-(?:ASI|ASA|ASC|ASR|ASM|ASF|ASG|DEE)", filename_upper
    ):
        add_family("access_control")
        if filename_upper.startswith(("ASC", "DHI-ASC", "DH-ASC")):
            add_sub("access_controller")
        elif filename_upper.startswith(("ASA", "DHI-ASA", "DH-ASA")):
            add_sub("attendance_terminal")
            add_sub("time_attendance_terminal")
        elif filename_upper.startswith(("ASI", "DHI-ASI", "DH-ASI")):
            add_sub("face_recognition_access_controller")
        else:
            add_sub("access_standalone")
    if filename_upper.startswith(("CS", "CHS", "PFS")) or re.match(r"^(?:DH|DHI)-(?:CS|CHS|PFS)", filename_upper):
        add_family("networking")
        add_sub("poe_switch")
    if filename_upper.startswith(("SSD", "HDD", "DDR")) or re.match(r"^(?:DH|DHI)-(?:SSD|HDD|DDR)", filename_upper):
        add_family("storage")
        add_sub("storage_media")
    if filename_upper.startswith(("PFA", "PFB", "PFC", "PFL", "PFR")) or re.match(
        r"^(?:DH|DHI)-(?:PFA|PFB|PFC|PFL|PFR)", filename_upper
    ):
        add_family("mount")
        add_sub("camera_mount_accessory")
    if filename_upper.startswith(("PFM", "PFH")) or re.match(r"^(?:DH|DHI)-PF", filename_upper):
        add_family("accessory")
    if filename_upper.startswith(("LCH", "LUH", "LM", "LPH", "LS", "PHB", "PHE", "PHG")) or re.match(
        r"^(?:DH|DHI)-(?:LCH|LUH|LM|LPH|LS|PHB|PHE|PHG)", filename_upper
    ):
        add_family("display")
        add_sub("lcd_display")
    if filename_upper.startswith("HAP") or re.match(r"^(?:DH|DHI)-HAP", filename_upper):
        add_family("audio")
        add_sub("ip_speaker")
    if filename_upper.startswith("VCS") or re.match(r"^(?:DH|DHI)-VCS", filename_upper):
        add_family("display")

    if re.search(r"\bDSS\b", upper) or "DSS PROFESSIONAL" in upper:
        add_family("dss")
        if "EXPRESS" in upper:
            add_sub("dss_express")
        elif "ONEBOX" in upper or "ONE BOX" in upper:
            add_sub("dss_onebox")
        elif "ULTIMATE" in upper:
            add_sub("dss_ultimate")
        else:
            add_sub("dss_professional")
    if "DMSS" in upper:
        add_family("dmss")
        add_sub("dmss_app")
    if (
        re.search(r"\b(?:ITC|IVD)[A-Z0-9-]", upper)
        or re.search(r"\bITS[A-Z0-9-]{2,}", upper)
        or "ANPR" in path_upper
        or "PARKING" in path_upper
        or "TRAFFIC" in path_upper
    ):
        add_family("traffic")
        add_sub("traffic_camera")
        if "ANPR" in upper or "PLATE" in path_upper or "LICENSE" in path_upper:
            add_sub("anpr_camera")
        if "PARKING" in upper:
            add_sub("parking_detector")
    if re.search(r"\b(VTO|VTH|VTS|VTN|VTNS|VTA|KTP|KTX)[A-Z0-9-]", upper) or "VIDEO INTERCOM" in upper:
        add_family("intercom")
        if "VTO" in upper or "DOOR STATION" in upper:
            add_sub("door_station")
        if "VTH" in upper or "INDOOR MONITOR" in upper:
            add_sub("indoor_monitor")
        if "VTNS" in upper or "TWO WIRE" in upper or "2-WIRE" in upper:
            add_sub("two_wire_intercom")
    if re.search(r"\b(NVR|XVR|DVR|IVSS|EVS)[A-Z0-9-]", upper) or "VIDEO STORAGE" in upper:
        add_family("recorder")
        if "NVR" in upper:
            add_sub("nvr")
        if "XVR" in upper:
            add_sub("xvr")
        if "DVR" in upper:
            add_sub("dvr")
        if "IVSS" in upper:
            add_sub("ivss")
        if "WIZMIND" in upper:
            add_sub("wizmind_recorder")
        if "VIDEO STORAGE" in upper or "EVS" in upper:
            add_sub("video_storage")
    if re.search(r"\b(IPC|HAC|SD|PTZ|TPC)[A-Z0-9-]", upper) or "NETWORK CAMERA" in upper or "HDCVI CAMERA" in upper:
        thermal_signal = (
            re.search(r"\bTPC[A-Z0-9-]", upper)
            or "THERMAL CAMERAS" in path_upper
            or "THERMAL_CAMERA" in path_upper
            or "THERMAL CAMERA" in path_upper
            or "THERMAL" in filename.upper()
        )
        if thermal_signal:
            add_family("thermal")
            add_sub("thermal_camera")
        else:
            add_family("camera")
        add_sub("network_camera")
        if "BULLET" in upper or "HFW" in upper:
            add_sub("bullet_camera")
        if "DOME" in upper or "HDBW" in upper:
            add_sub("dome_camera")
        if "EYEBALL" in upper or "HDW" in upper:
            add_sub("eyeball_camera")
        if "PTZ" in upper or re.search(r"\bSD[A-Z0-9]", upper):
            add_sub("ptz_camera")
        if "TIOC" in upper or "PV" in upper:
            add_sub("tioc_camera")
        if "WIZSENSE" in upper:
            add_sub("wizsense_camera")
        if "PANORAMIC" in upper or "180" in upper or "360" in upper:
            add_sub("panoramic_camera")
        if "SOLAR" in upper or "BATTERY" in upper:
            add_sub("battery_bullet_camera")
    if (
        re.search(r"\b(?:ASI|ASC|DEE|ASF|ASM|ASGB)[A-Z0-9-]", upper)
        or re.search(r"\bAC[0-9][A-Z0-9-]*", upper)
        or "ACCESS CONTROL" in upper
    ):
        add_family("access_control")
        if "FACE" in upper:
            add_sub("face_recognition_access_controller")
        elif "ASC" in upper or "CONTROLLER" in upper:
            add_sub("access_controller")
        else:
            add_sub("access_standalone")
    if re.search(r"\b(ASA)[A-Z0-9-]", upper) or "ATTENDANCE" in upper:
        add_family("attendance")
        add_sub("attendance_terminal")
        add_sub("time_attendance_terminal")
    if re.search(r"\b(ARC|ARA|ARD|ARM|ARK)[0-9-]", upper) or "ALARM HUB" in upper or "WIRELESS DETECTOR" in upper:
        add_family("alarm")
        if "ARC" in upper or "HUB" in upper:
            add_sub("alarm_hub")
        if "ARD" in upper or "DETECTOR" in upper:
            add_sub("wireless_detector")
        if "ARA" in upper or "SIREN" in upper:
            add_sub("wireless_siren")
        if "ARM" in upper or "REPEATER" in upper:
            add_sub("alarm_repeater")
        if "ARK" in upper or "KEYPAD" in upper:
            add_sub("wireless_keypad")
    if re.search(r"\bHY[-A-Z0-9]*", upper) and ("FIRE" in upper or "SMOKE" in upper):
        add_family("fire_alarm")
        if "SMOKE" in upper:
            add_sub("smoke_detector")
        if "GATEWAY" in upper:
            add_sub("wireless_gateway")
    if (
        re.search(r"\b(PFS|CS|CHS)[A-Z0-9-]", upper)
        or "POE SWITCH" in upper
        or "ETHERNET SWITCH" in upper
        or "UNMANAGED SWITCH" in upper
    ):
        add_family("networking")
        add_sub("poe_switch")
    if re.search(r"\b(PFA|PFB|PFR)[A-Z0-9-]", upper) or "BRACKET" in path_upper or "MOUNT" in path_upper:
        add_family("mount")
        add_sub("camera_mount_accessory")
        if "WALL" in upper:
            add_sub("wall_mount")
    if re.search(r"\b(PFM)[A-Z0-9-]", upper) or "TESTER" in upper or "ACCESSORY" in path_upper:
        add_family("accessory")
    if re.search(r"\b(SSD|HDD)\w*", upper) or "HARD DISK" in upper:
        add_family("storage")
        if "HDD" in upper or "HARD DISK" in upper:
            add_sub("hdd")
        else:
            add_sub("storage_media")
    if re.search(r"\b(NVD|NKB|M70)[A-Z0-9-]", upper) or "DECODER" in upper:
        add_family("decoder")
        add_sub("network_video_decoder")
    if re.search(r"\b(LCH|LUH|LM|LPH)[A-Z0-9-]", upper) or "DISPLAY" in path_upper or "WHITEBOARD" in upper:
        add_family("display")
        if "WHITEBOARD" in upper:
            add_sub("interactive_whiteboard")
        else:
            add_sub("lcd_display")
    if "DEEPHUB" in upper or "INTERACTIVE WHITEBOARD" in upper:
        add_family("display")
        add_sub("interactive_whiteboard")
    if "ROUTER" in upper:
        add_family("networking")
        add_sub("wireless_router")
    if "SPEAKER" in upper or "AUDIO" in path_upper:
        add_family("audio")
        add_sub("ip_speaker")
    if re.search(r"\bEAS\b", upper) or re.search(r"\bISC[-A-Z0-9]*", upper):
        add_family("security")
        add_sub("eas_system")
    if "BODY CAMERA" in upper:
        add_family("body_camera")
        add_sub("body_camera")
    if "EV CHARG" in upper or "D-VOLT" in upper:
        add_family("ev_charger")
        add_sub("ev_charger")

    current_families = current.get("product_families")
    current_subfamilies = current.get("product_subfamilies")
    if not families and isinstance(current_families, list):
        inherited = [str(item) for item in current_families]
        if inherited != ["security"]:
            families.extend(inherited)
    if not subfamilies and isinstance(current_subfamilies, list):
        subfamilies.extend(str(item) for item in current_subfamilies)
    families = [item for item in _dedupe(families) if item in PRODUCT_FAMILIES]
    subfamilies = _dedupe(subfamilies)
    if "display" in families and ("DEEPHUB" in upper or "INTERACTIVE WHITEBOARD" in upper):
        families = ["display"] + [item for item in families if item == "accessory"]
        subfamilies = _dedupe(["interactive_whiteboard"] + [item for item in subfamilies if item not in {"access_standalone", "network_camera", "ptz_camera"}])
    if "networking" in families and "ROUTER" in upper:
        families = ["networking"] + [item for item in families if item in {"accessory"}]
        subfamilies = _dedupe(["wireless_router"] + [item for item in subfamilies if item not in {"access_standalone", "face_recognition_access_controller"}])
    if hint == "dss" and "SOFTWARE-DSS" in path_upper and "ROUTER" not in filename_upper:
        families = ["dss"]
        subfamilies = _dedupe(["dss_professional"] + [item for item in subfamilies if item.startswith("dss_")])
    if families == ["intercom"]:
        allowed = {"door_station", "indoor_monitor", "two_wire_intercom", "video_door_phone"}
        subfamilies = [item for item in subfamilies if item in allowed]
    if families == ["display"]:
        allowed = {"interactive_whiteboard", "lcd_display"}
        subfamilies = [item for item in subfamilies if item in allowed]
    if families == ["networking"]:
        allowed = {"poe_switch", "wireless_router", "network_switch", "wireless_access_point"}
        subfamilies = [item for item in subfamilies if item in allowed]
    if not families:
        families = ["accessory"]
    return families, subfamilies


def _infer_topics(filename: str, content: str, families: list[str], subfamilies: list[str], doc_type: str, current: Any) -> list[str]:
    text = f"{filename}\n{content[:8000]}"
    upper = text.upper()
    norm = _norm(text)
    topics: list[str] = []

    def add(value: str) -> None:
        if value in TOPIC_TAGS:
            topics.append(value)

    if doc_type == "installation_guide":
        add("installation")
    if "wiring" in norm or "cable" in norm or "terminal" in norm:
        add("wiring")
    if "mount" in norm or "bracket" in norm:
        add("mounting")
        add("bracket")
        add("mounting_accessory")
    if "poe" in norm:
        add("poe")
    if "network" in norm or "ip address" in norm or "dhcp" in norm or "tcp" in norm:
        add("network_settings")
    if "password" in norm or "xml password" in norm:
        add("password_reset")
    if "firmware" in norm or "upgrade" in norm or "update" in norm:
        add("firmware_update")
    if "cloud update" in norm:
        add("cloud_update")
    if "smtp" in norm or "email" in norm:
        add("smtp_email")
    if "onvif" in norm:
        add("onvif")
    if "cgi" in norm:
        add("cgi")
    if "p2p" in norm:
        add("p2p")
    if "qr code" in norm:
        add("qr_code")
    if "share" in norm:
        add("device_sharing")
    if "add device" in norm or "adding device" in norm:
        add("add_device")
    if "initialization" in norm or "initialize" in norm:
        add("device_initialization")
    if "user management" in norm or "user account" in norm:
        add("user_management")
    if "device management" in norm:
        add("device_management")
    if "web" in norm or "browser" in norm:
        add("web_interface")
    if "live view" in norm:
        add("live_view")
    if "playback" in norm:
        add("playback")
    if "record" in norm or "schedule" in norm:
        add("recording_schedule")
    if "storage" in norm or "hdd" in norm or "hard disk" in norm:
        add("storage")
    if "hdd install" in norm:
        add("hdd_installation")
    if "snapshot" in norm:
        add("snapshot")
    if "stream" in norm:
        add("streaming")
    if "motion" in norm:
        add("motion_detection")
    if "tripwire" in norm:
        add("tripwire")
    if "ivs" in norm:
        add("ivs")
    if "privacy mask" in norm:
        add("privacy_masking")
    if "ptz" in norm:
        add("ptz")
    if "alarm" in norm:
        add("event_alarm")
    if "alarm input" in norm:
        add("alarm_input")
    if "alarm output" in norm:
        add("alarm_output")
    if "intrusion" in norm:
        add("intrusion")
    if "smoke" in norm:
        add("smoke_alarm")
    if "anpr" in norm or "plate" in norm or "license" in norm:
        add("anpr")
        add("plate_recognition")
        add("license")
    if "parking" in norm:
        add("parking")
    if "face" in norm:
        add("face_recognition")
    if "call" in norm:
        add("call")
    if "unlock" in norm:
        add("unlock")
    if "sip" in norm:
        add("sip")
    if "dmss" in upper:
        add("mobile_app")
    if "dolynk" in norm:
        add("dolynk")
    if "acupick" in norm:
        add("acupick")
    if "wizseek" in norm:
        add("wizseek")
    if "time zone" in norm:
        add("time_zone")
    if "system log" in norm or "log" in norm:
        add("system_log")
    if "compatib" in norm:
        add("compatibility")
    if "server requirement" in norm:
        add("server_requirements")
    if "security" in norm and ("security" in families or "security_guide" == doc_type):
        add("security_recommendations")
    if "temperature" in norm:
        add("temperature_measurement")
    if "thermal" in norm:
        add("thermal_alarm")

    for family in families:
        add(family if family in TOPIC_TAGS else "general_operation")
    for subfamily in subfamilies:
        if subfamily in {"door_station", "indoor_monitor", "video_door_phone"}:
            add("video_intercom")
            add(subfamily)
        if subfamily in {"eas_system"}:
            add("eas")
        if subfamily in {"lcd_display", "interactive_whiteboard"}:
            add("display")
        if subfamily in {"ip_speaker"}:
            add("audio")

    topics = [item for item in _dedupe(topics) if item in TOPIC_TAGS]
    if families == ["networking"]:
        incompatible = {"access_control", "anpr", "plate_recognition", "license", "face_recognition", "call", "playback"}
        topics = [item for item in topics if item not in incompatible]
        for value in ("network_settings", "web_interface"):
            if value not in topics:
                topics.append(value)
    if not topics:
        topics = ["general_operation"]
    return topics[:14]


def _infer_language(filename: str, content: str, current: Any) -> str | None:
    text = f"{filename}\n{content[:3000]}"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[\u0600-\u06ff]", text) or re.search(r"\barabic\b", text, re.IGNORECASE):
        return "ar"
    if re.search(r"\b(francais|français|guide d'installation|manuel utilisateur)\b", text, re.IGNORECASE):
        return "fr"
    if re.search(r"\b(polish|polski|instrukcja|użytkownik)\b", text, re.IGNORECASE):
        return "pl"
    if re.search(r"\b(italiano|installazione|manuale utente)\b", text, re.IGNORECASE):
        return "it"
    if isinstance(current, str) and current.strip():
        return current.strip()
    return "en"


def _build_metadata(relpath: str, pdf: Path, content_json: Path | None, current: dict[str, Any]) -> dict[str, Any]:
    content = _read_content_text(content_json)
    doc_type = _infer_doc_type(relpath, pdf.name, content, current.get("doc_type"))
    families, subfamilies = _classify_family_subfamilies(relpath, pdf.name, content, current)
    topics = _infer_topics(pdf.name, content, families, subfamilies, doc_type, current.get("topic_tags"))
    return {
        "filename": pdf.name,
        "product_families": families,
        "product_subfamilies": subfamilies,
        "doc_type": doc_type,
        "version": _infer_version(pdf.name, content, current.get("version")),
        "models": _extract_models(pdf.name, content, current.get("models")),
        "language": _infer_language(pdf.name, content, current.get("language")),
        "topic_tags": topics,
    }


def _target_relpath(metadata: dict[str, Any]) -> str:
    family = metadata["product_families"][0] if metadata.get("product_families") else "accessory"
    family_dir = FAMILY_PATH.get(family, family.replace("_", "-"))
    doc_dir = DOC_TYPE_PATH.get(metadata.get("doc_type") or "other", "other")
    return f"Dahua Italy/{family_dir}/{doc_dir}/{metadata['filename']}"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    text = str(value)
    return json.dumps(text, ensure_ascii=False)


def _write_root_metadata_schema(path: Path) -> None:
    lines = [
        "# Dahua Italy RAG metadata root.",
        "# This file is schema/control-vocabulary documentation only.",
        "# Per-document tagging must live next to each PDF as <pdf-stem>_metadata.yml.",
        "metadata_schema_version: 2",
        "fields:",
        "  filename: string",
        "  product_families: list[string]",
        "  product_subfamilies: list[string]",
        "  doc_type: string",
        "  version: string|null",
        "  models: list[string]",
        "  language: string|null",
        "  topic_tags: list[string]",
        "product_family_values:",
    ]
    for value in sorted(PRODUCT_FAMILIES):
        lines.append(f"  - {_yaml_scalar(value)}")
    lines.append("doc_type_values:")
    for value in sorted(DOC_TYPES):
        lines.append(f"  - {_yaml_scalar(value)}")
    lines.append("common_topic_tags:")
    for value in sorted(TOPIC_TAGS):
        lines.append(f"  - {_yaml_scalar(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_metadata_yaml(path: Path, plans: list[DocumentPlan]) -> None:
    _write_root_metadata_schema(path)


def _metadata_sidecar_path(pdf_path: Path) -> Path:
    return pdf_path.with_name(f"{pdf_path.stem}_metadata.yml")


def _write_metadata_sidecar(path: Path, *, source_relpath: str, metadata: dict[str, Any]) -> None:
    lines = [
        "# Dahua Italy per-document RAG metadata.",
        "# Generated from the parsed PDF content and filename; keep in sync with source/metadata.yml.",
        "metadata_schema_version: 2",
        f"source_relpath: {_yaml_scalar(source_relpath)}",
        f"filename: {_yaml_scalar(metadata['filename'])}",
    ]
    for field in ("product_families", "product_subfamilies"):
        values = metadata[field]
        if values:
            lines.append(f"{field}:")
            for value in values:
                lines.append(f"  - {_yaml_scalar(value)}")
        else:
            lines.append(f"{field}: []")
    lines.append(f"doc_type: {_yaml_scalar(metadata['doc_type'])}")
    lines.append(f"version: {_yaml_scalar(metadata['version'])}")
    if metadata["models"]:
        lines.append("models:")
        for value in metadata["models"]:
            lines.append(f"  - {_yaml_scalar(value)}")
    else:
        lines.append("models: []")
    lines.append(f"language: {_yaml_scalar(metadata['language'])}")
    if metadata["topic_tags"]:
        lines.append("topic_tags:")
        for value in metadata["topic_tags"]:
            lines.append(f"  - {_yaml_scalar(value)}")
    else:
        lines.append("topic_tags: []")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _replace_in_text(value: str, replacements: list[tuple[str, str]]) -> str:
    result = value
    for old, new in replacements:
        if old:
            result = result.replace(old, new)
    return result


def _rewrite_json_paths(path: Path, *, old_relpath: str, new_relpath: str, old_docdir: Path, new_docdir: Path) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    old_breadcrumb = source_breadcrumb(old_relpath)
    new_breadcrumb = source_breadcrumb(new_relpath)
    old_filename = PurePosixPath(old_relpath).name
    new_filename = PurePosixPath(new_relpath).name
    replacements = [
        (old_relpath, new_relpath),
        (old_breadcrumb, new_breadcrumb),
        (str(old_docdir), str(new_docdir)),
        (old_docdir.as_posix(), new_docdir.as_posix()),
    ]

    def visit(item: Any) -> Any:
        if isinstance(item, dict):
            changed: dict[str, Any] = {}
            for key, value in item.items():
                if key in {"source_relpath", "source"} and isinstance(value, str):
                    changed[key] = new_relpath if value == old_relpath else _replace_in_text(value, replacements)
                elif key == "source_filename" and isinstance(value, str):
                    changed[key] = new_filename if value == old_filename else value
                elif key == "breadcrumb" and isinstance(value, str):
                    changed[key] = _replace_in_text(value, replacements)
                elif key == "chunk_id" and isinstance(value, str):
                    changed[key] = _replace_in_text(value, replacements)
                elif isinstance(value, str):
                    changed[key] = _replace_in_text(value, replacements)
                else:
                    changed[key] = visit(value)
            return changed
        if isinstance(item, list):
            return [visit(value) for value in item]
        if isinstance(item, str):
            return _replace_in_text(item, replacements)
        return item

    new_data = visit(data)
    path.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_plans() -> tuple[list[DocumentPlan], dict[str, Any]]:
    output_index = _index_output_docdirs()
    plans: list[DocumentPlan] = []
    used_output_dirs: set[Path] = set()
    anomalies: dict[str, Any] = {
        "missing_output": [],
        "multiple_output": [],
        "target_collisions": [],
    }

    for pdf in sorted(SOURCE_ROOT.rglob("*.pdf"), key=lambda item: item.relative_to(SOURCE_ROOT).as_posix().lower()):
        current_relpath = pdf.relative_to(SOURCE_ROOT).as_posix()
        output_dirs = _output_dirs_for_pdf(pdf, output_index)
        chosen_output = _choose_output_dir(current_relpath, output_dirs)
        if chosen_output is not None:
            used_output_dirs.add(chosen_output)
        if not output_dirs:
            anomalies["missing_output"].append(current_relpath)
        if len(output_dirs) > 1:
            anomalies["multiple_output"].append({"source": current_relpath, "outputs": [p.relative_to(OUTPUT_ROOT).as_posix() for p in output_dirs]})
        content_json = _content_json_for_docdir(chosen_output) if chosen_output else None
        sidecar = _metadata_sidecar_path(pdf)
        current = load_document_metadata(sidecar).get(current_relpath, {}) if sidecar.exists() else {}
        doc_metadata = _build_metadata(current_relpath, pdf, content_json, current)
        target_relpath = _target_relpath(doc_metadata)
        target_pdf = SOURCE_ROOT / target_relpath
        target_output_dir = OUTPUT_ROOT / str(PurePosixPath(target_relpath).parent) / pdf.stem
        plans.append(
            DocumentPlan(
                current_relpath=current_relpath,
                target_relpath=target_relpath,
                current_pdf=pdf,
                target_pdf=target_pdf,
                metadata=doc_metadata,
                output_dirs=output_dirs,
                chosen_output_dir=chosen_output,
                target_output_dir=target_output_dir,
            )
        )

    by_target: dict[str, list[str]] = defaultdict(list)
    for plan in plans:
        by_target[plan.target_relpath].append(plan.current_relpath)
    anomalies["target_collisions"] = {target: sources for target, sources in by_target.items() if len(sources) > 1}
    all_output_dirs = {docdir for dirs in output_index.values() for docdir in dirs}
    anomalies["orphan_output_dirs"] = sorted(
        docdir.relative_to(OUTPUT_ROOT).as_posix()
        for docdir in all_output_dirs
        if docdir not in used_output_dirs
    )
    return plans, anomalies


def write_reports(plans: list[DocumentPlan], anomalies: dict[str, Any]) -> Path:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORT_ROOT / "manifest.json"
    summary_path = REPORT_ROOT / "summary.json"
    csv_path = REPORT_ROOT / "manifest.tsv"
    metadata_preview_path = REPORT_ROOT / "metadata.preview.yml"

    manifest = [
        {
            "current_relpath": plan.current_relpath,
            "target_relpath": plan.target_relpath,
            "target_sidecar": _metadata_sidecar_path(plan.target_pdf).relative_to(SOURCE_ROOT).as_posix(),
            "metadata": plan.metadata,
            "chosen_output_dir": plan.chosen_output_dir.relative_to(OUTPUT_ROOT).as_posix() if plan.chosen_output_dir else None,
            "target_output_dir": plan.target_output_dir.relative_to(OUTPUT_ROOT).as_posix() if plan.target_output_dir else None,
            "output_candidates": [item.relative_to(OUTPUT_ROOT).as_posix() for item in plan.output_dirs],
        }
        for plan in plans
    ]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = ["current_relpath\ttarget_relpath\tfamilies\tsubfamilies\tdoc_type\tmodels\ttopic_tags\tchosen_output"]
    for item in manifest:
        meta = item["metadata"]
        rows.append(
            "\t".join(
                [
                    item["current_relpath"],
                    item["target_relpath"],
                    ",".join(meta["product_families"]),
                    ",".join(meta["product_subfamilies"]),
                    meta["doc_type"],
                    ",".join(meta["models"]),
                    ",".join(meta["topic_tags"]),
                    item["chosen_output_dir"] or "",
                ]
            )
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    _write_metadata_yaml(metadata_preview_path, plans)
    summary = {
        "source_pdf_count": len(plans),
        "move_count": sum(plan.current_relpath != plan.target_relpath for plan in plans),
        "missing_output_count": len(anomalies["missing_output"]),
        "multiple_output_count": len(anomalies["multiple_output"]),
        "target_collision_count": len(anomalies["target_collisions"]),
        "orphan_output_dir_count": len(anomalies["orphan_output_dirs"]),
        "by_family": Counter(plan.metadata["product_families"][0] for plan in plans),
        "by_doc_type": Counter(plan.metadata["doc_type"] for plan in plans),
        "anomalies": anomalies,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def _safe_move(src: Path, dst: Path) -> None:
    if src == dst:
        return
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Target already exists: {dst}")
    shutil.move(str(src), str(dst))


def _prune_empty_dirs(root: Path) -> None:
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda item: len(item.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def apply_layout(plans: list[DocumentPlan], *, delete_orphan_outputs: bool) -> None:
    current_to_target = {plan.current_relpath: plan.target_relpath for plan in plans}
    if len(current_to_target) != len(plans):
        raise RuntimeError("Duplicate current source relpath in plans")
    target_counts = Counter(plan.target_relpath for plan in plans)
    collisions = [path for path, count in target_counts.items() if count > 1]
    if collisions:
        raise RuntimeError(f"Target path collisions: {collisions[:10]}")

    # Move outputs before source files so output metadata can still be rewritten
    # against the old source relpath.
    used_outputs_after_move: set[Path] = set()
    for plan in plans:
        if plan.chosen_output_dir is None or plan.target_output_dir is None:
            continue
        old_docdir = plan.chosen_output_dir
        new_docdir = plan.target_output_dir
        if old_docdir != new_docdir:
            if new_docdir.exists() and new_docdir != old_docdir:
                shutil.rmtree(new_docdir)
            _safe_move(old_docdir, new_docdir)
        else:
            new_docdir = old_docdir
        used_outputs_after_move.add(new_docdir)
        for json_path in new_docdir.rglob("*.json"):
            _rewrite_json_paths(
                json_path,
                old_relpath=plan.current_relpath,
                new_relpath=plan.target_relpath,
                old_docdir=old_docdir,
                new_docdir=new_docdir,
            )

    for plan in plans:
        current_sidecar = _metadata_sidecar_path(plan.current_pdf)
        target_sidecar = _metadata_sidecar_path(plan.target_pdf)
        if plan.current_pdf != plan.target_pdf:
            _safe_move(plan.current_pdf, plan.target_pdf)
            if current_sidecar.exists() and current_sidecar != target_sidecar:
                if target_sidecar.exists():
                    target_sidecar.unlink()
                _safe_move(current_sidecar, target_sidecar)

    _write_metadata_yaml(SOURCE_ROOT / "metadata.yml", plans)
    for plan in plans:
        _write_metadata_sidecar(
            _metadata_sidecar_path(plan.target_pdf),
            source_relpath=plan.target_relpath,
            metadata=plan.metadata,
        )

    if delete_orphan_outputs:
        output_index = _index_output_docdirs()
        all_output_dirs = {docdir for dirs in output_index.values() for docdir in dirs}
        orphan_output_dirs = all_output_dirs - used_outputs_after_move
        if all_output_dirs and len(orphan_output_dirs) >= max(10, int(len(all_output_dirs) * 0.9)):
            raise RuntimeError(
                "Refusing to delete almost all output directories as orphans. "
                f"orphans={len(orphan_output_dirs)} total={len(all_output_dirs)}"
            )
        for docdir in sorted(orphan_output_dirs, key=lambda p: len(p.parts), reverse=True):
            shutil.rmtree(docdir, ignore_errors=True)

    _prune_empty_dirs(SOURCE_ROOT)
    _prune_empty_dirs(OUTPUT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Dahua Italy PDF metadata and source/output layout.")
    parser.add_argument("--apply", action="store_true", help="Apply file moves and rewrite metadata.yml.")
    parser.add_argument(
        "--delete-orphan-outputs",
        action="store_true",
        help="When applying, remove MinerU output directories that do not map to a current source PDF.",
    )
    args = parser.parse_args()

    plans, anomalies = build_plans()
    summary_path = write_reports(plans, anomalies)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps({key: summary[key] for key in summary if key != "anomalies"}, ensure_ascii=False, indent=2))
    print(f"report: {summary_path}")
    if summary["target_collision_count"]:
        print("Refusing to apply because target collisions exist.", file=sys.stderr)
        raise SystemExit(2)
    if args.apply:
        apply_layout(plans, delete_orphan_outputs=args.delete_orphan_outputs)
        plans_after, anomalies_after = build_plans()
        summary_after = write_reports(plans_after, anomalies_after)
        print(f"applied. refreshed report: {summary_after}")


if __name__ == "__main__":
    main()
