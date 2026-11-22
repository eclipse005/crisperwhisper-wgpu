#!/usr/bin/env python3
"""Collect short, legally reusable speech clips for the official 10 languages.

Source: Internet Archive LibriVox collection (public domain).
Each language gets one 15-second 16 kHz mono WAV crop, plus SOURCES.md.
Unspaced languages (zh/ja/th) are collected from Wikimedia Commons Lingua Libre
(CC BY-SA) for segmentation-only checks, not timestamp quality claims.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "tests" / "audio" / "langs"
UA = "crisperwhisper-wgpu-research/0.1 (non-commercial evaluation)"

# Official 10-language board (all space-delimited).
SPACED = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "nl": "Dutch",
    "fr": "French",
    "it": "Italian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
}

# Segmentation-only (do not promise timestamp quality).
UNSPACED = {
    "zh": "Q7850",  # Mandarin Wikidata / Lingua Libre
    "ja": "Q5287",
    "th": "Q9217",
}

# Prefer these known identifiers when the search API is flaky.
PREFERRED = {
    "de": "ausgewahlte_erzahlungen_puschkin_1505_librivox",
}


def curl_json(url: str, timeout: int = 40):
    p = subprocess.run(
        ["curl", "-sS", "-L", "-m", str(timeout), "-A", UA, "-H", "Accept: application/json", url],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"curl {url}: {p.stderr[:300]}")
    return json.loads(p.stdout)


def curl_file(url: str, dest: Path, timeout: int = 180) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(
        ["curl", "-sS", "-L", "-m", str(timeout), "-A", UA, "-o", str(dest), url],
        capture_output=True, text=True,
    )
    if p.returncode != 0 or not dest.exists() or dest.stat().st_size < 1000:
        raise RuntimeError(f"download failed {url}: {p.stderr[:300]} size={dest.stat().st_size if dest.exists() else 0}")


def ia_search(language_name: str, rows: int = 8) -> list[dict]:
    q = f"collection:(librivoxaudio) AND languageSorter:{language_name}"
    url = (
        "https://archive.org/advancedsearch.php?q=" + urllib.parse.quote(q)
        + "&fl[]=identifier&fl[]=title&fl[]=language&fl[]=licenseurl"
        + "&sort[]=downloads+desc&rows=" + str(rows) + "&page=1&output=json"
    )
    data = curl_json(url)
    return data["response"]["docs"]


def ia_pick_audio(identifier: str) -> dict:
    meta = curl_json(f"https://archive.org/metadata/{identifier}", timeout=60)
    files = meta.get("files") or []
    # Prefer a mid-size mp3 section (not the 64kb whole-book, not tiny jingles).
    cands = []
    for f in files:
        name = f.get("name") or ""
        fmt = (f.get("format") or "").lower()
        if not (name.lower().endswith(".mp3") or "mp3" in fmt):
            continue
        if name.lower().endswith((".zip", ".jpg", ".png")):
            continue
        try:
            size = int(f.get("size") or 0)
        except ValueError:
            size = 0
        # 200 KB .. 15 MB is typically one chapter / a short poem
        if 200_000 <= size <= 15_000_000:
            cands.append((size, name, f))
    if not cands:
        # fallback: any mp3 under 30 MB
        for f in files:
            name = f.get("name") or ""
            if name.lower().endswith(".mp3"):
                try:
                    size = int(f.get("size") or 0)
                except ValueError:
                    size = 0
                if 50_000 <= size <= 30_000_000:
                    cands.append((size, name, f))
    if not cands:
        raise RuntimeError(f"no suitable mp3 in {identifier}")
    cands.sort()  # smallest first
    size, name, f = cands[0]
    return {
        "identifier": identifier,
        "title": (meta.get("metadata") or {}).get("title"),
        "licenseurl": (meta.get("metadata") or {}).get("licenseurl")
            or "https://creativecommons.org/publicdomain/mark/1.0/",
        "language": (meta.get("metadata") or {}).get("language"),
        "filename": name,
        "size": size,
        "url": f"https://archive.org/download/{identifier}/{urllib.parse.quote(name)}",
    }


def crop_wav(src: Path, dest: Path, start: float = 8.0, duration: float = 15.0) -> float:
    """ffmpeg crop + resample 16 kHz mono, skip a bit of intro silence."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start), "-t", str(duration), "-i", str(src),
        "-ac", "1", "-ar", "16000", str(dest),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not dest.exists():
        raise RuntimeError(f"ffmpeg failed: {p.stderr[-400:]}")
    # duration via ffprobe
    q = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(dest)],
        capture_output=True, text=True,
    )
    try:
        return float(q.stdout.strip())
    except ValueError:
        return duration


def collect_spaced(out_dir: Path) -> list[dict]:
    records = []
    for code, name in SPACED.items():
        wav = out_dir / f"{code}.wav"
        rec = {"code": code, "language": name, "license": "Public Domain (LibriVox / Internet Archive)"}
        try:
            ident = PREFERRED.get(code)
            docs = ia_search(name)
            if not ident:
                if not docs:
                    raise RuntimeError(f"no IA hits for {name}")
                ident = docs[0]["identifier"]
            picked = ia_pick_audio(ident)
            raw = out_dir / "_raw" / f"{code}_{picked['filename']}"
            print(f"== {code} downloading {picked['url']} ({picked['size']} bytes)", flush=True)
            curl_file(picked["url"], raw)
            dur = crop_wav(raw, wav, start=10.0 if code != "en" else 0.0, duration=15.0)
            # English: also copy jfk if we want a known clip; still keep LibriVox en.
            rec.update({
                "ok": True,
                "wav": str(wav.relative_to(out_dir.parent.parent)),  # tests/audio/langs/xx.wav from tests/
                "duration_s": dur,
                "source_id": picked["identifier"],
                "source_title": picked["title"],
                "source_file": picked["filename"],
                "source_url": picked["url"],
                "licenseurl": picked["licenseurl"],
                "ia_language": picked["language"],
            })
            print(f"   wrote {wav} ({dur:.2f}s)", flush=True)
        except Exception as e:
            rec.update({"ok": False, "error": str(e)})
            print(f"   FAIL {code}: {e}", flush=True)
        records.append(rec)
    return records


def commons_file_info(title: str) -> dict:
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&titles="
        + urllib.parse.quote(title)
        + "&prop=imageinfo&iiprop=url|extmetadata|size|mime&format=json"
    )
    data = curl_json(url)
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    info = (page.get("imageinfo") or [None])[0]
    if not info:
        raise RuntimeError(f"no imageinfo for {title}")
    meta = info.get("extmetadata") or {}
    lic = (meta.get("LicenseShortName") or {}).get("value") or (meta.get("License") or {}).get("value")
    return {
        "title": title,
        "url": info.get("url"),
        "size": info.get("size"),
        "license": lic,
        "artist": (meta.get("Artist") or {}).get("value"),
    }


def collect_unspaced(out_dir: Path) -> list[dict]:
    """Grab one slightly longer Lingua Libre utterance if possible, else first file."""
    records = []
    # category: Lingua Libre pronunciation-<iso>
    cats = {"zh": "cmn", "ja": "jpn", "th": "tha"}
    # Commons categories use Q-ids sometimes; try language ISO cats first
    cat_titles = {
        "zh": "Category:Lingua Libre pronunciation-cmn",
        "ja": "Category:Lingua Libre pronunciation-jpn",
        "th": "Category:Lingua Libre pronunciation-tha",
    }
    for code, cat in cat_titles.items():
        rec = {"code": code, "category": cat, "note": "segmentation-only, no timestamp quality claim"}
        try:
            url = (
                "https://commons.wikimedia.org/w/api.php?action=query&list=categorymembers"
                + "&cmtitle=" + urllib.parse.quote(cat)
                + "&cmtype=file&cmlimit=20&format=json"
            )
            data = curl_json(url)
            files = data["query"]["categorymembers"]
            if not files:
                raise RuntimeError(f"empty category {cat}")
            # pick the largest of the first 20 (more likely a phrase, not a letter)
            infos = []
            for f in files[:20]:
                try:
                    infos.append(commons_file_info(f["title"]))
                except Exception:
                    continue
            if not infos:
                raise RuntimeError("no file info")
            infos.sort(key=lambda x: int(x.get("size") or 0), reverse=True)
            picked = infos[0]
            url = picked["url"].split("?")[0]
            raw = out_dir / "_raw" / f"{code}_src{Path(url).suffix or '.wav'}"
            print(f"== {code} downloading {picked['url']}", flush=True)
            curl_file(url, raw, timeout=60)
            picked["url"] = url
            wav = out_dir / f"{code}.wav"
            # keep up to 15s; many are shorter
            dur = crop_wav(raw, wav, start=0.0, duration=15.0)
            rec.update({
                "ok": True,
                "wav": str(wav.name),
                "duration_s": dur,
                "source_title": picked["title"],
                "source_url": picked["url"],
                "license": picked["license"] or "CC BY-SA (Wikimedia Commons / Lingua Libre)",
            })
            print(f"   wrote {wav} ({dur:.2f}s)", flush=True)
        except Exception as e:
            rec.update({"ok": False, "error": str(e)})
            print(f"   FAIL {code}: {e}", flush=True)
        records.append(rec)
    return records


def write_sources(out_dir: Path, spaced: list[dict], unspaced: list[dict]) -> None:
    lines = [
        "# Language evaluation clips",
        "",
        "These files are **not** model weights.  They are short crops of",
        "public-domain or Creative Commons recordings, collected so the official",
        "10-language board (`en de es nl fr it pl pt ru sv`) can be regression-tested.",
        "",
        "Unspaced languages (`zh`, `ja`, `th`) are included only to verify",
        "segmentation.  Timestamp quality is **not** promised for unspaced scripts.",
        "",
        "## Spaced languages (LibriVox / Internet Archive, public domain)",
        "",
        "| code | title | duration | source | license |",
        "|---|---|---|---|---|",
    ]
    for r in spaced:
        if r.get("ok"):
            lines.append(
                f"| {r['code']} | {r.get('source_title','')} | {r.get('duration_s',0):.1f}s "
                f"| [{r.get('source_id')}](https://archive.org/details/{r.get('source_id')}) "
                f"| {r.get('licenseurl','PD')} |"
            )
        else:
            lines.append(f"| {r['code']} | FAILED |  |  | {r.get('error','')} |")
    lines += [
        "",
        "## Unspaced languages (Wikimedia Commons / Lingua Libre) — segmentation only",
        "",
        "| code | file | duration | license |",
        "|---|---|---|---|",
    ]
    for r in unspaced:
        if r.get("ok"):
            lines.append(f"| {r['code']} | {r.get('source_title','')} | {r.get('duration_s',0):.1f}s | {r.get('license')} |")
        else:
            lines.append(f"| {r['code']} | FAILED |  | {r.get('error','')} |")
    lines.append("")
    (out_dir / "SOURCES.md").write_text("\n".join(lines))
    (out_dir / "manifest.json").write_text(json.dumps({"spaced": spaced, "unspaced": unspaced}, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    out = ROOT
    out.mkdir(parents=True, exist_ok=True)
    spaced = collect_spaced(out)
    unspaced = collect_unspaced(out)
    write_sources(out, spaced, unspaced)
    ok_s = sum(1 for r in spaced if r.get("ok"))
    ok_u = sum(1 for r in unspaced if r.get("ok"))
    print(f"done spaced {ok_s}/{len(spaced)} unspaced {ok_u}/{len(unspaced)} -> {out}", flush=True)
    return 0 if ok_s == len(spaced) else 1


if __name__ == "__main__":
    raise SystemExit(main())
