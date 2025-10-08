#!/usr/bin/env python3
"""Embed regulations and collaborator profiles, build FAISS index, and emit mock recommendations."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

import numpy as np

DEFAULT_MODEL = "hashing"

csv.field_size_limit(10_000_000)
PROFILE_METADATA = {
    "alex-spangher": {
        "id": "alex-spangher",
        "name": "Alex Spangher",
        "role": "Postdoctoral scholar",
        "organization": "Stanford University",
        "homepage_url": "https://alexspangher.com",
        "interests": [
            "Computational Journalism",
            "Language Technologies",
            "Human-AI Collaboration",
        ],
    },
    "diyi-yang": {
        "id": "diyi-yang",
        "name": "Diyi Yang",
        "role": "Assistant Professor",
        "organization": "Stanford Computer Science",
        "homepage_url": "https://cs.stanford.edu/~diyiy/",
        "interests": [
            "Human-centered NLP",
            "Computational Social Science",
            "Responsible AI",
        ],
    },
    "sanmi-koyejo": {
        "id": "sanmi-koyejo",
        "name": "Sanmi Koyejo",
        "role": "Associate Professor",
        "organization": "Stanford Computer Science",
        "homepage_url": "https://cs.stanford.edu/~sanmi/",
        "interests": [
            "Machine Learning",
            "Neuroscience",
            "Bayesian Methods",
        ],
    },
    "dan-ho": {
        "id": "dan-ho",
        "name": "Daniel E. Ho",
        "role": "Professor",
        "organization": "Stanford Law School",
        "homepage_url": "https://law.stanford.edu/daniel-e-ho/",
        "interests": [
            "Administrative Law",
            "Data Science",
            "Technology Policy",
        ],
    },
}


@dataclass
class RegulationRecord:
    id: str
    title: str
    comment_due_date: str | None
    comment_active: bool
    metadata: Dict[str, Any]
    text: str


@dataclass
class ProfileRecord:
    id: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-glob",
        default="data/federal_rulemaking_*.csv",
        help="Glob pattern for regulation CSV snapshots.",
    )
    parser.add_argument(
        "--profiles-dir",
        default="data/user_profiles",
        help="Directory containing collaborator profile .txt files.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help='Embedding model ("tfidf" for lightweight mock or any sentence-transformers identifier).',
    )
    parser.add_argument(
        "--output-dir",
        default="data/vector_store",
        help="Directory to store embedding artifacts.",
    )
    parser.add_argument(
        "--recommendations-fixture",
        default="app/src/fixtures/recommendations.json",
        help="Path to write mock recommendations JSON for the frontend.",
    )
    parser.add_argument(
        "--profiles-fixture",
        default="app/src/fixtures/userProfiles.json",
        help="Path to write collaborator profile metadata JSON for the frontend.",
    )
    parser.add_argument(
        "--profile-summary-length",
        type=int,
        default=400,
        help="Maximum number of characters from each profile to include in the summary (<=0 for full text).",
    )
    parser.add_argument(
        "--regulation-text-length",
        type=int,
        default=1200,
        help="Maximum number of characters from each regulation to include in embeddings (<=0 for full text).",
    )
    parser.add_argument("--top-k", type=int, default=15, help="Top K regulations per collaborator.")
    return parser.parse_args()


def read_csv_records(pattern: str, max_regulation_length: int | None) -> Dict[str, RegulationRecord]:
    records: Dict[str, RegulationRecord] = {}
    for csv_path in sorted(Path().glob(pattern)):
        if not csv_path.is_file():
            continue
        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                text = build_regulation_text(row, max_regulation_length)
                if not text.strip():
                    continue
                identifier = row.get("fr_document_number") or row.get("source") or row.get("title")
                if not identifier:
                    continue
                identifier = identifier.strip()
                comment_due = sanitize_date(row.get("comment_due_date"))
                comment_active = parse_bool(row.get("comment_active"))
                metadata = {
                    "agency": row.get("agency", ""),
                    "comment_status": row.get("comment_status", ""),
                    "comment_due_date": comment_due,
                    "comment_active": comment_active,
                    "title": row.get("title", ""),
                }
                records[identifier] = RegulationRecord(
                    id=identifier,
                    title=row.get("title", identifier),
                    comment_due_date=comment_due,
                    comment_active=comment_active,
                    metadata=metadata,
                    text=text,
                )
    return records


def build_regulation_text(row: Dict[str, str], max_length: int | None) -> str:
    parts: List[str] = []
    for key in ("title", "agency", "abstract", "supplementary_information"):
        value = row.get(key)
        if value:
            parts.append(value)
    xml_blob = row.get("xml_dict")
    if xml_blob and len(xml_blob) <= 50000:
        try:
            xml_data = ast.literal_eval(xml_blob)
            xml_parts = list(iter_leaf_values(xml_data))
            if xml_parts:
                parts.append(" ".join(xml_parts))
        except Exception:
            # best effort – ignore malformed XML dicts
            pass
    text = "\n\n".join(p.strip() for p in parts if p and p.strip())
    if max_length and max_length > 0 and len(text) > max_length:
        text = text[:max_length]
    return text


def iter_leaf_values(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        for value in obj.values():
            yield from iter_leaf_values(value)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            yield from iter_leaf_values(item)
    elif obj is None:
        return
    else:
        yield str(obj)


def sanitize_date(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        # many dates already ISO; if parsing fails, fall back to value
        dt = datetime.fromisoformat(value)
        return dt.date().isoformat()
    except ValueError:
        try:
            dt = datetime.strptime(value, "%Y-%m-%d")
            return dt.date().isoformat()
        except Exception:
            return value


def parse_bool(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"true", "1", "yes", "y"}


def read_profile_records(profiles_dir: Path) -> List[ProfileRecord]:
    records: List[ProfileRecord] = []
    for txt_path in sorted(profiles_dir.glob("*.txt")):
        raw_text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw_text:
            continue
        slug = txt_path.stem.lower().replace("_", "-")
        if slug not in PROFILE_METADATA:
            PROFILE_METADATA[slug] = {
                "id": slug,
                "name": slug.replace("-", " ").title(),
                "role": "Collaborator",
                "organization": "",
                "homepage_url": "",
                "interests": [],
            }
        records.append(ProfileRecord(id=slug, text=raw_text))
    return records


def embed_with_model(
    regulation_texts: List[str],
    profile_texts: List[str],
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    lower = model_name.lower()
    if lower == "hashing":
        return embed_with_hashing(regulation_texts, profile_texts)
    if lower == "tfidf":
        return embed_with_tfidf(regulation_texts, profile_texts)

    from sentence_transformers import SentenceTransformer  # imported lazily

    model = SentenceTransformer(model_name)
    regulation_vectors = model.encode(
        regulation_texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    profile_vectors = model.encode(
        profile_texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    for vectors in (regulation_vectors, profile_vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors /= norms

    return regulation_vectors, profile_vectors


def tokenize(text: str) -> Iterable[str]:
    for token in re.findall(r"[A-Za-z0-9']+", text.lower()):
        if token:
            yield token


def embed_with_hashing(
    regulation_texts: List[str], profile_texts: List[str], dim: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    def embed(text: str) -> np.ndarray:
        vector = np.zeros(dim, dtype="float32")
        for token in tokenize(text):
            index = hash(token) % dim
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    reg_vectors = np.stack([embed(text) for text in regulation_texts], axis=0)
    profile_vectors = np.stack([embed(text) for text in profile_texts], axis=0)
    return reg_vectors, profile_vectors


def embed_with_tfidf(
    regulation_texts: List[str], profile_texts: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    combined = regulation_texts + profile_texts
    vectorizer = TfidfVectorizer(max_features=4096)
    matrix = vectorizer.fit_transform(combined)
    dense = matrix.astype("float32").toarray()

    reg_vectors = dense[: len(regulation_texts)]
    profile_vectors = dense[len(regulation_texts) :]

    for vectors in (reg_vectors, profile_vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors /= norms

    return reg_vectors, profile_vectors


def rank_recommendations(
    regulations: List[RegulationRecord],
    regulation_vectors: np.ndarray,
    profiles: List[ProfileRecord],
    profile_vectors: np.ndarray,
    top_k: int,
) -> Dict[str, List[Dict[str, Any]]]:
    if not profiles:
        return {}

    similarity = profile_vectors @ regulation_vectors.T
    top_k = min(top_k, regulation_vectors.shape[0])
    regulation_meta = {record.id: record for record in regulations}
    recommendation_map: Dict[str, List[Dict[str, Any]]] = {}

    for row, profile in enumerate(profiles):
        recs: List[Dict[str, Any]] = []
        row_scores = similarity[row]
        top_indices = np.argsort(-row_scores)[:top_k]
        for col in top_indices:
            score = float(row_scores[col])
            record = regulations[col]
            recs.append(
                {
                    "rule_id": record.id,
                    "score": score,
                    "comment_active": record.comment_active,
                    "comment_due_date": record.comment_due_date,
                }
            )
        recommendation_map[profile.id] = order_recommendations(recs, regulation_meta)

    return recommendation_map


def order_recommendations(
    recs: List[Dict[str, Any]],
    meta_lookup: Dict[str, RegulationRecord],
) -> List[Dict[str, Any]]:
    def sort_key(item: Dict[str, Any]) -> Tuple[int, float, float]:
        meta = meta_lookup.get(item["rule_id"])
        active_rank = 0 if meta and meta.comment_active else 1
        score = item.get("score", 0.0)
        due_date = meta.comment_due_date if meta else None
        due_value = parse_due_date(due_date)
        return (active_rank, -score, due_value)

    sorted_recs = sorted(recs, key=sort_key)
    return sorted_recs


def parse_due_date(value: str | None) -> float:
    if not value:
        return float("inf")
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return float("inf")


def priority_for(score: float, active: bool) -> str:
    if active and score >= 0.55:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    regulations_dict = read_csv_records(args.csv_glob, args.regulation_text_length)
    regulations = list(regulations_dict.values())
    print(f"Loaded {len(regulations)} regulations from {args.csv_glob}", flush=True)

    profiles_dir = Path(args.profiles_dir)
    profiles = read_profile_records(profiles_dir)
    if not profiles:
        raise SystemExit(f"No collaborator profiles found in {profiles_dir}")
    print(f"Loaded {len(profiles)} collaborator profiles from {profiles_dir}", flush=True)

    regulation_texts = [record.text for record in regulations]
    profile_texts = [record.text for record in profiles]

    print(
        f"Embedding {len(regulation_texts)} regulations and {len(profile_texts)} profiles using {args.model}...",
        flush=True,
    )
    regulation_vectors, profile_vectors = embed_with_model(regulation_texts, profile_texts, args.model)

    print("Computing similarity scores and ranking recommendations...", flush=True)
    recommendation_map = rank_recommendations(
        regulations,
        regulation_vectors,
        profiles,
        profile_vectors,
        top_k=args.top_k,
    )

    # Persist embeddings for future experimentation.
    np.save(output_dir / "regulation_vectors.npy", regulation_vectors)
    np.save(output_dir / "profile_vectors.npy", profile_vectors)

    with (output_dir / "regulations.json").open("w", encoding="utf-8") as fh:
        json.dump([record.__dict__ for record in regulations], fh, indent=2)

    with (output_dir / "profiles.json").open("w", encoding="utf-8") as fh:
        json.dump([profile.__dict__ for profile in profiles], fh, indent=2)

    # Build frontend fixtures.
    recommendation_fixture = []
    for profile in profiles:
        recs = recommendation_map.get(profile.id, [])
        formatted = [
            {
                "user_id": profile.id,
                "rule_id": item["rule_id"],
                "score": round(float(item["score"]), 4),
                "priority": priority_for(float(item["score"]), bool(item.get("comment_active"))),
                "match_topics": [],
                "reasons": [],
            }
            for item in recs
        ]
        recommendation_fixture.append({"user_id": profile.id, "recommendations": formatted})

    Path(args.recommendations_fixture).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.recommendations_fixture).open("w", encoding="utf-8") as fh:
        json.dump(recommendation_fixture, fh, indent=2)

    # User profile metadata fixture
    today = datetime.utcnow().date().isoformat()
    profile_fixture = []
    for profile in profiles:
        meta = PROFILE_METADATA.get(profile.id, {"id": profile.id, "name": profile.id, "interests": []})
        summary_length = args.profile_summary_length
        summary_text = profile.text
        if summary_length and summary_length > 0:
            summary_text = summary_text[:summary_length]

        profile_fixture.append(
            {
                **meta,
                "last_updated": today,
                "summary": summary_text,
            }
        )

    with Path(args.profiles_fixture).open("w", encoding="utf-8") as fh:
        json.dump(profile_fixture, fh, indent=2)

    print(f"Wrote frontend fixtures: {args.recommendations_fixture}, {args.profiles_fixture}", flush=True)
    print(f"Embeddings stored under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
