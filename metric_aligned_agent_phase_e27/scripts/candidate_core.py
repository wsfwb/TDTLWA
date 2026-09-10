import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
import numpy as np

INTERFACES = ("I1", "I2", "I3")

def _label(row: Mapping[str, object], source: str) -> int:
    value = int(row[f"{source.lower()}_reason_prediction"])
    if value not in range(6): raise ValueError(f"invalid label for {source}: {value}")
    return value

def _confidence(row: Mapping[str, object], source: str) -> float:
    value = float(row[f"{source.lower()}_confidence"])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0: raise ValueError(f"invalid confidence for {source}: {value}")
    return value

def _uncertainty(row: Mapping[str, object], source: str) -> float:
    value = float(row[f"{source.lower()}_uncertainty"])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0: raise ValueError(f"invalid uncertainty for {source}: {value}")
    return value

def stable_prefix_mask(scores: Sequence[float], canonical_ids: Sequence[str], k: int) -> np.ndarray:
    if len(scores) != len(canonical_ids) or len(set(canonical_ids)) != len(canonical_ids): raise ValueError("score/id alignment or uniqueness failure")
    if not 0 <= k <= len(scores): raise ValueError("k outside valid prefix range")
    numeric = [float(value) for value in scores]
    if any(not math.isfinite(value) for value in numeric): raise ValueError("non-finite score")
    order = sorted(range(len(numeric)), key=lambda index: (-numeric[index], str(canonical_ids[index])))
    mask = np.zeros(len(numeric), dtype=bool); mask[order[:k]] = True
    return mask

def majority_label(labels: Sequence[int]) -> tuple[int | None, int]:
    values = [int(value) for value in labels]
    if any(value not in range(6) for value in values): raise ValueError("label outside 0..5")
    counts = Counter(values); best_count = max(counts.values(), default=0)
    if best_count < 2: return None, best_count
    best_label = min(label for label, count in counts.items() if count == best_count)
    return best_label, best_count

def proposal_for(source: str, row: Mapping[str, object]) -> tuple[int | None, tuple[str, ...]]:
    labels = {name: _label(row, name) for name in INTERFACES}
    if source == "majority_2of3":
        label, support = majority_label(list(labels.values())); supporters = tuple(name for name in INTERFACES if labels[name] == label) if support >= 2 else ()
        return label, supporters
    if source == "unanimous_3of3": return (labels["I3"], INTERFACES) if len(set(labels.values())) == 1 else (None, ())
    if source == "i3_i1": return (labels["I3"], ("I3", "I1")) if labels["I3"] == labels["I1"] else (None, ())
    if source == "i3_i2": return (labels["I3"], ("I3", "I2")) if labels["I3"] == labels["I2"] else (None, ())
    if source == "i3_any_confirm":
        supporters = tuple(name for name in INTERFACES if labels[name] == labels["I3"])
        return (labels["I3"], supporters) if len(supporters) >= 2 else (None, ())
    if source == "best_supported_label":
        candidates = []
        for label, support in Counter(labels.values()).items():
            if support >= 2:
                supporters = tuple(name for name in INTERFACES if labels[name] == label)
                margin = sum(_confidence(row, name) - _uncertainty(row, name) for name in supporters) / len(supporters)
                candidates.append((-margin, -support, label, supporters))
        if not candidates: return None, ()
        _, _, label, supporters = min(candidates); return label, supporters
    raise ValueError(f"unknown proposal source: {source}")

def proposal_score(mode: str, supporters: Sequence[str], row: Mapping[str, object]) -> float:
    if not supporters: raise ValueError("proposal has no supporters")
    confidence = [_confidence(row, name) for name in supporters]
    margin = [_confidence(row, name) - _uncertainty(row, name) for name in supporters]
    if mode == "mean_confidence": return sum(confidence) / len(confidence)
    if mode == "min_confidence": return min(confidence)
    if mode == "mean_margin": return sum(margin) / len(margin)
    if mode == "max_margin": return max(margin)
    if mode.startswith("support_bonus_"):
        gamma = float(mode.removeprefix("support_bonus_"))
        if gamma not in {0.10, 0.25, 0.50}: raise ValueError(f"unregistered support bonus: {gamma}")
        return sum(margin) / len(margin) + gamma * (len(supporters) - 1)
    raise ValueError(f"unknown proposal score: {mode}")

def weighted_vote(labels: Mapping[str, int], weights: Mapping[str, float], tie_order: Sequence[str]) -> int:
    if set(labels) != set(weights): raise ValueError("label/weight sources differ")
    totals = {label: 0.0 for label in range(6)}
    for source, raw_label in labels.items():
        label = int(raw_label); weight = float(weights[source])
        if label not in range(6) or not math.isfinite(weight) or weight < 0: raise ValueError("invalid vote input")
        totals[label] += weight
    best = max(totals.values()); tied = {label for label, total in totals.items() if abs(total - best) <= 1e-12}
    for source in tie_order:
        if source in labels and int(labels[source]) in tied: return int(labels[source])
    raise ValueError("tie order does not cover vote sources")

def action_for(final_label: int, full_label: int, d3_label: int) -> str:
    if any(int(value) not in range(6) for value in (final_label, full_label, d3_label)): raise ValueError("label outside 0..5")
    if int(final_label) == int(full_label): return "KEEP"
    if int(final_label) == int(d3_label): return "RESIDUAL"
    return "REASON"

def candidate_id(parts: Mapping[str, object]) -> str:
    normalized = json.dumps(dict(sorted(parts.items())), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return "cand_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
