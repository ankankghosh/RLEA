r"""
Windowed dataset loader for the RLEA frozen corpus.
===================================================

Module role in RLEA
-------------------
**M2.1.** The only module in Modules 2-5 that touches Parquet. Everything
downstream -- the ensemble (M2.3), the feature ranking, the learned similarity
embedding :math:`\phi` (M4.1), the attribution machinery (M4.2) and the
perturbation harness (M5) -- consumes windows produced here and never reads
`data/raw` directly. Changing the window geometry or the normalisation is
therefore a breaking change for every downstream result.

Contract
--------
Input   ``data/raw/`` as written by ``generate.py`` at tag ``data-raw-v1``:
        ``episodes.parquet``, ``norm_stats.json``, ``run_info.json``,
        ``timeseries/split=<name>/part-NNNN.parquet``.
Output  ``Window`` records: ``x`` (T, F) normalised features, ``mask`` (T, F)
        observation mask, ``y`` targets, ``meta`` provenance.

Design commitments
------------------
**1. A dropped sample is NaN, never zero (A-SEN-03).** ``degradation.py`` is
careful never to encode absence as zero, because zero is a legal value for
angle of attack, vertical speed and the wind components. This module preserves
that care: standardisation happens first, the fill happens second, so a missing
sample lands at the training mean rather than at a false measurement, and the
companion ``mask`` channel records that it was missing at all.

The mask is *not* redundant with ``hlth_*_valid``. Health flags are per
**group** (a failed air-data computer takes six channels with it); the mask is
per channel. More importantly the two disagree by design: a ``frozen`` channel
reports ``valid = 1`` with a finite, plausible value (A-SEN-04). Detecting that
requires temporal or cross-channel reasoning, which is the capability under
evaluation -- so the loader must not paper over it.

**2. Not every feature is standardised.** Standardising a binary flag whose
train mean is 0.997 turns a rare event into a 17-sigma spike, and the
sensor-novelty split would then look separable for entirely the wrong reason.
Staleness counters are non-negative and heavy-tailed (geometric dropout bursts;
``frozen`` grows without bound), so they are ``log1p``-compressed first. See
:class:`Transform`.

**3. Normalisation statistics are train-only, and are re-derived once.**
``norm_stats.json`` holds raw-scale mean/std fitted on ``train`` alone. For
``log1p`` features those statistics are on the wrong scale, so a single
train-only pass computes post-transform statistics and caches them. That pass
also cross-checks the untransformed features against ``norm_stats.json``, which
independently verifies that this reader and the generator agree about the data.

Never recompute statistics per split. Fitting on ``test_sensor`` normalises
away precisely the shift the OOD experiment is supposed to detect.

**4. The leakage guard is mechanical (A-METH-04).** ``generate.py`` asserts no
``gt_*`` or ``phys_*`` column is *written* into the feature set. This module
asserts none is *read* into it. Two independent checks, because the prefix
discipline is the assumption the entire evaluation rests on and one guard is a
single point of failure. A ``gt_*`` column may still be used to *select* rows
(see ``exclude_post_breach``); selection is not featurisation.

**5. Windows never cross an episode boundary**, and splits are per episode, so
no near-duplicate window can straddle train and test. At 60 s / 1 s geometry
consecutive windows overlap 98 %; window-level splitting would be fatal and is
structurally impossible here.

Window geometry
---------------
Default 60 s windows (600 rows at 10 Hz) with a 1 s stride, matching the
geometry ``generate.py``'s split rationale was written against. Stride is a
configuration knob: 1 s over the full corpus is ~25 M windows, which is a
training-time subsampling decision, not a dataset decision.

The label is taken at the **last** sample of the window, so the model predicts
the present from the preceding minute and never sees its own future. Retrieved
precedents' futures are of course available -- they are finished flights -- and
that asymmetry is the point of M4.1.

Exclusions
----------
* ``gt_post_breach == 1`` at the label row: post-stall aerodynamics are
  unmodelled, so these samples are outside the validity domain (A-AC-03). The
  episodes are retained; only the affected windows are dropped.
* Short windows at episode start.
* ``beyond_calibration`` episodes are **kept** (open item 1b, closed
  2026-09-09). Analyses that need the calibrated region condition on *crew
  policy*, not on the flag -- the flagged population is 92 %
  ``crew_never_reacts`` and filtering on it would strip half of one level of
  the crew-policy factor.

Assumptions and limitations
---------------------------
* Rows within an episode are assumed contiguous and in ``step`` order within a
  shard. Verified on load rather than trusted.
* Shard-level caching assumes the access pattern is episode-grouped; random
  access across the corpus will thrash. Use :class:`EpisodeGroupedSampler`.
* Label columns may be stored as strings or as codes depending on the writer;
  both are handled, and the mapping is frozen here (see :data:`RISK_CODES`).

Usage
-----
::

    python dataset.py --data data/raw --split train --verify

"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np
import pyarrow.parquet as pq

__all__ = [
    "FEATURE_PREFIXES",
    "FORBIDDEN_PREFIXES",
    "RISK_CODES",
    "SEVERITY_CODES",
    "Transform",
    "LeakageError",
    "SchemaError",
    "FeatureSchema",
    "TransformStats",
    "WindowIndex",
    "ShardStore",
    "WindowDataset",
    "EpisodeGroupedSampler",
]

# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

#: Only these prefixes may become model inputs (A-METH-04).
FEATURE_PREFIXES: tuple[str, ...] = ("obs_", "hlth_")

#: These must never appear in the feature set. ``gt_`` may be *read* for row
#: selection; it may never be featurised.
FORBIDDEN_PREFIXES: tuple[str, ...] = ("gt_", "lbl_", "phys_")

#: Substrings that must not appear inside a feature-prefixed column name. A
#: column called ``obs_gt_ice_mm`` would pass a prefix check and is exactly the
#: kind of silent leak this catches.
FORBIDDEN_MARKERS: tuple[str, ...] = ("_gt_", "phys_", "_truth", "_ground_truth")

#: Expected model-input count: 29 ``obs_`` + 33 ``hlth_``. The generator prints
#: this as ``features: 62``; it is not the row width (118).
EXPECTED_N_FEATURES: int = 62

#: Envelope state, ordered by severity. Frozen in ``episode.py``.
RISK_CODES: dict[str, int] = {"nominal": 0, "caution": 1, "warning": 2, "breach": 3}

#: Icing severity from accretion rate, mm/min: none <0.01, trace <0.10,
#: light <0.60, moderate <2.00, severe >=2.00. Frozen in ``episode.py``.
SEVERITY_CODES: dict[str, int] = {
    "none": 0,
    "trace": 1,
    "light": 2,
    "moderate": 3,
    "severe": 4,
}

SAMPLE_RATE_HZ: float = 10.0
DT_S: float = 1.0 / SAMPLE_RATE_HZ

_LABEL_COLUMNS: tuple[str, ...] = (
    "lbl_risk_class",
    "lbl_icing_severity",
    "lbl_time_to_critical_s",
    "lbl_critical_censored",
)

#: ``gt_`` columns read for row *selection* only. Never featurised.
_SELECTION_COLUMNS: tuple[str, ...] = ("gt_post_breach",)

_INDEX_COLUMNS: tuple[str, ...] = ("episode_id", "step")


class LeakageError(RuntimeError):
    """A forbidden column reached, or nearly reached, the feature set."""


class SchemaError(RuntimeError):
    """The on-disk schema does not match the frozen data contract."""


# ---------------------------------------------------------------------------
# Feature schema and per-feature transforms
# ---------------------------------------------------------------------------

Transform = Literal["standardise", "log1p", "binary"]


def _classify(name: str) -> Transform:
    r"""Per-feature transform, decided by name.

    ``binary``
        0/1 indicators. Passed through untouched. Standardising these against a
        train mean near 1.0 would convert a rare, informative event into an
        extreme-valued spike and hand any density-based OOD score a trivially
        separable axis.
    ``log1p``
        Non-negative, heavy-tailed counters. Dropouts are geometric bursts and
        a ``frozen`` channel's staleness grows without bound, so one long fault
        would otherwise set the scale for the whole channel.
    ``standardise``
        Everything else, using train-only mean and standard deviation.
    """
    if name.endswith(("_valid", "_imputed", "_faulted")):
        return "binary"
    if name in ("obs_ice_detector_active",):
        return "binary"
    if name.endswith("_staleness_s") or name == "hlth_n_channels_invalid":
        return "log1p"
    return "standardise"


@dataclass(frozen=True)
class FeatureSchema:
    """Resolved, validated, deterministically ordered feature set.

    Attributes
    ----------
    names : tuple of str
        Feature column names, sorted. Order is fixed here and is the order of
        the feature axis everywhere downstream; do not re-sort elsewhere.
    transforms : tuple of str
        Parallel to ``names``.
    """

    names: tuple[str, ...]
    transforms: tuple[Transform, ...]

    @property
    def n_features(self) -> int:
        return len(self.names)

    def index_of(self, name: str) -> int:
        return self.names.index(name)

    @classmethod
    def from_parquet(
        cls, path: Path, *, expect: int | None = EXPECTED_N_FEATURES
    ) -> "FeatureSchema":
        """Resolve the feature set from a shard's schema, with leakage guard.

        Raises
        ------
        LeakageError
            A forbidden prefix or marker appears in a feature-prefixed name.
        SchemaError
            The feature count differs from the frozen contract, or an
            unrecognised prefix is present (so a newly added column cannot
            silently default into the feature set).
        """
        schema = pq.ParquetFile(str(path)).schema_arrow
        all_names = list(schema.names)

        feats = [n for n in all_names if n.startswith(FEATURE_PREFIXES)]

        bad = [n for n in feats if any(m in n for m in FORBIDDEN_MARKERS)]
        if bad:
            raise LeakageError(
                f"forbidden marker inside feature-prefixed column(s): {bad}"
            )
        bad = [n for n in feats if n.startswith(FORBIDDEN_PREFIXES)]
        if bad:
            raise LeakageError(f"forbidden prefix in feature set: {bad}")

        known = FEATURE_PREFIXES + FORBIDDEN_PREFIXES
        unknown = [
            n
            for n in all_names
            if not n.startswith(known) and n not in ("episode_id", "step", "t_s", "phase")
        ]
        if unknown:
            raise SchemaError(
                "column(s) with unrecognised prefix -- classify them explicitly "
                f"before proceeding: {unknown}"
            )

        feats.sort()
        if expect is not None and len(feats) != expect:
            raise SchemaError(
                f"expected {expect} features (29 obs_ + 33 hlth_), got {len(feats)}. "
                "If the schema genuinely changed, update EXPECTED_N_FEATURES and "
                "record it in CHANGELOG.md -- do not silence this."
            )

        return cls(tuple(feats), tuple(_classify(n) for n in feats))


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------


@dataclass
class TransformStats:
    r"""Post-transform mean and standard deviation, fitted on ``train`` only.

    ``norm_stats.json`` carries raw-scale statistics. They are correct for
    ``standardise`` features and wrong for ``log1p`` ones, so this class runs a
    single train-only pass to derive statistics *after* the transform, caches
    the result, and cross-checks the untransformed features against the
    generator's own numbers. A disagreement there means this reader and
    ``generate.py`` disagree about the data, which is worth knowing loudly.
    """

    names: tuple[str, ...]
    mean: np.ndarray  # (F,) float32
    std: np.ndarray  # (F,) float32
    nan_fraction: np.ndarray  # (F,) float32, informational

    CACHE_NAME = "norm_stats_transformed.json"

    # -- application -------------------------------------------------------

    def apply(self, raw: np.ndarray, schema: FeatureSchema) -> tuple[np.ndarray, np.ndarray]:
        r"""Normalise a raw block and return ``(x, mask)``.

        Order matters. Standardise first, fill second: a missing sample then
        sits at the training mean instead of at zero, which for angle of
        attack, vertical speed or a wind component would be a legal -- and
        false -- measurement (A-SEN-03).
        """
        mask = np.isfinite(raw).astype(np.float32)
        x = raw.astype(np.float32, copy=True)

        log_idx = [i for i, t in enumerate(schema.transforms) if t == "log1p"]
        if log_idx:
            col = x[:, log_idx]
            np.log1p(np.clip(col, 0.0, None), out=col)
            x[:, log_idx] = col

        std_idx = [i for i, t in enumerate(schema.transforms) if t != "binary"]
        if std_idx:
            x[:, std_idx] = (x[:, std_idx] - self.mean[std_idx]) / self.std[std_idx]

        np.copyto(x, 0.0, where=(mask == 0.0))
        return x, mask

    # -- fitting and caching ----------------------------------------------

    @classmethod
    def fit_or_load(
        cls,
        root: Path,
        schema: FeatureSchema,
        *,
        train_split: str = "train",
        refit: bool = False,
        verbose: bool = True,
    ) -> "TransformStats":
        cache = root / cls.CACHE_NAME
        if cache.exists() and not refit:
            blob = json.loads(cache.read_text())
            if tuple(blob["names"]) == schema.names:
                return cls(
                    names=tuple(blob["names"]),
                    mean=np.asarray(blob["mean"], dtype=np.float32),
                    std=np.asarray(blob["std"], dtype=np.float32),
                    nan_fraction=np.asarray(blob["nan_fraction"], dtype=np.float32),
                )

        if verbose:
            print(f"[norm] fitting post-transform statistics on '{train_split}' ...")

        shards = sorted((root / "timeseries" / f"split={train_split}").glob("*.parquet"))
        if not shards:
            raise SchemaError(f"no shards found for split '{train_split}' under {root}")

        f = schema.n_features
        count = np.zeros(f, dtype=np.float64)
        total = np.zeros(f, dtype=np.float64)
        ssum = np.zeros(f, dtype=np.float64)
        sqsum = np.zeros(f, dtype=np.float64)

        for shard in shards:
            block = _read_features(shard, schema)
            finite = np.isfinite(block)
            transformed = block.astype(np.float64, copy=True)
            for i, t in enumerate(schema.transforms):
                if t == "log1p":
                    col = transformed[:, i]
                    transformed[:, i] = np.log1p(np.clip(col, 0.0, None))
            transformed[~finite] = 0.0

            count += finite.sum(axis=0)
            total += block.shape[0]
            ssum += transformed.sum(axis=0)
            sqsum += (transformed**2).sum(axis=0)

        n = np.maximum(count, 1.0)
        mean = ssum / n
        var = np.maximum(sqsum / n - mean**2, 0.0)
        std = np.sqrt(var)

        # A zero-variance channel is not an error -- a flag can be constant on
        # train -- but dividing by it is. Neutralise instead.
        degenerate = std < 1e-8
        std[degenerate] = 1.0
        mean[degenerate] = np.where(
            np.asarray([t == "binary" for t in schema.transforms])[degenerate],
            mean[degenerate],
            0.0,
        )

        stats = cls(
            names=schema.names,
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            nan_fraction=(1.0 - count / np.maximum(total, 1.0)).astype(np.float32),
        )
        stats._cross_check(root, schema, verbose=verbose)

        cache.write_text(
            json.dumps(
                {
                    "names": list(stats.names),
                    "mean": stats.mean.tolist(),
                    "std": stats.std.tolist(),
                    "nan_fraction": stats.nan_fraction.tolist(),
                    "fitted_on": train_split,
                    "note": "post-transform, train-only. Do not refit per split.",
                },
                indent=1,
            )
        )
        if verbose:
            print(f"[norm] cached -> {cache}")
        return stats

    def _cross_check(self, root: Path, schema: FeatureSchema, *, verbose: bool) -> None:
        """Compare untransformed features against the generator's own stats."""
        path = root / "norm_stats.json"
        if not path.exists():
            if verbose:
                print("[norm] norm_stats.json absent -- cross-check skipped")
            return
        ref = json.loads(path.read_text())
        worst_name, worst_rel = None, 0.0
        for i, name in enumerate(schema.names):
            if schema.transforms[i] != "standardise":
                continue
            entry = ref.get(name)
            if not isinstance(entry, dict) or "mean" not in entry:
                continue
            denom = max(abs(float(entry["mean"])), float(entry.get("std", 1.0)), 1e-6)
            rel = abs(float(entry["mean"]) - float(self.mean[i])) / denom
            if rel > worst_rel:
                worst_name, worst_rel = name, rel
        if verbose:
            verdict = "OK" if worst_rel < 1e-3 else "MISMATCH"
            print(
                f"[norm] cross-check vs norm_stats.json: {verdict} "
                f"(worst {worst_name} rel {worst_rel:.2e})"
            )
        if worst_rel >= 1e-2:
            raise SchemaError(
                f"reader and generator disagree on '{worst_name}' "
                f"(relative {worst_rel:.3e}). Investigate before training."
            )


# ---------------------------------------------------------------------------
# Low-level shard reading
# ---------------------------------------------------------------------------


def _read_features(path: Path, schema: FeatureSchema) -> np.ndarray:
    """Read one shard's feature block as ``(rows, F)`` float32."""
    table = pq.read_table(str(path), columns=list(schema.names))
    cols = [table.column(n).to_numpy(zero_copy_only=False) for n in schema.names]
    return np.stack(cols, axis=1).astype(np.float32, copy=False)


def _encode_labels(values: np.ndarray, mapping: dict[str, int]) -> np.ndarray:
    """Map a label column to int8 codes, accepting strings or existing codes."""
    if values.dtype.kind in "iuf":
        out = values.astype(np.int16)
        if out.max(initial=0) >= len(mapping):
            raise SchemaError(f"label code out of range for mapping {mapping}")
        return out.astype(np.int8)
    out = np.full(values.shape, -1, dtype=np.int8)
    for text, code in mapping.items():
        out[values == text] = code
    unknown = np.unique(values[out < 0])
    if unknown.size:
        raise SchemaError(f"unmapped label value(s): {unknown.tolist()}")
    return out


# ---------------------------------------------------------------------------
# Window index
# ---------------------------------------------------------------------------


@dataclass
class WindowIndex:
    r"""Flat, array-backed index of every admissible window in a split.

    Arrays, not a list of records: at 60 s / 1 s the full corpus is ~25 M
    windows and the index itself has to stay cheap.

    ``row_start`` / ``row_end`` are offsets **within the shard**, so a window
    is read without knowing anything about episode boundaries at access time;
    the boundaries were enforced when the index was built.
    """

    split: str
    window_rows: int
    stride_rows: int
    shard_paths: tuple[str, ...]
    shard_idx: np.ndarray  # (N,) int16
    episode_id: np.ndarray  # (N,) int32
    row_start: np.ndarray  # (N,) int32
    row_end: np.ndarray  # (N,) int32, exclusive; label at row_end - 1
    t_s: np.ndarray  # (N,) float32
    y_risk: np.ndarray  # (N,) int8
    y_severity: np.ndarray  # (N,) int8
    y_ttc_s: np.ndarray  # (N,) float32, NaN when censored
    y_censored: np.ndarray  # (N,) int8
    n_dropped_post_breach: int = 0
    n_dropped_short: int = 0

    def __len__(self) -> int:
        return int(self.episode_id.shape[0])

    # -- build -------------------------------------------------------------

    @classmethod
    def build(
        cls,
        root: Path,
        split: str,
        *,
        window_s: float = 60.0,
        stride_s: float = 1.0,
        exclude_post_breach: bool = True,
        verbose: bool = True,
    ) -> "WindowIndex":
        window_rows = int(round(window_s * SAMPLE_RATE_HZ))
        stride_rows = max(int(round(stride_s * SAMPLE_RATE_HZ)), 1)

        shards = sorted((root / "timeseries" / f"split={split}").glob("*.parquet"))
        if not shards:
            raise SchemaError(f"no shards found for split '{split}' under {root}")

        cols = list(_INDEX_COLUMNS) + list(_LABEL_COLUMNS) + list(_SELECTION_COLUMNS)

        s_idx, e_id, r0, r1, ts = [], [], [], [], []
        yr, ysev, yttc, ycen = [], [], [], []
        n_breach = n_short = 0

        for k, shard in enumerate(shards):
            table = pq.read_table(str(shard), columns=cols)
            ep = table.column("episode_id").to_numpy(zero_copy_only=False).astype(np.int32)
            step = table.column("step").to_numpy(zero_copy_only=False).astype(np.int64)

            risk = _encode_labels(
                table.column("lbl_risk_class").to_numpy(zero_copy_only=False), RISK_CODES
            )
            sev = _encode_labels(
                table.column("lbl_icing_severity").to_numpy(zero_copy_only=False),
                SEVERITY_CODES,
            )
            ttc = (
                table.column("lbl_time_to_critical_s")
                .to_numpy(zero_copy_only=False)
                .astype(np.float32)
            )
            cen = (
                table.column("lbl_critical_censored")
                .to_numpy(zero_copy_only=False)
                .astype(np.int8)
            )
            post = (
                table.column("gt_post_breach")
                .to_numpy(zero_copy_only=False)
                .astype(np.int8)
            )

            # Episode boundaries. Trust nothing: verify contiguity and order.
            bounds = np.flatnonzero(np.diff(ep) != 0) + 1
            starts = np.concatenate(([0], bounds))
            ends = np.concatenate((bounds, [ep.shape[0]]))

            for a, b in zip(starts, ends):
                seg = step[a:b]
                if seg.shape[0] > 1 and not np.all(np.diff(seg) == 1):
                    raise SchemaError(
                        f"episode {int(ep[a])} in {shard.name} is not contiguous "
                        "in 'step'; the row-range assumption is violated."
                    )
                n_rows = b - a
                if n_rows < window_rows:
                    n_short += 1
                    continue

                local_end = np.arange(window_rows, n_rows + 1, stride_rows, dtype=np.int64)
                local_lbl = local_end - 1

                if exclude_post_breach:
                    keep = post[a + local_lbl] == 0
                    n_breach += int((~keep).sum())
                    local_end = local_end[keep]
                    local_lbl = local_lbl[keep]
                if local_end.size == 0:
                    continue

                s_idx.append(np.full(local_end.shape, k, dtype=np.int16))
                e_id.append(np.full(local_end.shape, ep[a], dtype=np.int32))
                r1.append((a + local_end).astype(np.int32))
                r0.append((a + local_end - window_rows).astype(np.int32))
                ts.append((step[a + local_lbl] * DT_S).astype(np.float32))
                yr.append(risk[a + local_lbl])
                ysev.append(sev[a + local_lbl])
                yttc.append(ttc[a + local_lbl])
                ycen.append(cen[a + local_lbl])

            if verbose:
                print(f"[index] {split}: {shard.name} -> {sum(x.size for x in e_id):,} windows")

        def cat(chunks, dtype):
            return (
                np.concatenate(chunks).astype(dtype)
                if chunks
                else np.zeros(0, dtype=dtype)
            )

        return cls(
            split=split,
            window_rows=window_rows,
            stride_rows=stride_rows,
            shard_paths=tuple(str(p) for p in shards),
            shard_idx=cat(s_idx, np.int16),
            episode_id=cat(e_id, np.int32),
            row_start=cat(r0, np.int32),
            row_end=cat(r1, np.int32),
            t_s=cat(ts, np.float32),
            y_risk=cat(yr, np.int8),
            y_severity=cat(ysev, np.int8),
            y_ttc_s=cat(yttc, np.float32),
            y_censored=cat(ycen, np.int8),
            n_dropped_post_breach=n_breach,
            n_dropped_short=n_short,
        )

    # -- cache -------------------------------------------------------------

    @classmethod
    def cached(
        cls, root: Path, split: str, *, window_s: float = 60.0, stride_s: float = 1.0, **kw
    ) -> "WindowIndex":
        tag = f"{split}_w{int(window_s)}_s{stride_s:g}"
        path = root / "index" / f"{tag}.npz"
        if path.exists():
            blob = np.load(path, allow_pickle=False)
            return cls(
                split=split,
                window_rows=int(blob["window_rows"]),
                stride_rows=int(blob["stride_rows"]),
                shard_paths=tuple(str(s) for s in blob["shard_paths"]),
                shard_idx=blob["shard_idx"],
                episode_id=blob["episode_id"],
                row_start=blob["row_start"],
                row_end=blob["row_end"],
                t_s=blob["t_s"],
                y_risk=blob["y_risk"],
                y_severity=blob["y_severity"],
                y_ttc_s=blob["y_ttc_s"],
                y_censored=blob["y_censored"],
                n_dropped_post_breach=int(blob["n_dropped_post_breach"]),
                n_dropped_short=int(blob["n_dropped_short"]),
            )
        index = cls.build(root, split, window_s=window_s, stride_s=stride_s, **kw)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            window_rows=index.window_rows,
            stride_rows=index.stride_rows,
            shard_paths=np.asarray(index.shard_paths),
            shard_idx=index.shard_idx,
            episode_id=index.episode_id,
            row_start=index.row_start,
            row_end=index.row_end,
            t_s=index.t_s,
            y_risk=index.y_risk,
            y_severity=index.y_severity,
            y_ttc_s=index.y_ttc_s,
            y_censored=index.y_censored,
            n_dropped_post_breach=index.n_dropped_post_breach,
            n_dropped_short=index.n_dropped_short,
        )
        return index


# ---------------------------------------------------------------------------
# Shard store
# ---------------------------------------------------------------------------


class ShardStore:
    """LRU cache of decoded shard feature blocks.

    A shard is 25 episodes, roughly 445 k rows x 62 float32 ~ 110 MB decoded.
    Two cached shards is a comfortable default; raise it only with the access
    pattern in mind, since random access across the corpus defeats the cache
    entirely. See :class:`EpisodeGroupedSampler`.
    """

    def __init__(self, schema: FeatureSchema, capacity: int = 2) -> None:
        self.schema = schema
        self.capacity = max(capacity, 1)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def block(self, path: str) -> np.ndarray:
        cached = self._cache.get(path)
        if cached is not None:
            self.hits += 1
            self._cache.move_to_end(path)
            return cached
        self.misses += 1
        block = _read_features(Path(path), self.schema)
        self._cache[path] = block
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return block


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class Window:
    """One window. ``meta`` exists so M4 can point back at a real instant."""

    x: np.ndarray  # (T, F) normalised
    mask: np.ndarray  # (T, F) 1 = observed
    y_risk: int
    y_severity: int
    y_ttc_s: float
    y_censored: int
    episode_id: int
    t_s: float
    split: str


class WindowDataset:
    """Windows over one split. Deliberately free of any deep-learning import.

    Wrap in a framework ``Dataset`` downstream if wanted; keeping this module
    dependency-light means the feature ranking and the M4 retrieval index can
    use it without dragging a training stack in.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        window_s: float = 60.0,
        stride_s: float = 1.0,
        schema: FeatureSchema | None = None,
        stats: TransformStats | None = None,
        cache_shards: int = 2,
        exclude_post_breach: bool = True,
        verbose: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        shards = sorted((self.root / "timeseries" / f"split={split}").glob("*.parquet"))
        if not shards:
            raise SchemaError(f"no shards found for split '{split}' under {self.root}")

        self.schema = schema or FeatureSchema.from_parquet(shards[0])
        self.stats = stats or TransformStats.fit_or_load(
            self.root, self.schema, verbose=verbose
        )
        self.index = WindowIndex.cached(
            self.root,
            split,
            window_s=window_s,
            stride_s=stride_s,
            exclude_post_breach=exclude_post_breach,
            verbose=verbose,
        )
        self.store = ShardStore(self.schema, capacity=cache_shards)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Window:
        idx = self.index
        block = self.store.block(idx.shard_paths[int(idx.shard_idx[i])])
        raw = block[int(idx.row_start[i]) : int(idx.row_end[i])]
        x, mask = self.stats.apply(raw, self.schema)
        return Window(
            x=x,
            mask=mask,
            y_risk=int(idx.y_risk[i]),
            y_severity=int(idx.y_severity[i]),
            y_ttc_s=float(idx.y_ttc_s[i]),
            y_censored=int(idx.y_censored[i]),
            episode_id=int(idx.episode_id[i]),
            t_s=float(idx.t_s[i]),
            split=self.split,
        )

    def batch(self, ids: Sequence[int]) -> dict[str, np.ndarray]:
        """Collate a batch. ``meta`` is carried through for episode-level metrics."""
        windows = [self[int(i)] for i in ids]
        return {
            "x": np.stack([w.x for w in windows]),
            "mask": np.stack([w.mask for w in windows]),
            "y_risk": np.asarray([w.y_risk for w in windows], dtype=np.int64),
            "y_severity": np.asarray([w.y_severity for w in windows], dtype=np.int64),
            "y_ttc_s": np.asarray([w.y_ttc_s for w in windows], dtype=np.float32),
            "y_censored": np.asarray([w.y_censored for w in windows], dtype=np.int64),
            "episode_id": np.asarray([w.episode_id for w in windows], dtype=np.int64),
            "t_s": np.asarray([w.t_s for w in windows], dtype=np.float32),
        }


class EpisodeGroupedSampler:
    """Shuffle episodes, not windows.

    Shuffling 25 M windows uniformly would evict the shard cache on nearly
    every access. Shuffling at episode level -- and shards before that -- keeps
    reads sequential while still decorrelating consecutive batches, which is
    all the shuffling is for. Within a batch the windows come from one episode,
    so any batch-level statistic is an episode-level statistic; that is a
    feature for grouped evaluation and a caveat for batch normalisation.
    """

    def __init__(
        self,
        dataset: WindowDataset,
        batch_size: int = 64,
        *,
        seed: int = 0,
        subsample: float = 1.0,
        drop_last: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.subsample = float(np.clip(subsample, 1e-6, 1.0))
        self.drop_last = drop_last
        idx = dataset.index
        order = np.lexsort((idx.row_start, idx.episode_id, idx.shard_idx))
        self._groups: list[np.ndarray] = np.split(
            order, np.flatnonzero(np.diff(idx.episode_id[order]) != 0) + 1
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        groups = list(self._groups)
        rng.shuffle(groups)
        for group in groups:
            g = group
            if self.subsample < 1.0:
                take = max(int(round(g.size * self.subsample)), 1)
                g = rng.choice(g, size=take, replace=False)
            g = g[rng.permutation(g.size)]
            for s in range(0, g.size, self.batch_size):
                chunk = g[s : s + self.batch_size]
                if self.drop_last and chunk.size < self.batch_size:
                    continue
                yield chunk


# ---------------------------------------------------------------------------
# Verification CLI
# ---------------------------------------------------------------------------


def verify(root: Path, split: str, *, window_s: float, stride_s: float) -> None:
    """Structural check of one split. Run this before trusting any training."""
    print(f"\n=== RLEA dataset verification: split '{split}' ===")
    t0 = time.time()

    shards = sorted((root / "timeseries" / f"split={split}").glob("*.parquet"))
    print(f"shards              : {len(shards)}")

    schema = FeatureSchema.from_parquet(shards[0])
    n_obs = sum(1 for n in schema.names if n.startswith("obs_"))
    n_hlth = sum(1 for n in schema.names if n.startswith("hlth_"))
    print(f"features            : {schema.n_features}  ({n_obs} obs_ + {n_hlth} hlth_)")
    print("leakage guard       : PASS")

    kinds = {t: sum(1 for x in schema.transforms if x == t) for t in ("standardise", "log1p", "binary")}
    print(f"transforms          : {kinds}")

    stats = TransformStats.fit_or_load(root, schema)
    ds = WindowDataset(
        root, split, window_s=window_s, stride_s=stride_s, schema=schema, stats=stats
    )
    idx = ds.index

    print(f"episodes            : {np.unique(idx.episode_id).size}")
    print(f"windows             : {len(ds):,}  ({window_s:g} s @ {stride_s:g} s stride)")
    print(f"dropped post-breach : {idx.n_dropped_post_breach:,}")
    print(f"dropped short       : {idx.n_dropped_short}")

    counts = np.bincount(idx.y_risk.astype(np.int64), minlength=len(RISK_CODES))
    share = counts / max(counts.sum(), 1)
    inv = {v: k for k, v in RISK_CODES.items()}
    print("risk class          : " + ", ".join(
        f"{inv[i]} {counts[i]:,} ({share[i]:.1%})" for i in range(len(RISK_CODES))
    ))
    print(f"censored ttc        : {idx.y_censored.mean():.1%}")

    w = ds[0]
    print(f"window shape        : x {w.x.shape}, mask {w.mask.shape}")
    print(f"observed fraction   : {w.mask.mean():.4f}  (window 0)")
    assert np.isfinite(w.x).all(), "non-finite value survived normalisation"
    print("finite after fill   : PASS")

    rng = np.random.default_rng(0)
    probe = rng.integers(0, len(ds), size=min(256, len(ds)))
    acc = np.stack([ds[int(i)].mask.mean(axis=0) for i in probe]).mean(axis=0)
    worst = np.argsort(acc)[:3]
    print("least-observed      : " + ", ".join(
        f"{schema.names[i]} {acc[i]:.3f}" for i in worst
    ))

    t1 = time.time()
    sampler = EpisodeGroupedSampler(ds, batch_size=64, subsample=0.02)
    n = 0
    for batch_ids in sampler:
        ds.batch(batch_ids)
        n += 1
        if n >= 20:
            break
    dt = time.time() - t1
    print(f"throughput          : {n * 64 / max(dt, 1e-9):,.0f} windows/s "
          f"(cache {ds.store.hits}/{ds.store.hits + ds.store.misses} hits)")
    print(f"total               : {time.time() - t0:.1f} s\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="RLEA M2.1 window loader")
    ap.add_argument("--data", default="data/raw", type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--window-s", default=60.0, type=float)
    ap.add_argument("--stride-s", default=1.0, type=float)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--refit-stats", action="store_true")
    args = ap.parse_args()

    if args.refit_stats:
        shards = sorted((args.data / "timeseries" / "split=train").glob("*.parquet"))
        schema = FeatureSchema.from_parquet(shards[0])
        TransformStats.fit_or_load(args.data, schema, refit=True)

    if args.verify or not args.refit_stats:
        verify(args.data, args.split, window_s=args.window_s, stride_s=args.stride_s)


if __name__ == "__main__":
    main()
