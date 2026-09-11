#!/usr/bin/env python3
"""Patch generate.py for the flight-profile axis and the no-danger fix.

Run from the repo root:

    python apply_patch.py

Idempotent: re-running detects the patch is already applied and exits cleanly.
Writes generate.py.bak before touching anything.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET = Path("generate.py")


def fail(msg: str) -> None:
    print(f"FAILED: {msg}")
    print("No changes written.")
    sys.exit(1)


def main() -> int:
    if not TARGET.exists():
        fail(f"{TARGET} not found. Run this from the repo root.")

    s = TARGET.read_text()

    if "flight_profiles" in s:
        print("Already patched. Nothing to do.")
        return 0

    edits: list[tuple[str, str, str]] = []

    # -- 1. flight profiles on SplitSpec ---------------------------------
    edits.append((
        "SplitSpec.flight_profiles",
        """    onset_regimes: tuple[str, ...] = ("gradual", "moderate", "rapid")
    novelty_axes: tuple[str, ...] = ()""",
        """    onset_regimes: tuple[str, ...] = ("gradual", "moderate", "rapid")
    flight_profiles: tuple[str, ...] = ("cruise", "approach", "hold", "climb")
    novelty_axes: tuple[str, ...] = ()""",
    ))

    # -- 2. actually generate UC-1 and UC-2 -------------------------------
    # These builders have existed since M1.4 and have never once been called:
    # every split took the ("UC3",) default, so the dataset has only ever
    # contained UC3_* and NUIS_* episodes.
    edits.append((
        "use_cases default",
        '''    use_cases: tuple[str, ...] = ("UC3",)''',
        '''    use_cases: tuple[str, ...] = ("UC1", "UC2", "UC3")''',
    ))

    # -- 3. sample the profile and pass it through ------------------------
    edits.append((
        "profile sampling",
        """        regime = str(chooser.choice(spec.onset_regimes))

        # NUIS builder takes no regime argument.
        scenario_kwargs: dict[str, Any] = {}
        if use_case in ("UC1", "UC2", "UC3"):
            scenario_kwargs["regime"] = regime""",
        """        regime = str(chooser.choice(spec.onset_regimes))
        profile = str(chooser.choice(spec.flight_profiles))

        # NUIS builder takes neither a regime nor a flight profile.
        scenario_kwargs: dict[str, Any] = {}
        if use_case in ("UC1", "UC2", "UC3"):
            scenario_kwargs["regime"] = regime
            scenario_kwargs["flight_profile"] = profile""",
    ))

    edits.append((
        "manifest profile record",
        """        m["onset_regime_requested"] = regime""",
        """        m["onset_regime_requested"] = regime
        m["flight_profile_requested"] = profile if use_case != "NUIS" else "cruise\"""",
    ))

    # -- 4. refuse to write into a populated directory --------------------
    # generate.py writes part-NNNN.parquet with mkdir(exist_ok=True) and no
    # clobber check. If a rerun produces fewer shards for a split, the surplus
    # old files SURVIVE, leaving shards of two different schemas in one
    # directory. Since data/ is gitignored, the old dataset is unrecoverable.
    edits.append((
        "overwrite guard",
        """    args.out.mkdir(parents=True, exist_ok=True)
    all_manifests: list[dict[str, Any]] = []""",
        """    if (args.out / "episodes.parquet").exists() and not args.force:
        print(
            f"\\n  REFUSING TO WRITE: {args.out}/ already holds a dataset.\\n"
            "  Shards are written by name with no clobber check, so a rerun\\n"
            "  that produces fewer shards for any split leaves stale files\\n"
            "  behind and mixes two schemas in one directory.\\n"
            "  Use a new --out directory, or pass --force if you mean it."
        )
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    all_manifests: list[dict[str, Any]] = []""",
    ))

    edits.append((
        "--force flag",
        """    ap.add_argument("--dry-run", action="store_true",""",
        """    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing dataset in --out")
    ap.add_argument("--dry-run", action="store_true",""",
    ))

    # -- 5. show profiles in the plan printout ---------------------------
    edits.append((
        "plan printout",
        """            f"{','.join(s.degradation_profiles):<20}{','.join(s.novelty_axes) or '-'}\"""",
        """            f"{','.join(s.degradation_profiles):<20}"
            f"{','.join(p[:4] for p in s.flight_profiles):<24}"
            f"{','.join(s.novelty_axes) or '-'}\"""",
    ))

    for name, old, new in edits:
        n = s.count(old)
        if n != 1:
            fail(f"anchor {name!r} matched {n} times, expected 1")
        s = s.replace(old, new)

    shutil.copy(TARGET, TARGET.with_suffix(".py.bak"))
    TARGET.write_text(s)

    compile(s, str(TARGET), "exec")
    print(f"Patched {TARGET} ({len(edits)} edits). Backup: {TARGET}.bak")
    print("\nChanged:")
    print("  - every split now draws from UC1/UC2/UC3, not UC3 alone")
    print("  - flight_profile sampled per episode: cruise/approach/hold/climb")
    print("  - refuses to overwrite a populated --out unless --force")
    return 0


if __name__ == "__main__":
    sys.exit(main())
