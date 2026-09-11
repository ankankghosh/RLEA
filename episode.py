r"""
Episode runner: one seed in, one labelled 10 Hz episode out.
============================================================

Module role in RLEA
-------------------
Owns the simulation loop. Everything upstream (`atmosphere`, `icing`,
`aircraft`, `transitions`, `degradation`) is a component; this module wires
them together, steps them in the correct order, applies the sensor layer, and
assembles the labelled row table. The batch generator (`generate.py`) is then
a thin loop over this function plus Parquet writing.

Consolidating the loop here matters for reproducibility: the child-RNG scheme,
step ordering, and label definitions are stated once, in one place, rather than
being retyped per experiment.

Determinism contract
--------------------
One integer seed fully determines an episode. Four **independent** child
generators are spawned via ``SeedSequence(seed).spawn(4)``:

===== =============================================
Index Subsystem
===== =============================================
0     Scenario construction (`transitions` builders)
1     Atmosphere (OU perturbations)
2     Icing (physics-family draw, shedding)
3     Aircraft (Dryden gusts, CG events)
4     Sensor degradation
===== =============================================

Five streams, not four: scenario construction was originally folded into the
atmosphere stream, but the builders consume a variable number of draws
depending on scenario type, which would have shifted every downstream stream as
a function of scenario. Independent streams mean adding a draw in one subsystem
cannot perturb another. (See ASSUMPTIONS A-METH-05.)

Step ordering
-------------
Order is load-bearing and fixed:

1. flight plan -> guidance command for time *t*
2. atmosphere sampled **at the aircraft's current position**
3. crew policy updated (may arm ice protection)
4. icing stepped -> ice state + aerodynamic degradation
5. aircraft stepped -> new kinematics, control effort, envelope margins
6. sensor suite applied to the clean observable set -> ``obs_*`` + ``hlth_*``
7. row assembled
8. atmosphere advanced (forcings, advection, OU)
9. clock advanced

The atmosphere is sampled *before* it is advanced so that all components in a
given row describe the same instant.

Label definitions (frozen 2026-09-04)
-------------------------------------
These are **design choices**, documented here and in METHODOLOGY. They are a
thin layer over continuous ground truth (`gt_stall_margin_deg`,
`gt_accretion_rate_mm_min`), which remains available for regression or for
re-binning downstream.

**`lbl_icing_severity`** — from instantaneous accretion rate, in the
operational vocabulary (none / trace / light / moderate / severe):

======== ==========================
Class    Accretion rate (mm/min)
======== ==========================
none     < 0.01
trace    0.01 – 0.10
light    0.10 – 0.60
moderate 0.60 – 2.00
severe   >= 2.00
======== ==========================

**`lbl_risk_class`** — the aircraft's current envelope state, mirroring the
Tier-1 monitor's classification (nominal / caution / warning / breach). Derived
from true stall and speed margins in `aircraft.py`.

**`lbl_time_to_critical_s`** — computed *backwards* after the episode
completes: seconds from the current sample until the first sample whose
`envelope_state` is `warning` or `breach`. Where no such sample occurs, the
value is `NaN` and `lbl_critical_censored` is 1. Censoring must be handled
explicitly by any downstream survival model; imputing a large finite value
would silently teach the model that "never" equals "eventually".

Post-breach validity
--------------------
Episodes are **not** terminated at envelope breach (ASSUMPTIONS A-CREW-05:
outcomes are results, not design targets). However, post-stall aerodynamics
are not modelled (A-AC-03), so every sample at or after the first breach
carries ``gt_post_breach = 1``. Those samples are outside the model's validity
domain and must be excluded from any claim about physical realism. A run is
terminated early only on ground contact, which is recorded as
``terminated_reason = "ground_contact"``.

Column conventions
------------------
``obs_`` post-degradation sensor values (model inputs) ·
``hlth_`` sensor health flags (model inputs) ·
``gt_`` simulator internals (never inputs) ·
``lbl_`` targets.

Physics-family identity and sampled physics parameters appear in the episode
**manifest only**, never in the row table (A-METH-04).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from rlea.simdata.aircraft import AircraftModel
from rlea.simdata.atmosphere import AtmosphereModel
from rlea.simdata.degradation import DEGRADATION_PROFILES, SensorSuite
from rlea.simdata.icing import PHYSICS_FAMILIES, IcingModel
from rlea.simdata.transitions import SCENARIO_BUILDERS, ScenarioSpec

__all__ = [
    "DT_S",
    "SEVERITY_BINS",
    "EpisodeResult",
    "CALIBRATION_MAX_THICKNESS_MM",
    "classify_icing_severity",
    "run_episode",
]

#: Simulation timestep, s. 10 Hz.
DT_S = 0.1

#: Frozen severity bin edges, mm/min. See module docstring.
SEVERITY_BINS: list[tuple[float, str]] = [
    (0.01, "none"),
    (0.10, "trace"),
    (0.60, "light"),
    (2.00, "moderate"),
    (float("inf"), "severe"),
]


#: Upper bound of the surrogate's empirical anchor, mm.
#: The thickness-to-penalty mapping is calibrated on the NASA TM-83564
#: natural-icing flights, which span 1.67-15.20 mm. Samples above this
#: extrapolate beyond any measurement (A-ICE-14) and are flagged, following
#: the gt_post_breach pattern (A-AC-03): integration continues, the region
#: is marked, and it is excluded from claims about physical realism.
CALIBRATION_MAX_THICKNESS_MM = 15.20


def classify_icing_severity(rate_mm_min: float) -> str:
    """Map instantaneous accretion rate (mm/min) to the severity vocabulary."""
    for edge, label in SEVERITY_BINS:
        if rate_mm_min < edge:
            return label
    return "severe"


@dataclass
class EpisodeResult:
    """One completed episode: row table plus manifest metadata.

    Attributes
    ----------
    rows : list of dict
        One dict per 10 Hz sample, with `obs_`/`hlth_`/`gt_`/`lbl_` keys plus
        the index columns ``episode_id``, ``step``, ``t_s``, ``phase``.
    manifest : dict
        Episode-level record: seed, scenario, sampled physics parameters,
        degradation profile, outcome summary, split tag. This is the only
        place physics-family identity is stored.
    """

    rows: list[dict[str, Any]]
    manifest: dict[str, Any]

    @property
    def n_steps(self) -> int:
        return len(self.rows)


def run_episode(
    *,
    episode_id: int,
    seed: int,
    use_case: str = "UC3",
    physics_family: str = "A",
    degradation_profile: str = "nominal",
    split: str = "train",
    scenario_kwargs: dict[str, Any] | None = None,
    max_duration_s: float | None = None,
) -> EpisodeResult:
    r"""Run one full episode and return its rows and manifest.

    Parameters
    ----------
    episode_id : int
        Identifier written into every row; the join key to the manifest.
    seed : int
        Master seed. Fully determines the episode.
    use_case : str
        Key into :data:`~rlea.simdata.transitions.SCENARIO_BUILDERS`
        ("UC1", "UC2", "UC3", "NUIS").
    physics_family : str
        Key into :data:`~rlea.simdata.icing.PHYSICS_FAMILIES`. "A" is
        in-distribution; "B"/"C" form the physics-novelty axis.
    degradation_profile : str
        Key into :data:`~rlea.simdata.degradation.DEGRADATION_PROFILES`.
        "nominal"/"degraded" are in-distribution; "severe" is held out.
    split : str
        Split tag recorded in the manifest ("train", "val", "test_iid",
        "test_physics", ...). Assignment is the caller's responsibility;
        this function only records it.
    scenario_kwargs : dict, optional
        Passed through to the scenario builder (e.g. ``regime="rapid"``).
    max_duration_s : float, optional
        Cap on simulated duration, for fast tests. Production runs leave this
        as None and use the scenario's own sampled duration.

    Returns
    -------
    EpisodeResult
    """
    # -- five independent child streams (see module docstring) ------------
    ss = np.random.SeedSequence(seed).spawn(5)
    rng_scn, rng_atm, rng_ice, rng_ac, rng_deg = (np.random.default_rng(s) for s in ss)

    # -- construct scenario ------------------------------------------------
    builder = SCENARIO_BUILDERS[use_case]
    spec: ScenarioSpec = builder(rng_scn, **(scenario_kwargs or {}))

    duration_s = spec.duration_s
    if max_duration_s is not None:
        duration_s = min(duration_s, max_duration_s)

    # -- instantiate components -------------------------------------------
    atm = AtmosphereModel(spec.atmosphere_config, rng=rng_atm)
    spec.apply_to(atm)

    params = PHYSICS_FAMILIES[physics_family].sample(rng_ice)
    ice = IcingModel(params, spec.protection, rng=rng_ice)

    ac = AircraftModel(spec.aircraft_config, spec.nuisance, rng=rng_ac)
    ac.initialise(
        altitude_m=spec.flight_plan.initial_altitude_m,
        true_airspeed_ms=spec.flight_plan.initial_airspeed_ms,
        heading_rad=spec.flight_plan.heading_rad,
    )

    profile = DEGRADATION_PROFILES[degradation_profile]
    sensors = SensorSuite(profile, rng_deg, dt_s=DT_S)

    # UC-1 carries a ground-contamination initial condition (residual ice at
    # brake release). Applied here rather than in the builder, because only the
    # runner owns simulator state.
    ground_contam_mm = float(spec.metadata.get("ground_contamination_mm", 0.0) or 0.0)
    if ground_contam_mm > 0.0:
        ice.ice_thickness_m = ground_contam_mm * 1e-3
        ice.ice_mass_per_area_kg_m2 = ice.ice_thickness_m * 917.0
        ice.total_accreted_mass_kg_m2 = ice.ice_mass_per_area_kg_m2

    # -- main loop ---------------------------------------------------------
    rows: list[dict[str, Any]] = []
    n_steps = int(round(duration_s / DT_S))
    t = 0.0
    post_breach = False
    first_breach_step: int | None = None
    terminated_reason = "completed"

    for step in range(n_steps):
        cmd = spec.flight_plan.command_at(t)

        atm_state = atm.sample(
            ac.x_m, ac.y_m, ac.altitude_m, true_airspeed_ms=ac.tas_ms
        )

        spec.crew_policy.update(
            time_s=t,
            tat_c=atm_state.tat_c,
            in_cloud=atm_state.in_cloud,
            protection=spec.protection,
        )

        ice_state, aero = ice.step(atm_state, ac.tas_ms, DT_S)
        ac_state = ac.step(atm_state, aero, cmd, DT_S)

        if ac_state.envelope_breach and first_breach_step is None:
            first_breach_step = step
        if first_breach_step is not None:
            post_breach = True

        # -- clean observable set, keyed to SENSOR_SPECS ------------------
        clean = {
            "ias_ms": ac_state.indicated_airspeed_ms,
            "tas_ms": ac_state.true_airspeed_ms,
            "mach": ac_state.mach,
            "alt_baro_m": ac_state.altitude_m,
            "vs_ms": ac_state.vertical_speed_ms,
            "static_pressure_pa": atm_state.pressure_pa,
            # Redundant vanes see the same true AoA; they diverge only through
            # independent sensor faults (A-SEN-06).
            "aoa_l_deg": ac_state.alpha_deg,
            "aoa_r_deg": ac_state.alpha_deg,
            "pitch_deg": ac_state.pitch_deg,
            "roll_deg": ac_state.bank_deg,
            "heading_deg": ac_state.heading_deg,
            "pitch_rate_dps": ac_state.pitch_rate_dps,
            "nz_g": ac_state.load_factor_g,
            "gs_ms": ac_state.ground_speed_ms,
            "tat_c": atm_state.tat_c,
            "sat_c": atm_state.oat_c,
            "elevator_deg": ac_state.elevator_deg,
            "elevator_rate_dps": ac_state.elevator_rate_dps,
            "torque_pct": ac_state.torque_pct,
            "fuel_flow_kgs": ac_state.fuel_flow_kg_s,
            # The ice detector observes accretion rate, laggily and noisily.
            "ice_detector_rate_mmmin": ice_state.thickness_rate_mm_min,
            "ice_detector_active": 1.0 if ice_state.thickness_rate_mm_min > 0.05 else 0.0,
            "wind_u_ms": atm_state.wind_u_ms,
            "wind_v_ms": atm_state.wind_v_ms,
            "wind_shear_est_per_s": atm_state.wind_shear_per_s,
        }
        obs, health = sensors.step(clean, ice_thickness_mm=ice_state.ice_thickness_mm)

        row: dict[str, Any] = {
            "episode_id": episode_id,
            "step": step,
            "t_s": step * DT_S,          # derived, never accumulated
            "phase": cmd.phase,
        }
        row.update({f"obs_{k}": v for k, v in obs.items()})
        row.update(health)  # already hlth_-prefixed

        # Switch positions are known to the crew, hence observable.
        row["obs_antiice_on"] = float(spec.protection.antiice_on)
        row["obs_boots_on"] = float(spec.protection.boots_on)
        row["obs_flap_deg"] = ac_state.flap_deg
        row["obs_gear_down"] = float(ac_state.gear_down)

        # -- ground truth: icing ------------------------------------------
        row.update({
            "gt_ice_thickness_mm": ice_state.ice_thickness_mm,
            "gt_ice_mass_kg_m2": ice_state.ice_mass_per_area_kg_m2,
            "gt_accretion_rate_mm_min": ice_state.thickness_rate_mm_min,
            "gt_collection_efficiency": ice_state.collection_efficiency,
            "gt_freezing_fraction": ice_state.freezing_fraction,
            "gt_ice_density_kg_m3": ice_state.ice_density_kg_m3,
            "gt_ice_type": ice_state.ice_type,
            "gt_shape_factor": aero.shape_factor,
            "gt_delta_cd": aero.delta_cd,
            "gt_delta_cl_max": aero.delta_cl_max,
            "gt_delta_alpha_stall_deg": aero.delta_alpha_stall_deg,
            "gt_delta_cm": aero.delta_cm,
            "gt_ridge_height_mm": ice_state.ridge_height_m * 1e3,
            "gt_ridge_active": float(ice_state.ridge_active),
            "gt_ice_regime": ice_state.ice_regime,
            "gt_boot_fired": float(ice_state.boot_fired),
            "gt_shed_event": float(ice_state.shed_event),
            "gt_antiice_active": float(ice_state.antiice_active),
        })

        # -- ground truth: environment ------------------------------------
        row.update({
            "gt_lwc_g_m3": atm_state.lwc_g_m3,
            "gt_slw_g_m3": atm_state.supercooled_lwc_g_m3,
            "gt_mvd_um": atm_state.mvd_um,
            "gt_oat_c": atm_state.oat_c,
            "gt_dewpoint_c": atm_state.dewpoint_c,
            "gt_rh": atm_state.relative_humidity,
            "gt_density_kg_m3": atm_state.density_kg_m3,
            "gt_freezing_level_m": atm_state.freezing_level_m,
            "gt_in_cloud": float(atm_state.in_cloud),
            "gt_sld_flag": float(atm_state.sld_flag),
        })

        # -- ground truth: envelope ---------------------------------------
        row.update({
            "gt_alpha_deg": ac_state.alpha_deg,
            "gt_alpha_stall_eff_deg": ac_state.alpha_stall_eff_deg,
            "gt_cl_max_eff": ac_state.cl_max_eff,
            "gt_stall_margin_deg": ac_state.stall_margin_deg,
            "gt_stall_speed_ms": ac_state.stall_speed_ms,
            "gt_speed_margin_ms": ac_state.speed_margin_ms,
            "gt_thrust_margin_n": ac_state.thrust_margin_n,
            "gt_envelope_state": ac_state.envelope_state,
            "gt_envelope_breach": float(ac_state.envelope_breach),
            "gt_cl_limited": float(ac_state.cl_limited),
            "gt_post_breach": float(post_breach),
            "gt_altitude_m": ac_state.altitude_m,
            "gt_tas_ms": ac_state.true_airspeed_ms,
        })
        row["gt_beyond_calibration"] = float(
            row["gt_ice_thickness_mm"] > CALIBRATION_MAX_THICKNESS_MM
        )

        # -- ground truth: nuisance sources (false-positive attribution) ---
        row.update({
            "gt_mass_kg": ac_state.mass_kg,
            "gt_cg_pct_mac": ac_state.cg_pct_mac,
            "gt_gust_u_ms": ac_state.gust_u_ms,
            "gt_gust_w_ms": ac_state.gust_w_ms,
            "gt_gust_alpha_perturb_deg": math.degrees(ac_state.gust_alpha_perturb_rad),
            "gt_friction_offset_deg": ac_state.friction_offset_deg,
            "gt_cg_shift_event": float(ac_state.cg_shift_event),
            "gt_config_change_active": float(ac_state.config_change_active),
        })

        # -- labels (time_to_critical filled in after the loop) ------------
        row["lbl_icing_severity"] = classify_icing_severity(
            ice_state.thickness_rate_mm_min
        )
        row["lbl_risk_class"] = ac_state.envelope_state
        row["lbl_envelope_breach"] = float(ac_state.envelope_breach)

        rows.append(row)

        atm.step(DT_S)
        t += DT_S

        if ac.altitude_m <= 0.0:
            terminated_reason = "ground_contact"
            break

    # -- backward pass: time to first critical sample ---------------------
    critical_steps = [
        i for i, r in enumerate(rows)
        if r["gt_envelope_state"] in ("warning", "breach")
    ]
    next_critical = float("inf")
    ttc: list[float] = [float("nan")] * len(rows)
    censored: list[float] = [1.0] * len(rows)
    crit_set = set(critical_steps)
    for i in range(len(rows) - 1, -1, -1):
        if i in crit_set:
            next_critical = i
        if math.isfinite(next_critical):
            ttc[i] = (next_critical - i) * DT_S
            censored[i] = 0.0
    for i, r in enumerate(rows):
        r["lbl_time_to_critical_s"] = ttc[i]
        r["lbl_critical_censored"] = censored[i]

    # -- manifest ----------------------------------------------------------
    max_ice = max((r["gt_ice_thickness_mm"] for r in rows), default=0.0)
    min_margin = min((r["gt_stall_margin_deg"] for r in rows), default=float("nan"))
    max_slw = max((r["gt_slw_g_m3"] for r in rows), default=0.0)
    exposure = (
        sum(1 for r in rows if r["gt_slw_g_m3"] > 0.01) / len(rows) if rows else 0.0
    )

    manifest: dict[str, Any] = {
        "episode_id": episode_id,
        "seed": seed,
        "use_case": use_case,
        "scenario_id": spec.scenario_id,
        "split": split,
        "physics_family": physics_family,
        "degradation_profile": degradation_profile,
        "n_steps": len(rows),
        "duration_s": len(rows) * DT_S,
        "terminated_reason": terminated_reason,
        # outcome summary
        "max_ice_mm": max_ice,
        "beyond_calibration": float(max_ice > CALIBRATION_MAX_THICKNESS_MM),
        "min_stall_margin_deg": min_margin,
        "max_slw_g_m3": max_slw,
        "max_ridge_height_mm": max(
            (r["gt_ridge_height_mm"] for r in rows), default=0.0
        ),
        "ridge_occurred": float(
            any(r["gt_ridge_active"] > 0.0 for r in rows)
        ),
        "exposure_fraction": exposure,
        "breach_occurred": float(first_breach_step is not None),
        "time_to_first_breach_s": (
            first_breach_step * DT_S if first_breach_step is not None else float("nan")
        ),
        # crew / equipment state
        "crew_never_reacts": float(spec.crew_policy.never_reacts),
        "crew_reaction_delay_s": spec.crew_policy.reaction_delay_s,
        "antiice_available": float(spec.protection.antiice_available),
        "boots_available": float(spec.protection.boots_available),
        "aoa_channel_faulted": float(sensors.faulty_aoa_channel is not None),
    }
    # Sampled physics parameters: manifest ONLY, never the row table.
    manifest.update({f"phys_{k}": v for k, v in params.as_dict().items()})
    if spec.onset is not None:
        manifest.update(spec.onset.as_dict())
    manifest.update({f"scn_{k}": v for k, v in spec.metadata.items()})

    return EpisodeResult(rows=rows, manifest=manifest)
