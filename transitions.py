r"""
Scenario library: post-takeoff environmental transitions for UC-1/2/3.
=====================================================================

Module role in RLEA
-------------------
Turns sampled parameters into a complete, runnable episode specification: the
initial atmosphere, the weather cells, the *scripted forcings* that make the
environment evolve, the flight plan the crew is following, and the crew's ice
protection policy.

This module is where **onset rate becomes a controlled experimental variable**.
The framework's trajectory-divergence term :math:`T(s_i, s_j)` and the whole
UC-3 argument rest on the claim that two states with identical instantaneous
values but different derivatives are operationally different. That claim is
only testable if onset rate is a knob, held out in its own split, rather than
an incidental consequence of the weather sampling.

Scenario taxonomy
-----------------

============  ==================================================  ==================
Scenario      Operational question                                Dominant driver
============  ==================================================  ==================
``UC1_*``     Is departure into this weather advisable?           Surface/low-level
``UC2_*``     Crew profile vs. deteriorating conditions           Altitude choice
``UC3_*``     Rapid in-flight icing onset -- act now?             Onset rate
``NUIS_*``    (hard negative) Disturbance without any icing       Nuisance sources
============  ==================================================  ==================

**UC-1 -- takeoff readiness.** Departure into a shallow but active icing layer
in the climb. Ground contamination is represented through the initial-condition
penalty (residual accretion at brake release) rather than through a ground-roll
model. The operational question is whether the climb through the layer is
survivable given the aircraft's protection state.

**UC-2 -- pilot/system disagreement.** The crew flies the profile *they* chose
(typically a cruise altitude selected pre-departure from a forecast) while the
actual conditions at that altitude deteriorate and a better altitude exists
above or below. The scenario deliberately does not force a resolution: it
produces the divergence between crew intent and environmental reality that the
disagreement protocol has to reason about. Ground truth marks the window during
which a better altitude was available, so a recommendation can be scored.

**UC-3 -- rapid in-flight icing.** The core scenario. A parameterised, jointly
evolving transition:

.. math::

    T_{sl}(t) = T_{sl}(0) - \dot{T}\,(t - t_0),
    \qquad
    w_{peak}(t) = w_0 + \dot{w}\,(t - t_0),
    \qquad
    d_{MVD}(t) = d_0 + \dot{d}\,(t - t_0)

together with a descending freezing level,

.. math::

    z_{fl}(t) = z_{fl}(0) - \dot{z}_{fl}\,(t - t_0)

implemented by raising the lapse rate rather than by moving a level directly
(the freezing level is a *derived* quantity in ``atmosphere.py`` and must stay
derived, or the thermal profile and the freezing level would disagree).

Onset-rate parameterisation
---------------------------
Three named onset regimes span the same terminal conditions at different rates.
This is what makes the **onset-novelty split** possible: train on ``gradual``
and ``moderate``, evaluate on ``rapid``. A model that has learned instantaneous
state alone will transfer; a model that has correctly learned rate structure
should show a measurable difference. Either outcome is an informative result.

===========  ====================  ==================  ====================
Regime       dT/dt (degC/min)      dLWC/dt (g/m3/min)  Freezing level (m/min)
===========  ====================  ==================  ====================
``gradual``  0.10 - 0.30           0.02 - 0.08         10 - 40
``moderate`` 0.30 - 0.80           0.08 - 0.20         40 - 110
``rapid``    0.80 - 2.50           0.20 - 0.60         110 - 320
===========  ====================  ==================  ====================

Assumptions and limitations
---------------------------
* Forcings are open-loop functions of time: the environment does not respond to
  the aircraft.
* Flight plans are piecewise-constant altitude/speed targets, not a full FMS.
* The crew ice-protection policy is a simple reactive rule (see
  :class:`CrewPolicy`); modelling crew decision-making properly is out of scope
  and would in any case pre-empt the question the framework exists to study.
* Terrain, airspace and traffic constraints are absent; the "descend to warmer
  air" option is always assumed available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from rlea.simdata.aircraft import AircraftConfig, FlightCommand, NuisanceConfig
from rlea.simdata.airframes import DHC6_CONFIG
from rlea.simdata.atmosphere import (
    T_FREEZE,
    AtmosphereConfig,
    AtmosphereModel,
    WeatherCell,
)
from rlea.simdata.icing import IceProtectionConfig

__all__ = [
    "OnsetRegime",
    "UseCase",
    "OnsetParameters",
    "CrewPolicy",
    "FlightPlan",
    "ScenarioSpec",
    "ONSET_REGIMES",
    "sample_onset",
    "build_uc1",
    "build_uc2",
    "build_uc3",
    "build_nuisance_only",
    "SCENARIO_BUILDERS",
]

OnsetRegime = Literal["gradual", "moderate", "rapid"]
UseCase = Literal["UC1", "UC2", "UC3", "UC4", "NUIS"]


#: Onset-rate regime bounds. Units: degC/min, g m^-3 /min, m/min, um/min.
#: Tabulated in ``METHODOLOGY.md`` for the paper's experimental setup.
ONSET_REGIMES: dict[str, dict[str, tuple[float, float]]] = {
    "gradual": {
        "temp_drop_c_per_min": (0.10, 0.30),
        "lwc_rate_g_m3_per_min": (0.02, 0.08),
        "freezing_level_rate_m_per_min": (10.0, 40.0),
        "mvd_rate_um_per_min": (0.05, 0.30),
    },
    "moderate": {
        "temp_drop_c_per_min": (0.30, 0.80),
        "lwc_rate_g_m3_per_min": (0.08, 0.20),
        "freezing_level_rate_m_per_min": (40.0, 110.0),
        "mvd_rate_um_per_min": (0.30, 0.90),
    },
    "rapid": {
        "temp_drop_c_per_min": (0.80, 2.50),
        "lwc_rate_g_m3_per_min": (0.20, 0.60),
        "freezing_level_rate_m_per_min": (110.0, 320.0),
        "mvd_rate_um_per_min": (0.90, 2.60),
    },
}


@dataclass(frozen=True)
class OnsetParameters:
    r"""Rates governing how fast the environment deteriorates.

    Attributes
    ----------
    regime : str
        Named onset regime, one of ``ONSET_REGIMES``.
    onset_time_s : float
        Time at which deterioration begins, s.
    temp_drop_c_per_min : float
        :math:`\dot{T}`, degC per minute, applied to the sea-level temperature
        (and hence to the whole column).
    lwc_rate_g_m3_per_min : float
        :math:`\dot{w}`, growth rate of the active cell's peak LWC.
    freezing_level_rate_m_per_min : float
        :math:`\dot{z}_{fl}`, descent rate of the freezing level, implemented
        via lapse-rate steepening.
    mvd_rate_um_per_min : float
        :math:`\dot{d}`, droplet growth rate. Drives the transition toward
        SLD conditions.
    duration_s : float
        How long the forcing remains active, s. After this it holds.
    """

    regime: str
    onset_time_s: float
    temp_drop_c_per_min: float
    lwc_rate_g_m3_per_min: float
    freezing_level_rate_m_per_min: float
    mvd_rate_um_per_min: float
    duration_s: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "onset_regime": self.regime,
            "onset_time_s": self.onset_time_s,
            "temp_drop_c_per_min": self.temp_drop_c_per_min,
            "lwc_rate_g_m3_per_min": self.lwc_rate_g_m3_per_min,
            "freezing_level_rate_m_per_min": self.freezing_level_rate_m_per_min,
            "mvd_rate_um_per_min": self.mvd_rate_um_per_min,
            "onset_duration_s": self.duration_s,
        }


def sample_onset(
    regime: OnsetRegime,
    rng: np.random.Generator,
    *,
    onset_time_s: float | None = None,
    duration_s: float = 900.0,
) -> OnsetParameters:
    """Draw onset rates uniformly from a named regime's bounds."""
    bounds = ONSET_REGIMES[regime]
    return OnsetParameters(
        regime=regime,
        onset_time_s=(
            onset_time_s if onset_time_s is not None else float(rng.uniform(120.0, 420.0))
        ),
        temp_drop_c_per_min=float(rng.uniform(*bounds["temp_drop_c_per_min"])),
        lwc_rate_g_m3_per_min=float(rng.uniform(*bounds["lwc_rate_g_m3_per_min"])),
        freezing_level_rate_m_per_min=float(
            rng.uniform(*bounds["freezing_level_rate_m_per_min"])
        ),
        mvd_rate_um_per_min=float(rng.uniform(*bounds["mvd_rate_um_per_min"])),
        duration_s=duration_s,
    )


# ---------------------------------------------------------------------------
# Crew behaviour
# ---------------------------------------------------------------------------


@dataclass
class CrewPolicy:
    r"""Reactive crew ice-protection policy.

    A deliberately simple rule, so that crew behaviour is a *documented
    baseline* rather than a confound: anti-ice and boots are armed once the
    total air temperature falls below a threshold while the aircraft is in
    visible moisture, after a reaction delay drawn per episode.

    The reaction delay matters. An instantaneous-response crew would suppress
    exactly the accretion transient the model is supposed to detect; a delay of
    tens of seconds is both realistic and preserves the signal.

    Attributes
    ----------
    arm_tat_threshold_c : float
        TAT below which protection is considered, degC.
    reaction_delay_s : float
        Delay between the trigger condition being met and action, s.
    uses_boots : bool
        Whether the crew cycles de-ice boots.
    uses_antiice : bool
        Whether the crew selects thermal anti-ice.
    never_reacts : bool
        If True the crew takes no protective action at all. Used to generate
        the severe tail of the outcome distribution without having to force
        implausible weather.
    """

    arm_tat_threshold_c: float = 5.0
    reaction_delay_s: float = 45.0
    uses_boots: bool = True
    uses_antiice: bool = True
    never_reacts: bool = False

    _trigger_time_s: float | None = field(default=None, repr=False)
    _armed: bool = field(default=False, repr=False)

    def update(
        self,
        *,
        time_s: float,
        tat_c: float,
        in_cloud: bool,
        protection: IceProtectionConfig,
    ) -> None:
        """Advance the policy and mutate ``protection`` in place."""
        if self.never_reacts or self._armed:
            return
        condition = (tat_c <= self.arm_tat_threshold_c) and in_cloud
        if condition and self._trigger_time_s is None:
            self._trigger_time_s = time_s
        if not condition:
            self._trigger_time_s = None
            return
        if time_s - self._trigger_time_s >= self.reaction_delay_s:
            if self.uses_antiice and protection.antiice_available:
                protection.antiice_on = True
            if self.uses_boots and protection.boots_available:
                protection.boots_on = True
            self._armed = True


# ---------------------------------------------------------------------------
# Flight plan
# ---------------------------------------------------------------------------


@dataclass
class FlightPlan:
    """Piecewise-constant guidance schedule.

    Each segment is ``(start_time_s, altitude_m, airspeed_ms, flap_deg,
    gear_down, phase)``. The active segment is the last one whose start time
    has passed. Altitude targets are *commanded*, not achieved -- whether the
    aircraft can hold them under accumulating ice is an outcome, not an input.
    """

    segments: list[tuple[float, float, float, float, bool, str]]
    heading_rad: float = 0.0
    speedbrake_pct: float = 0.0

    def command_at(self, t_s: float) -> FlightCommand:
        """Return the active :class:`FlightCommand` at time ``t_s``."""
        active = self.segments[0]
        for seg in self.segments:
            if t_s >= seg[0]:
                active = seg
            else:
                break
        _, alt, spd, flap, gear, phase = active
        return FlightCommand(
            altitude_cmd_m=alt,
            airspeed_cmd_ms=spd,
            heading_cmd_rad=self.heading_rad,
            flap_deg=flap,
            gear_down=gear,
            speedbrake_pct=self.speedbrake_pct,
            phase=phase,
        )

    @property
    def initial_altitude_m(self) -> float:
        return self.segments[0][1]

    @property
    def initial_airspeed_ms(self) -> float:
        return self.segments[0][2]


# ---------------------------------------------------------------------------
# Scenario specification
# ---------------------------------------------------------------------------


@dataclass
class ScenarioSpec:
    """A complete, runnable episode definition.

    Everything the episode runner needs, and nothing it does not. Metadata is
    carried through to ``episodes.parquet`` so that any episode can be
    reconstructed from its manifest row plus its seed.
    """

    scenario_id: str
    use_case: UseCase
    description: str
    duration_s: float
    atmosphere_config: AtmosphereConfig
    cells: list[WeatherCell]
    forcings: list[Callable[[AtmosphereModel, float, float], None]]
    flight_plan: FlightPlan
    crew_policy: CrewPolicy
    protection: IceProtectionConfig
    nuisance: NuisanceConfig
    aircraft_config: AircraftConfig
    onset: OnsetParameters | None
    metadata: dict[str, float | str | bool] = field(default_factory=dict)

    def apply_to(self, atmosphere: AtmosphereModel) -> None:
        """Install this scenario's cells and forcings on an atmosphere model."""
        for cell in self.cells:
            atmosphere.add_cell(cell)
        for fn in self.forcings:
            atmosphere.add_forcing(fn)


# ---------------------------------------------------------------------------
# Forcing factories
# ---------------------------------------------------------------------------


def _make_cooling_forcing(onset: OnsetParameters):
    r"""Sea-level temperature ramp: :math:`T_{sl}(t) = T_{sl}(0) - \dot{T}(t-t_0)`.

    Applied as an incremental decrement each step so it composes with any other
    forcing that also writes ``sea_level_temperature_k``.
    """

    def forcing(model: AtmosphereModel, t_s: float, dt_s: float) -> None:
        if t_s < onset.onset_time_s:
            return
        if t_s > onset.onset_time_s + onset.duration_s:
            return
        model.config.sea_level_temperature_k -= (
            onset.temp_drop_c_per_min / 60.0
        ) * dt_s

    return forcing


def _make_freezing_level_forcing(onset: OnsetParameters, initial_fl_m: float):
    r"""Lower the freezing level by steepening the lapse rate.

    The freezing level in ``atmosphere.py`` is derived from the thermal
    profile, so it must be moved *through* the profile. For the linear part of
    the profile, :math:`z_{fl} = (T_{sl} - 273.15)/\Gamma`, so a target
    freezing level :math:`z^*` implies

    .. math::
        \Gamma^* = \frac{T_{sl} - 273.15}{z^*}

    which is applied with clipping to a physically plausible band
    [3.0, 12.0] degC/km. Combined with the cooling forcing this produces a
    freezing level that descends faster than either mechanism alone -- the
    realistic case.
    """

    def forcing(model: AtmosphereModel, t_s: float, dt_s: float) -> None:
        if t_s < onset.onset_time_s:
            return
        elapsed = min(t_s - onset.onset_time_s, onset.duration_s)
        target_fl = initial_fl_m - (onset.freezing_level_rate_m_per_min / 60.0) * elapsed
        target_fl = max(target_fl, 150.0)
        excess_k = model.config.sea_level_temperature_k - T_FREEZE
        if excess_k <= 0.1:
            return  # already sub-freezing at the surface; nothing to lower
        gamma = excess_k / target_fl
        model.config.lapse_rate_k_per_m = float(np.clip(gamma, 3.0e-3, 12.0e-3))

    return forcing


def _make_cell_growth_forcing(cell: WeatherCell, onset: OnsetParameters, max_lwc_g_m3: float):
    r"""Ramp a cell's peak LWC and MVD.

    .. math::
        w_{peak}(t) = \min\!\left(w_0 + \dot{w}(t - t_0),\, w_{max}\right),
        \qquad
        d(t) = d_0 + \dot{d}(t - t_0)

    MVD growth is what carries an episode from small-droplet stratiform icing
    into the SLD regime, where the collection-efficiency nonlinearity in
    ``icing.py`` produces a sharp change in accretion behaviour.
    """

    def forcing(model: AtmosphereModel, t_s: float, dt_s: float) -> None:
        if t_s < onset.onset_time_s:
            return
        if t_s > onset.onset_time_s + onset.duration_s:
            return
        cell.peak_lwc_kg_m3 = min(
            cell.peak_lwc_kg_m3 + (onset.lwc_rate_g_m3_per_min * 1e-3 / 60.0) * dt_s,
            max_lwc_g_m3 * 1e-3,
        )
        cell.mvd_m = min(
            cell.mvd_m + (onset.mvd_rate_um_per_min * 1e-6 / 60.0) * dt_s,
            120e-6,
        )

    return forcing


def _make_cell_descent_forcing(cell: WeatherCell, rate_m_per_min: float, floor_m: float):
    """Lower a cell's centre altitude, so the icing layer descends onto the aircraft."""

    def forcing(model: AtmosphereModel, t_s: float, dt_s: float) -> None:
        cell.z_m = max(cell.z_m - (rate_m_per_min / 60.0) * dt_s, floor_m)

    return forcing


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def build_uc1(
    rng: np.random.Generator,
    *,
    regime: OnsetRegime = "moderate",
    duration_s: float = 1200.0,
) -> ScenarioSpec:
    """UC-1: departure and climb through an active low-level icing layer.

    The decision under study is takeoff readiness, so the episode begins on the
    runway with a residual ground-contamination penalty (expressed as an
    initial accretion offset in metadata, applied by the runner) and climbs
    through a layer whose severity is still evolving.
    """
    surface_temp_c = float(rng.uniform(-4.0, 2.5))
    layer_base = float(rng.uniform(400.0, 1200.0))
    layer_thickness = float(rng.uniform(600.0, 1600.0))
    lwc0 = float(rng.uniform(0.15, 0.55))
    mvd0 = float(rng.uniform(14.0, 30.0))
    cruise_alt = float(rng.uniform(3000.0, 4600.0))

    atmosphere = AtmosphereConfig(
        sea_level_temperature_k=T_FREEZE + surface_temp_c,
        lapse_rate_k_per_m=float(rng.uniform(5.0e-3, 7.5e-3)),
        inversion_strength_k=float(rng.uniform(0.0, 2.5)),
        inversion_altitude_m=layer_base + layer_thickness * 0.5,
        surface_dewpoint_depression_k=float(rng.uniform(0.5, 3.0)),
        wind_u_surface_ms=float(rng.uniform(-8.0, 8.0)),
        wind_v_surface_ms=float(rng.uniform(-8.0, 8.0)),
        wind_shear_u_per_m=float(rng.uniform(1e-3, 6e-3)),
    )

    cell = WeatherCell(
        x_m=0.0,
        y_m=0.0,
        z_m=layer_base + layer_thickness / 2.0,
        sigma_h_m=45_000.0,
        sigma_v_m=layer_thickness / 2.0,
        peak_lwc_kg_m3=lwc0 * 1e-3,
        mvd_m=mvd0 * 1e-6,
        grow_time_s=60.0,
        mature_time_s=duration_s,
        decay_time_s=120.0,
        label="departure_layer",
    )

    onset = sample_onset(regime, rng, onset_time_s=float(rng.uniform(60.0, 240.0)), duration_s=600.0)
    forcings = [
        _make_cooling_forcing(onset),
        _make_cell_growth_forcing(cell, onset, max_lwc_g_m3=1.2),
    ]

    # Airspeeds are DHC-6 values: clean stall ~40-47 m/s depending on altitude,
    # published cruise 150 kt (77 m/s), V_MO 170 kt (87.5 m/s). Earlier values
    # (95-140 m/s) were set for a heavier airframe and exceeded V_MO, producing
    # spurious overspeed breaches after the switch to the DHC-6.
    # Staged climb schedule. Episodes begin in the initial climb rather than
    # on the runway: a step from 15 m to cruise altitude at 65 m/s demands more
    # lift than the wing can produce, which saturates guidance and would be
    # recorded as an envelope breach with no ice present.
    plan = FlightPlan(
        segments=[
            (0.0, 250.0, 58.0, 15.0, False, "takeoff"),
            (40.0, 1200.0, 64.0, 15.0, False, "climb"),
            (120.0, cruise_alt, 72.0, 0.0, False, "climb"),
            (duration_s * 0.6, cruise_alt, 77.0, 0.0, False, "cruise"),
        ],
        heading_rad=float(rng.uniform(0.0, 2.0 * math.pi)),
    )

    return ScenarioSpec(
        scenario_id=f"UC1_{regime}",
        use_case="UC1",
        description=(
            "Departure into an active low-level icing layer with evolving "
            "severity; climb through the layer to a low cruise altitude."
        ),
        duration_s=duration_s,
        atmosphere_config=atmosphere,
        cells=[cell],
        forcings=forcings,
        flight_plan=plan,
        crew_policy=CrewPolicy(
            reaction_delay_s=float(rng.uniform(20.0, 90.0)),
            never_reacts=bool(rng.random() < 0.12),
        ),
        protection=IceProtectionConfig(),
        nuisance=NuisanceConfig(
            turbulence_w20_ms=float(rng.uniform(5.0, 14.0)),
            cg_shift_events=False,
        ),
        aircraft_config=DHC6_CONFIG,
        onset=onset,
        metadata={
            "surface_temp_c": surface_temp_c,
            "layer_base_m": layer_base,
            "layer_thickness_m": layer_thickness,
            "initial_lwc_g_m3": lwc0,
            "initial_mvd_um": mvd0,
            "cruise_altitude_m": cruise_alt,
            "ground_contamination_mm": float(rng.uniform(0.0, 2.5)),
        },
    )


def build_uc2(
    rng: np.random.Generator,
    *,
    regime: OnsetRegime = "gradual",
    duration_s: float = 2400.0,
) -> ScenarioSpec:
    """UC-2: crew holds a chosen cruise altitude while conditions deteriorate.

    Construction: the icing layer is centred *on* the crew's selected altitude
    and intensifies, while a materially better altitude exists (above the layer
    top or below the freezing level). The scenario records the better altitude
    and the window during which it was available, so a recommendation can be
    scored against a defined alternative rather than against an outcome alone.
    """
    surface_temp_c = float(rng.uniform(1.0, 9.0))
    lapse = float(rng.uniform(5.5e-3, 7.5e-3))
    freezing_level = (surface_temp_c) / (lapse * 1000.0) * 1000.0
    freezing_level = max(freezing_level, 300.0)

    chosen_alt = float(freezing_level + rng.uniform(600.0, 1800.0))
    layer_thickness = float(rng.uniform(700.0, 1500.0))
    better_alt_above = chosen_alt + layer_thickness / 2.0 + 700.0
    better_alt_below = max(freezing_level - 500.0, 700.0)
    prefer_above = bool(rng.random() < 0.5)
    better_alt = better_alt_above if prefer_above else better_alt_below

    atmosphere = AtmosphereConfig(
        sea_level_temperature_k=T_FREEZE + surface_temp_c,
        lapse_rate_k_per_m=lapse,
        inversion_strength_k=float(rng.uniform(0.0, 3.5)),
        inversion_altitude_m=chosen_alt,
        surface_dewpoint_depression_k=float(rng.uniform(1.0, 5.0)),
        wind_shear_u_per_m=float(rng.uniform(1e-3, 5e-3)),
    )

    cell = WeatherCell(
        x_m=0.0,
        y_m=0.0,
        z_m=chosen_alt,
        sigma_h_m=90_000.0,
        sigma_v_m=layer_thickness / 2.0,
        peak_lwc_kg_m3=float(rng.uniform(0.08, 0.25)) * 1e-3,
        mvd_m=float(rng.uniform(12.0, 25.0)) * 1e-6,
        grow_time_s=120.0,
        mature_time_s=duration_s,
        decay_time_s=180.0,
        label="cruise_layer",
    )

    onset = sample_onset(regime, rng, onset_time_s=float(rng.uniform(300.0, 900.0)), duration_s=1200.0)
    forcings = [
        _make_cooling_forcing(onset),
        _make_cell_growth_forcing(cell, onset, max_lwc_g_m3=0.95),
    ]

    plan = FlightPlan(
        segments=[
            (0.0, chosen_alt, 77.0, 0.0, False, "cruise"),
        ],
        heading_rad=float(rng.uniform(0.0, 2.0 * math.pi)),
    )

    return ScenarioSpec(
        scenario_id=f"UC2_{regime}_{'above' if prefer_above else 'below'}",
        use_case="UC2",
        description=(
            "Crew maintains a pre-selected cruise altitude while the icing "
            "layer at that altitude intensifies; a materially better altitude "
            "is available but not taken."
        ),
        duration_s=duration_s,
        atmosphere_config=atmosphere,
        cells=[cell],
        forcings=forcings,
        flight_plan=plan,
        crew_policy=CrewPolicy(
            reaction_delay_s=float(rng.uniform(60.0, 240.0)),
            never_reacts=bool(rng.random() < 0.25),
        ),
        protection=IceProtectionConfig(),
        nuisance=NuisanceConfig(
            turbulence_w20_ms=float(rng.uniform(4.0, 12.0)),
            cg_shift_events=bool(rng.random() < 0.3),
        ),
        aircraft_config=DHC6_CONFIG,
        onset=onset,
        metadata={
            "chosen_altitude_m": chosen_alt,
            "better_altitude_m": better_alt,
            "better_altitude_is_above": prefer_above,
            "initial_freezing_level_m": freezing_level,
            "layer_thickness_m": layer_thickness,
            "disagreement_start_s": onset.onset_time_s,
        },
    )


def build_uc3(
    rng: np.random.Generator,
    *,
    regime: OnsetRegime = "rapid",
    duration_s: float | None = None,
) -> ScenarioSpec:
    """UC-3: rapid in-flight icing onset requiring immediate response.

    All four forcings act together: the column cools, the freezing level
    descends, the cell intensifies and its droplets grow toward SLD, and the
    layer itself descends onto the aircraft. Terminal conditions are comparable
    across onset regimes by construction, so the regimes differ in *rate*
    rather than in *destination* -- which is what makes the onset-novelty split
    a clean test of rate sensitivity rather than a severity confound.
    """
    surface_temp_c = float(rng.uniform(0.0, 7.0))
    lapse = float(rng.uniform(5.5e-3, 7.0e-3))
    cruise_alt = float(rng.uniform(1800.0, 6000.0))
    initial_fl = max(surface_temp_c / lapse, 200.0)
    if duration_s is None:
        duration_s = float(rng.uniform(1800.0, 4200.0))

    atmosphere = AtmosphereConfig(
        sea_level_temperature_k=T_FREEZE + surface_temp_c,
        lapse_rate_k_per_m=lapse,
        inversion_strength_k=float(rng.uniform(0.0, 4.0)),
        inversion_altitude_m=cruise_alt - float(rng.uniform(0.0, 800.0)),
        surface_dewpoint_depression_k=float(rng.uniform(0.5, 4.0)),
        wind_shear_u_per_m=float(rng.uniform(2e-3, 9e-3)),
        wind_shear_v_per_m=float(rng.uniform(-3e-3, 3e-3)),
    )

    cell = WeatherCell(
        x_m=0.0,
        y_m=0.0,
        z_m=cruise_alt + float(rng.uniform(-200.0, 600.0)),
        # Horizontal extent is log-uniform over the full physical range of
        # icing regions, ~15 km convective cells to ~600 km frontal bands
        # (P3, coverage over control). Log-uniform, because the quantity is
        # scale-like: uniform sampling would make sub-50 km encounters, and
        # hence brief edge-clip episodes, vanishingly rare.
        sigma_h_m=float(np.exp(rng.uniform(np.log(15_000.0), np.log(600_000.0)))),
        sigma_v_m=float(rng.uniform(200.0, 1200.0)),
        peak_lwc_kg_m3=float(rng.uniform(0.02, 0.35)) * 1e-3,
        mvd_m=float(rng.uniform(12.0, 35.0)) * 1e-6,
        grow_time_s=float(rng.uniform(60.0, 180.0)),
        # The cloud may persist past the episode or decay mid-flight; both
        # "sustained exposure" and "encounter then clear air" are generated.
        mature_time_s=float(rng.uniform(0.45, 1.0)) * duration_s,
        decay_time_s=150.0,
        temperature_anomaly_k=float(rng.uniform(-0.8, 0.2)),
        label="rapid_onset_layer",
    )

    onset = sample_onset(regime, rng, duration_s=float(rng.uniform(300.0, 1800.0)))
    forcings = [
        _make_cooling_forcing(onset),
        _make_freezing_level_forcing(onset, initial_fl),
        _make_cell_growth_forcing(cell, onset, max_lwc_g_m3=float(rng.uniform(0.8, 1.6))),
        _make_cell_descent_forcing(
            cell,
            rate_m_per_min=onset.freezing_level_rate_m_per_min * 0.4,
            floor_m=max(cruise_alt - 300.0, 500.0),
        ),
    ]

    plan = FlightPlan(
        segments=[
            (0.0, cruise_alt, 77.0, 0.0, False, "cruise"),
        ],
        heading_rad=float(rng.uniform(0.0, 2.0 * math.pi)),
    )

    return ScenarioSpec(
        scenario_id=f"UC3_{regime}",
        use_case="UC3",
        description=(
            "Rapidly evolving in-flight icing: simultaneous column cooling, "
            "freezing-level descent, LWC intensification and droplet growth "
            "toward SLD, with the layer descending onto the aircraft."
        ),
        duration_s=duration_s,
        atmosphere_config=atmosphere,
        cells=[cell],
        forcings=forcings,
        flight_plan=plan,
        crew_policy=CrewPolicy(
            reaction_delay_s=float(rng.uniform(30.0, 150.0)),
            never_reacts=bool(rng.random() < 0.20),
        ),
        # Equipment state is part of the scenario space: ~10% of episodes fly
        # with one protection system inoperative (dispatch under MEL relief).
        protection=IceProtectionConfig(
            antiice_available=bool(rng.random() > 0.10),
            boots_available=bool(rng.random() > 0.10),
        ),
        nuisance=NuisanceConfig(
            turbulence_w20_ms=float(rng.uniform(4.0, 22.0)),
            cg_shift_events=bool(rng.random() < 0.25),
        ),
        aircraft_config=DHC6_CONFIG,
        onset=onset,
        metadata={
            "cruise_altitude_m": cruise_alt,
            "initial_freezing_level_m": initial_fl,
            "surface_temp_c": surface_temp_c,
        },
    )


def build_nuisance_only(
    rng: np.random.Generator,
    *,
    duration_s: float = 1500.0,
) -> ScenarioSpec:
    """Hard negative: strong dynamic disturbance with **no** supercooled water.

    The column is warm enough that no accretion is thermodynamically possible,
    but turbulence is elevated, CG shifts occur, control friction is high and
    the flight plan includes configuration changes. Any OOD or icing-risk
    activation on this scenario is a false positive by construction, which
    makes it the direct measurement of the failure mode where a model has
    learned "unusual dynamics" rather than "ice".
    """
    # The whole profile must stay above freezing, or the "warm" cloud becomes
    # supercooled and this scenario stops being a hard negative. Altitude and
    # lapse are therefore chosen jointly and the margin is asserted below.
    surface_temp_c = float(rng.uniform(16.0, 26.0))
    lapse = float(rng.uniform(5.0e-3, 6.8e-3))
    cruise_alt = float(rng.uniform(1200.0, 2400.0))
    # Coldest point of the profile is the top of the climb; require >= +3 degC
    # there so OU temperature perturbations cannot dip it below zero.
    coldest_c = surface_temp_c - lapse * (cruise_alt + 200.0)
    if coldest_c < 3.0:
        cruise_alt = max((surface_temp_c - 3.0) / lapse - 200.0, 600.0)

    atmosphere = AtmosphereConfig(
        sea_level_temperature_k=T_FREEZE + surface_temp_c,
        lapse_rate_k_per_m=lapse,
        surface_dewpoint_depression_k=float(rng.uniform(2.0, 8.0)),
        wind_shear_u_per_m=float(rng.uniform(3e-3, 1.1e-2)),
        wind_shear_v_per_m=float(rng.uniform(-5e-3, 5e-3)),
    )

    # Cloud with liquid water but entirely above freezing: no accretion.
    cell = WeatherCell(
        x_m=0.0,
        y_m=0.0,
        z_m=cruise_alt,
        sigma_h_m=50_000.0,
        sigma_v_m=700.0,
        peak_lwc_kg_m3=float(rng.uniform(0.2, 0.8)) * 1e-3,
        mvd_m=float(rng.uniform(15.0, 40.0)) * 1e-6,
        grow_time_s=60.0,
        mature_time_s=duration_s,
        decay_time_s=120.0,
        label="warm_cloud",
    )

    # Configuration changes partway through: a non-icing drag + trim event.
    plan = FlightPlan(
        segments=[
            (0.0, cruise_alt, 77.0, 0.0, False, "cruise"),
            (duration_s * 0.35, cruise_alt, 64.0, 10.0, False, "cruise"),
            (duration_s * 0.50, cruise_alt, 77.0, 0.0, False, "cruise"),
            (duration_s * 0.70, max(cruise_alt - 600.0, 400.0), 70.0, 0.0, False, "descent"),
        ],
        heading_rad=float(rng.uniform(0.0, 2.0 * math.pi)),
        speedbrake_pct=0.0,
    )

    return ScenarioSpec(
        scenario_id="NUIS_warm_turbulent",
        use_case="NUIS",
        description=(
            "Hard negative. Elevated turbulence, CG shifts, high control "
            "friction and configuration changes in warm cloud with zero "
            "supercooled liquid water. Any icing activation here is a false "
            "positive by construction."
        ),
        duration_s=duration_s,
        atmosphere_config=atmosphere,
        cells=[cell],
        forcings=[],
        flight_plan=plan,
        crew_policy=CrewPolicy(never_reacts=True),
        protection=IceProtectionConfig(antiice_available=True, boots_available=True),
        nuisance=NuisanceConfig(
            turbulence_enabled=True,
            turbulence_w20_ms=float(rng.uniform(14.0, 24.0)),
            cg_drift_enabled=True,
            cg_drift_sigma_pct_mac=float(rng.uniform(1.0, 2.2)),
            cg_shift_events=True,
            cg_shift_rate_per_s=float(rng.uniform(3e-4, 9e-4)),
            control_friction_enabled=True,
            friction_deadband_deg=float(rng.uniform(0.10, 0.30)),
            config_changes_enabled=True,
        ),
        aircraft_config=DHC6_CONFIG,
        onset=None,
        metadata={
            "cruise_altitude_m": cruise_alt,
            "surface_temp_c": surface_temp_c,
            "lapse_rate_k_per_m": lapse,
            "coldest_profile_temp_c": surface_temp_c - lapse * (cruise_alt + 200.0),
            "icing_possible": False,
        },
    )



# ---------------------------------------------------------------------------
# UC-4 -- low-margin flight regimes
# ---------------------------------------------------------------------------
#
# WHY THIS FAMILY EXISTS. The first production dataset contained no danger at
# all: `lbl_risk_class` was `nominal` on all 25.26M rows. Every episode flew
# cruise at 77 m/s, starting ~12 deg from stall; ice removed ~5 deg and left
# ~7 deg, never reaching even the 5 deg caution threshold. The detection task
# had nothing to detect.
#
# The cause was scope, not physics. Cruise is the highest-margin phase of
# flight. The icing accident record is not a cruise record: Cao Table 15's
# encounters and NTSB AAR-96/01 (Roselawn, a HOLDING pattern) sit in slow
# flight, where margin is small before ice does anything.
#
# Speeds below are multiples of the DHC-6's published clean stall range
# (40-47 m/s, airframes.py) at standard operational factors - 1.3 Vs approach,
# 1.35 Vs hold, 1.4 Vs climb. They are not tuned to produce warnings. The
# thresholds in aircraft.py are untouched: 5.0 deg caution, 2.5 deg warning.


def build_uc4(
    rng: np.random.Generator,
    *,
    regime: str = "approach",
    duration_s: float = 1800.0,
) -> ScenarioSpec:
    """Low-margin regimes: approach, holding, and performance-limited climb."""
    if regime not in ("approach", "hold", "climb"):
        raise ValueError(f"unknown UC-4 regime: {regime}")

    # DHC-6 clean 1g stall, airframes.py / RLEA_PROJECT_CONTEXT Sec. 3.
    v_stall = float(rng.uniform(40.0, 47.0))

    surface_temp_c = float(rng.uniform(-2.0, 8.0))
    lapse = float(rng.uniform(5.5e-3, 8.5e-3))
    initial_fl = max(surface_temp_c / lapse, 0.0)

    if regime == "approach":
        # Descent through the icing layer to circling minima. Flaps and gear
        # each consume margin before ice contributes anything.
        top_alt = float(rng.uniform(2400.0, 3400.0))
        v_app = 1.30 * v_stall
        plan = FlightPlan(
            segments=[
                (0.0, top_alt, 1.45 * v_stall, 0.0, False, "cruise"),
                (0.45 * duration_s, 1500.0, 1.35 * v_stall, 10.0, False, "descent"),
                (0.72 * duration_s, 700.0, v_app, 20.0, True, "approach"),
                (0.88 * duration_s, 400.0, v_app, 20.0, True, "approach"),
            ],
            heading_rad=float(rng.uniform(0.0, 2.0 * math.pi)),
        )
        cell_alt = float(rng.uniform(900.0, 2600.0))
        label = "approach_layer"

    elif regime == "hold":
        # Racetrack hold in icing, the Roselawn configuration. Bank angle
        # raises required lift, so margin is consumed by the turn itself.
        hold_alt = float(rng.uniform(1800.0, 3000.0))
        v_hold = 1.35 * v_stall
        plan = FlightPlan(
            segments=[(0.0, hold_alt, v_hold, 0.0, False, "cruise")],
            heading_rad=float(rng.uniform(0.0, 2.0 * math.pi)),
        )
        cell_alt = hold_alt + float(rng.uniform(-300.0, 300.0))
        label = "holding_layer"

    else:  # climb
        # Climb at max continuous through the icing layer. There is no spare
        # thrust, so added drag cannot be answered with power: the aircraft
        # falls behind its commanded profile and decays toward stall.
        top_alt = float(rng.uniform(3500.0, 5000.0))
        v_climb = 1.40 * v_stall
        plan = FlightPlan(
            segments=[
                (0.0, 800.0, v_climb, 10.0, False, "climb"),
                (0.15 * duration_s, top_alt, v_climb, 0.0, False, "climb"),
            ],
            heading_rad=float(rng.uniform(0.0, 2.0 * math.pi)),
        )
        cell_alt = float(rng.uniform(1500.0, 3800.0))
        label = "climb_layer"

    atmosphere = AtmosphereConfig(
        sea_level_temperature_k=T_FREEZE + surface_temp_c,
        lapse_rate_k_per_m=lapse,
        inversion_strength_k=float(rng.uniform(0.0, 4.0)),
        inversion_altitude_m=cell_alt - float(rng.uniform(0.0, 600.0)),
        surface_dewpoint_depression_k=float(rng.uniform(0.3, 3.0)),
        wind_shear_u_per_m=float(rng.uniform(2e-3, 9e-3)),
        wind_shear_v_per_m=float(rng.uniform(-3e-3, 3e-3)),
    )

    # MVD spans into the Appendix O freezing-drizzle band on a minority of
    # episodes, so that the SLD ridge mechanism in icing.py can trigger. The
    # 50 um threshold is the Appendix C/O boundary, not a tuned value.
    sld_episode = rng.random() < 0.30
    mvd0 = (
        float(rng.uniform(55.0, 140.0)) if sld_episode
        else float(rng.uniform(12.0, 35.0))
    )

    cell = WeatherCell(
        x_m=0.0,
        y_m=0.0,
        z_m=cell_alt,
        sigma_h_m=float(np.exp(rng.uniform(np.log(15_000.0), np.log(600_000.0)))),
        sigma_v_m=float(rng.uniform(200.0, 1200.0)),
        peak_lwc_kg_m3=float(rng.uniform(0.05, 0.45)) * 1e-3,
        mvd_m=mvd0 * 1e-6,
        grow_time_s=float(rng.uniform(60.0, 180.0)),
        mature_time_s=float(rng.uniform(0.50, 1.0)) * duration_s,
        decay_time_s=150.0,
        temperature_anomaly_k=float(rng.uniform(-0.8, 0.2)),
        label=label,
    )

    onset = sample_onset(
        str(rng.choice(["gradual", "moderate", "rapid"])),
        rng,
        duration_s=float(rng.uniform(300.0, 1800.0)),
    )
    forcings = [
        _make_cooling_forcing(onset),
        _make_freezing_level_forcing(onset, initial_fl),
        _make_cell_growth_forcing(cell, onset, max_lwc_g_m3=float(rng.uniform(0.8, 1.6))),
    ]

    return ScenarioSpec(
        scenario_id=f"UC4_{regime}",
        use_case="UC4",
        description=(
            "Low-margin flight regime in icing. Margin is small before ice "
            "contributes: slow flight, high-lift configuration, bank, or "
            "thrust-limited climb. Mirrors the phases in which the icing "
            "accident record actually sits."
        ),
        duration_s=duration_s,
        atmosphere_config=atmosphere,
        cells=[cell],
        forcings=forcings,
        flight_plan=plan,
        crew_policy=CrewPolicy(
            reaction_delay_s=float(rng.uniform(30.0, 150.0)),
            never_reacts=bool(rng.random() < 0.20),
        ),
        protection=IceProtectionConfig(
            antiice_available=bool(rng.random() > 0.10),
            boots_available=bool(rng.random() > 0.10),
        ),
        nuisance=NuisanceConfig(
            turbulence_w20_ms=float(rng.uniform(4.0, 22.0)),
            cg_shift_events=bool(rng.random() < 0.25),
        ),
        aircraft_config=DHC6_CONFIG,
        onset=onset,
        metadata={
            "regime": regime,
            "stall_speed_ms": v_stall,
            "cell_altitude_m": cell_alt,
            "initial_freezing_level_m": initial_fl,
            "surface_temp_c": surface_temp_c,
            "sld_episode": float(sld_episode),
        },
    )


#: Dispatch table used by the batch generator.
SCENARIO_BUILDERS: dict[str, Callable[..., ScenarioSpec]] = {
    "UC1": build_uc1,
    "UC2": build_uc2,
    "UC3": build_uc3,
    "UC4": build_uc4,
    "NUIS": build_nuisance_only,
}
