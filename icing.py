r"""
Physics-informed ice accretion with randomised parameter families.
==================================================================

Module role in RLEA
-------------------
This module converts an :class:`~rlea.simdata.atmosphere.AtmosphericState` plus
the aircraft's airspeed into (a) accreted ice mass and thickness and (b) the
aerodynamic degradation that ice imposes. It is the physical bridge between the
environment and the flight dynamics, and it is the source of the *ground truth*
the RLEA evaluation uses: accreted mass, ice type, freezing fraction and
aerodynamic penalties are all read directly from simulator state, so no
external label is ever required.

Why parameter families exist
----------------------------
A single fixed accretion parameterisation would give the pipeline exactly one
generative process. Every OOD, retrieval and attribution-stability number would
then be measured against that one process, and "distribution shift" would
reduce to "different weather through the same physics."

Instead, each episode draws a :class:`PhysicsParameters` sample from a named
:class:`PhysicsFamily`. Families differ in collection efficiency scaling,
freezing-fraction bias, ice density, the ice-shape-to-drag transfer function
and shedding behaviour. This yields the **physics-parameter novelty axis**:

* train / in-distribution : family ``A``
* OOD evaluation         : families ``B`` and ``C``

A model that has learned a genuine *dynamics-degradation* signature should
retain some skill across families; a model that has memorised one accretion
transfer function should not. Distributions are tabulated in
``METHODOLOGY.md``.

Physics
-------
**1. Droplet inertia and collection efficiency.** Whether a droplet impacts the
leading edge or follows the streamlines around it is set by the modified
inertia parameter. With droplet diameter :math:`d`, airspeed :math:`V`, air
viscosity :math:`\mu`, and :math:`c_{LE}` the **impingement length scale**
(the body dimension droplets flow around, order twice the leading-edge radius
-- emphatically *not* the wing chord, which is used only for the aerodynamic
surrogate in section 5):

.. math::

    K = \frac{\rho_w d^2 V}{18\,\mu\,c_{LE}},
    \qquad
    \mathrm{Re}_\delta = \frac{\rho_a d V}{\mu}

The Langmuir--Blodgett range correction and stagnation-line collection
efficiency (Langmuir & Blodgett 1946; as used in LEWICE):

.. math::

    \frac{\lambda}{\lambda_{Stokes}}
        = \frac{1}{0.8388 + 0.001483\,\mathrm{Re}_\delta
                   + 0.1847\sqrt{\mathrm{Re}_\delta}}

.. math::

    K_0 = \frac{1}{8} + \left(K - \frac{1}{8}\right)\frac{\lambda}{\lambda_{Stokes}},
    \qquad
    \beta_0 = \frac{1.40\,(K_0 - 1/8)^{0.84}}{1 + 1.40\,(K_0 - 1/8)^{0.84}}

with :math:`\beta_0 = 0` for :math:`K_0 \le 1/8`. This is the single most
important nonlinearity in the model: it is why MVD matters as much as LWC, and
why small droplets in high LWC can accrete less than large droplets in low LWC.

**2. Messinger energy balance and freezing fraction.** Following Messinger
(1953) in the Ruff/LEWICE algebraic form, the freezing fraction :math:`n` is
the fraction of impinging water that freezes within the control volume:

.. math::

    n = \frac{c_{p,w}}{L_f}\left[\phi + \frac{\theta}{b}\right]

.. math::

    \phi = T_f - T_\infty - \frac{V^2}{2\,c_{p,w}}
    \qquad\text{(droplet kinetic + sensible term)}

.. math::

    \theta = T_f - T_\infty - \frac{r\,V^2}{2\,c_{p,a}}
    \qquad\text{(air energy transfer, } r \approx 0.9 \text{)}

.. math::

    b = \frac{\mathrm{LWC}\;V\,\beta_0\,c_{p,w}}{h_c}
    \qquad\text{(relative heat factor)}

with :math:`n` clipped to :math:`[0, 1]`. The physical reading:

* :math:`n \to 1` : all impinging water freezes on impact -> **rime** ice,
  opaque, low density, conforms to the leading edge.
* :math:`0 < n < 1` : partial freezing, remaining water runs back and freezes
  downstream -> **glaze** ice, dense, and forms the horn shapes that cause
  disproportionate aerodynamic penalty.

**3. Convective heat transfer.** Flat-plate turbulent correlation, evaluated at
the stagnation region:

.. math::

    \mathrm{Nu} = 0.0296\,\mathrm{Re}_c^{0.8}\,\mathrm{Pr}^{1/3},
    \qquad h_c = \frac{\mathrm{Nu}\,k_{air}}{c}

Surface roughness from accreted ice raises :math:`h_c`, which raises :math:`n`,
which is a genuine positive feedback in early accretion. Modelled by the
roughness augmentation factor :math:`f_{rough}`.

**4. Mass accumulation.** Impingement rate per unit span-area at the stagnation
region:

.. math::

    \dot{m}_{imp} = \beta_0\,\mathrm{LWC}\,V \quad [\mathrm{kg\,m^{-2}\,s^{-1}}]

.. math::

    \dot{m}_{ice} = n\,\dot{m}_{imp},
    \qquad
    \dot{t}_{ice} = \frac{\dot{m}_{ice}}{\rho_{ice}(n)}

Ice density interpolates between a rime value and glaze (solid ice,
917 kg m^-3) with freezing fraction:

.. math::

    \rho_{ice}(n) = \rho_{glaze} - n\,(\rho_{glaze} - \rho_{rime})

**5. Aerodynamic degradation (parameterised surrogate).** The mapping from ice
geometry to aerodynamic penalty is *not* first-principles here. It is an
explicitly parameterised power law in non-dimensional thickness
:math:`\tau = t_{ice}/c`, whose exponents and gains are sampled per family:

.. math::

    f_{shape} = 1 + \kappa_{horn}\,(1 - n)

.. math::

    \Delta C_D = k_D\,\tau^{p_D}\,f_{shape},
    \qquad
    \Delta C_{L,max} = -k_L\,\tau^{p_L}\,f_{shape},
    \qquad
    \Delta \alpha_{stall} = -k_\alpha\,\tau^{p_\alpha}\,f_{shape}

.. math::

    \Delta C_{m} = -k_m\,\tau^{p_D}\,f_{shape}
    \qquad\text{(tailplane / trim change)}

with :math:`p < 1`, so sensitivity is steepest at small thickness -- the
empirically well-established result that the first millimetres of rough ice
cost disproportionately more than later smooth growth. The
:math:`f_{shape}` is the **ice-type** factor: it peaks at freezing fraction
:math:`n = 0.2`, falls away on both sides, and is normalised to 1.0 at the peak
with floor :math:`\phi`. Form after Cao et al. (2018) Sec. 5.4.2; an earlier
monotonic form over-penalised rime ~5x against DHC-6 flight data (METHODOLOGY
Sec. 12.7).

**This surrogate is a design parameter, not a physical claim.** Its gains and
exponents are exactly what varies across families, and that is the point: the
downstream model must not be permitted to rely on one particular transfer
function.

**6. Ice protection.** Two mechanisms, both operationally real and both
generating distinctive telemetry signatures:

* *Anti-ice* (evaporative / running-wet): continuously suppresses accretion by
  a fixed effectiveness fraction, with a lag.
* *De-ice boots*: cyclic. Ice accumulates to a threshold, the boot inflates and
  sheds a fraction, leaving residual ice. Produces the characteristic sawtooth
  in drag that is a strong retrieval cue.

**7. Shedding.** Self-shedding above a thickness threshold, with a stochastic
trigger. Asymmetric shedding is not modelled (3-DOF has no roll asymmetry
channel); it is noted as a limitation.

Assumptions and known limitations
---------------------------------
* Single-point (stagnation-line) accretion. No chordwise impingement
  distribution, no ice-shape geometry, no runback tracking beyond its effect on
  :math:`n`.
* Steady-state Messinger balance evaluated per step; no thermal inertia of the
  accreted layer.
* No explicit SLD physics (splashing, re-impingement aft of protected
  surfaces). SLD is represented only through the MVD dependence of
  :math:`\beta_0` and through family ``C``'s inflated large-droplet gains.
* No asymmetric or empennage-specific accretion.
* The aero-degradation surrogate is calibrated to the *general shape* of
  published correlations, not to any specific airfoil dataset.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from rlea.simdata.atmosphere import (
    CP_AIR,
    T_FREEZE,
    AtmosphericState,
    dynamic_viscosity,
)

__all__ = [
    "RHO_WATER",
    "L_FUSION",
    "TAU_REF",
    "CP_WATER",
    "IceType",
    "PhysicsParameters",
    "PhysicsFamily",
    "PHYSICS_FAMILIES",
    "IceProtectionConfig",
    "IceState",
    "AeroDegradation",
    "IcingModel",
]

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------

RHO_WATER = 1000.0     #: Density of liquid water, kg m^-3
RHO_GLAZE = 917.0      #: Density of solid (glaze) ice, kg m^-3
L_FUSION = 3.34e5      #: Latent heat of fusion of water, J kg^-1
CP_WATER = 4218.0      #: Specific heat of liquid water, J kg^-1 K^-1
K_AIR = 0.0243         #: Thermal conductivity of air, W m^-1 K^-1
PR_AIR = 0.72          #: Prandtl number of air, dimensionless
RECOVERY_R = 0.90      #: Adiabatic recovery factor for the surface, dimensionless

#: Reference non-dimensional ice thickness for the aerodynamic surrogate.
#: :math:`\tau_{ref} = t_{ice}/c = 0.005` corresponds to roughly 7 mm of ice on
#: a 1.4 m chord. Gains are defined *at this reference*, which decouples them
#: from the exponents so that families can differ in curvature without also
#: differing by orders of magnitude in magnitude.
TAU_REF = 0.005

IceType = Literal["none", "rime", "mixed", "glaze"]


# ---------------------------------------------------------------------------
# Randomised physics families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicsParameters:
    r"""One concrete draw of accretion physics, held fixed for an episode.

    These parameters are **hidden from the model**. They are recorded in
    episode metadata solely so that the physics-parameter novelty split can be
    constructed and audited.

    Attributes
    ----------
    family : str
        Name of the originating :class:`PhysicsFamily` ("A", "B", "C").
    beta_scale : float
        Multiplier on the Langmuir--Blodgett collection efficiency
        :math:`\beta_0`, dimensionless. Absorbs airfoil-geometry dependence
        that the single-point model cannot resolve.
    beta_mvd_exponent : float
        Additional power-law sensitivity of :math:`\beta_0` to MVD relative to
        a 20 um reference, dimensionless. Zero recovers pure
        Langmuir--Blodgett; positive values exaggerate large-droplet capture
        (the SLD-leaning behaviour of family ``C``).
    freezing_fraction_bias : float
        Additive bias on :math:`n` before clipping, dimensionless. Represents
        unmodelled surface-energy effects.
    roughness_augmentation : float
        Multiplier on convective heat transfer :math:`h_c` due to ice
        roughness, dimensionless (>= 1).
    rho_rime : float
        Rime ice density, kg m^-3. Literature spans roughly 300-900.
    drag_gain, drag_exponent : float
        :math:`k_D`, :math:`p_D` in the drag surrogate. ``drag_gain`` is the
        :math:`\Delta C_D` produced at the reference thickness ``TAU_REF``.
    lift_gain, lift_exponent : float
        :math:`k_L`, :math:`p_L` in the maximum-lift surrogate.
    alpha_gain, alpha_exponent : float
        :math:`k_\alpha`, :math:`p_\alpha` for stall-angle reduction.
        ``alpha_gain`` is the reduction in **degrees** at ``TAU_REF``.
    moment_gain : float
        :math:`k_m`, pitching-moment coefficient shift gain.
    oswald_gain, oswald_exponent : float
        :math:`k_e`, :math:`p_e` in the span-efficiency surrogate.
        ``oswald_gain`` is the *fractional* loss of Oswald efficiency at the
        reference thickness ``TAU_REF``.
    shape_floor : float
        :math:`\phi`, residual value of the ice-type shape function for pure
        rime (:math:`n \to 1`) and pure runback (:math:`n \to 0`).
    shed_threshold_m : float
        Ice thickness above which self-shedding may trigger, m.
    shed_probability_per_s : float
        Hazard rate of a self-shedding event once above threshold, s^-1.
    shed_fraction : float
        Fraction of accreted mass removed by a shedding event, dimensionless.
    chord_m : float
        Reference wing chord, m. Used **only** to non-dimensionalise ice
        thickness in the aerodynamic surrogate, :math:`\tau = t_{ice}/c`.
    leading_edge_scale_m : float
        Effective impingement length scale, m. Used **only** in the droplet
        inertia parameter :math:`K`. This is the body dimension droplets flow
        around (order twice the leading-edge radius), not the chord; the two
        differ by more than an order of magnitude and conflating them
        suppresses :math:`\beta_0` by a comparable factor. The single-point
        model absorbs unresolved airfoil geometry into this scale, so it is an
        effective parameter rather than a measured radius.
    """

    family: str
    beta_scale: float
    beta_mvd_exponent: float
    freezing_fraction_bias: float
    roughness_augmentation: float
    rho_rime: float
    drag_gain: float
    drag_exponent: float
    lift_gain: float
    lift_exponent: float
    alpha_gain: float
    alpha_exponent: float
    moment_gain: float
    oswald_gain: float
    oswald_exponent: float
    shape_floor: float
    shed_threshold_m: float
    shed_probability_per_s: float
    shed_fraction: float
    chord_m: float
    leading_edge_scale_m: float

    def as_dict(self) -> dict[str, float | str]:
        """Flat dict for episode metadata / novelty-split auditing."""
        return asdict(self)


@dataclass(frozen=True)
class PhysicsFamily:
    r"""A named distribution over :class:`PhysicsParameters`.

    Each field is a ``(low, high)`` bound for an independent uniform draw,
    except ``rho_rime_range`` which is also uniform. Uniform (rather than
    Gaussian) sampling is deliberate: it gives *bounded, fully-specified
    support*, so "in-distribution" and "out-of-distribution" are crisp set
    memberships rather than tail probabilities. That crispness is what makes
    the OOD claim auditable in the paper.

    Notes
    -----
    Family separation is by design *not* uniform across parameters. Families
    ``B`` and ``C`` overlap family ``A`` on some axes and are disjoint on
    others, which produces a graded rather than binary novelty signal --
    matching the compositional-novelty concern raised in the framework design.
    """

    name: str
    description: str
    beta_scale_range: tuple[float, float]
    beta_mvd_exponent_range: tuple[float, float]
    freezing_fraction_bias_range: tuple[float, float]
    roughness_augmentation_range: tuple[float, float]
    rho_rime_range: tuple[float, float]
    drag_gain_range: tuple[float, float]
    drag_exponent_range: tuple[float, float]
    lift_gain_range: tuple[float, float]
    lift_exponent_range: tuple[float, float]
    alpha_gain_range: tuple[float, float]
    alpha_exponent_range: tuple[float, float]
    moment_gain_range: tuple[float, float]
    oswald_gain_range: tuple[float, float]
    oswald_exponent_range: tuple[float, float]
    shape_floor_range: tuple[float, float]
    shed_threshold_range: tuple[float, float]
    shed_probability_range: tuple[float, float]
    shed_fraction_range: tuple[float, float]
    chord_range: tuple[float, float]
    leading_edge_scale_range: tuple[float, float]

    def sample(self, rng: np.random.Generator) -> PhysicsParameters:
        """Draw one parameter set from this family."""

        def u(bounds: tuple[float, float]) -> float:
            return float(rng.uniform(bounds[0], bounds[1]))

        return PhysicsParameters(
            family=self.name,
            beta_scale=u(self.beta_scale_range),
            beta_mvd_exponent=u(self.beta_mvd_exponent_range),
            freezing_fraction_bias=u(self.freezing_fraction_bias_range),
            roughness_augmentation=u(self.roughness_augmentation_range),
            rho_rime=u(self.rho_rime_range),
            drag_gain=u(self.drag_gain_range),
            drag_exponent=u(self.drag_exponent_range),
            lift_gain=u(self.lift_gain_range),
            lift_exponent=u(self.lift_exponent_range),
            alpha_gain=u(self.alpha_gain_range),
            alpha_exponent=u(self.alpha_exponent_range),
            moment_gain=u(self.moment_gain_range),
            oswald_gain=u(self.oswald_gain_range),
            oswald_exponent=u(self.oswald_exponent_range),
            shape_floor=u(self.shape_floor_range),
            shed_threshold_m=u(self.shed_threshold_range),
            shed_probability_per_s=u(self.shed_probability_range),
            shed_fraction=u(self.shed_fraction_range),
            chord_m=u(self.chord_range),
            leading_edge_scale_m=u(self.leading_edge_scale_range),
        )


# ---------------------------------------------------------------------------
# SLD ridge ice (Bragg et al.; Cao et al. 2018 Sec. 3, Figs. 9-17)
# ---------------------------------------------------------------------------
#
# Bragg et al. investigated simulated ridge shapes "which may form aft of
# protected surfaces in SLD conditions", and found the degradation "primarily a
# function of ice shape size and location and nearly independent of Reynolds
# number and ice shape geometry". For the forward-loaded NACA 23012m airfoil
# this included "an 80% loss of Cl,max for an upper surface ice shape location
# of x/c = 0.12".
#
# This is the worst lift loss in the published record and it is a LOCATION
# effect, not a thickness effect: under the re-derived leading-edge law above,
# reaching 80 % by accretion alone would require ~134 mm of ice. It is
# therefore modelled as its own mechanism, triggered by conditions the
# simulator already samples - SLD droplets (MVD above the Appendix O threshold)
# impinging aft of a surface that protection is actively keeping clear.
#
# The operational significance is the Roselawn mechanism (NTSB AAR-96/01): the
# ridge forms BECAUSE the protection is running, so the configuration this model
# previously treated as safest carries the worst documented case.

#: Ridge height at which the measured 80 % C_L,max loss is reached, as k/c.
#: Cao Fig. 17 tests the NACA 23012m ridge at k/c = 0.0139; that is the
#: simulation height the published loss corresponds to.
#: Appendix C/O supercooled-large-drop threshold, um.
SLD_THRESHOLD_UM: float = 50.0

RIDGE_REF_KC: float = 0.0139

#: Peak C_L,max loss carried by a fully developed SLD ridge, as a fraction.
#: Bragg et al., upper-surface shape at x/c = 0.12.
RIDGE_CLMAX_LOSS: float = 0.80

#: MVD (um) at which impingement is taken to fall entirely aft of the protected
#: region. Lower bound is the Appendix C/O SLD threshold of 50 um; the upper
#: bound spans into the Appendix O freezing-drizzle band. Linear in between.
RIDGE_MVD_FULL_UM: float = 200.0

#: Family ``A`` -- nominal / in-distribution.
#: Centred on values consistent with conventional airfoil icing behaviour.
#: This is the **training family**.
_FAMILY_A = PhysicsFamily(
    name="A",
    description=(
        "Nominal conventional-airfoil accretion. Training / in-distribution "
        "family. Moderate collection efficiency, standard rime density, "
        "conventional drag transfer function."
    ),
    # Family A ranges are the OBSERVED SPREAD across the three DHC-6 natural-icing
    # flights of NASA TM-83564, obtained by inverting each flight separately -
    # not a fitted mean with an arbitrary tolerance. Where replicate measurements
    # disagree, that disagreement measures real physical variance driven by ice
    # shape and location, which our single-point accretion model cannot resolve
    # (A-ICE-01). See PROVENANCE, "Principle worth adopting".
    #   drag_gain   : per-flight 0.0287 / 0.0320 / 0.0468
    #   oswald_gain : per-flight 0.300 / 0.481 (glaze flights only; the rime
    #                 Oswald change was within the report's stated scatter)
    #
    # lift/alpha RE-DERIVED 2026-09-11 (A-ICE-15 closed). The previous values
    # came from TM-83564's 17% / 16% lift losses, which the report states are
    # measured **at alpha = 6 deg** on the C_L-vs-alpha plot - a vertical offset
    # on the lift curve, not a loss of its peak. Applying them as delta_cl_max
    # was a definitional error and left family A below every published C_L,max
    # measurement (18-26% at 59 mm, vs 30-80% published).
    #
    # Re-derived by regressing Cao et al. (2018) Table 13 - three protection
    # states on the same airframe series - against the accretion thicknesses
    # METHODOLOGY 12.6 records (0.95 / 3.74 / 6.67 mm):
    #
    #     ice shape      C_L,max loss   alpha_stall loss
    #     inter-cycle         30 %          2.3 deg
    #     failed boot         41 %          7.3 deg
    #     S&C                 50 %          9.5 deg     (DHC-6 flight record)
    #
    #   -> lift : exponent 0.256, gain 0.542 at tau_ref   (fit 0.30/0.42/0.49)
    #   -> alpha: exponent 0.749, gain 13.69 at tau_ref   (fit 2.37/6.61/10.19)
    #
    # The raw alpha gain of 13.69 deg exceeds the DHC-6's whole 13.0 deg stall
    # angle, which is physical for a TAILPLANE (Table 13's source) and not for a
    # wing. Cao's review states tailplane ice accumulates 3-6x thicker than wing
    # ice for the same encounter, so the measured losses correspond to surface
    # thicknesses 3-6x those regressed. Correcting r by that factor:
    #
    #     k = 3x  ->  lift_gain 0.410, alpha_gain 6.00
    #     k = 6x  ->  lift_gain 0.343, alpha_gain 3.57
    #
    # The 3-6x transfer uncertainty becomes the sampling range, exactly as the
    # three-flight drag spread became drag_gain's range. Exponents are unchanged
    # by the rescaling (a power law's exponent is invariant under scaling of r),
    # so 0.256 / 0.749 are determined outright. Note they now DIFFER between the
    # two channels: measured C_L,max loss grows 1.7x across the series while
    # stall-angle loss grows 4.1x, which the previous near-identical exponents
    # could not represent.
    beta_scale_range=(0.90, 1.10),
    beta_mvd_exponent_range=(-0.05, 0.05),
    freezing_fraction_bias_range=(-0.03, 0.03),
    roughness_augmentation_range=(1.05, 1.25),
    rho_rime_range=(600.0, 800.0),
    drag_gain_range=(0.029, 0.047),
    drag_exponent_range=(0.30, 0.50),
    lift_gain_range=(0.343, 0.410),
    lift_exponent_range=(0.22, 0.30),
    alpha_gain_range=(3.57, 6.00),
    alpha_exponent_range=(0.68, 0.82),
    moment_gain_range=(0.045, 0.075),
    oswald_gain_range=(0.300, 0.481),
    oswald_exponent_range=(0.005, 0.050),
    shape_floor_range=(0.005, 0.020),
    shed_threshold_range=(0.020, 0.035),
    shed_probability_range=(0.0015, 0.0040),
    shed_fraction_range=(0.35, 0.60),
    chord_range=(1.98, 1.98),
    leading_edge_scale_range=(0.070, 0.070),
)

#: Family ``B`` -- shifted transfer function, same qualitative physics.
#: Higher roughness feedback, lower rime density, steeper and stronger drag
#: response, more aggressive horn penalty. Tests whether the model learned a
#: *degradation signature* or a *specific gain*.
_FAMILY_B = PhysicsFamily(
    name="B",
    description=(
        "Shifted aerodynamic transfer function. Same qualitative accretion "
        "physics, materially different gains: lower rime density (bulkier ice "
        "per unit mass), stronger roughness feedback, steeper drag exponent, "
        "harsher glaze horn penalty. OOD evaluation family."
    ),
    beta_scale_range=(1.10, 1.35),
    beta_mvd_exponent_range=(0.05, 0.18),
    freezing_fraction_bias_range=(0.02, 0.10),
    roughness_augmentation_range=(1.30, 1.70),
    rho_rime_range=(350.0, 550.0),
    drag_gain_range=(0.055, 0.090),
    drag_exponent_range=(0.20, 0.35),
    # B brackets the harsher end of the published record: C_L,max at the S&C /
    # LEWICE measurement of 50 %, stall-angle spanning failed-boot 7.3 deg to
    # S&C 9.5 deg. Exponents shallower than A - B saturates earlier. Family
    # bounds remain declared design parameters (A-ICE-10), but each endpoint now
    # points at a published number rather than at a multiple of family A.
    lift_gain_range=(0.45, 0.55),
    lift_exponent_range=(0.15, 0.24),
    alpha_gain_range=(6.00, 9.50),
    alpha_exponent_range=(0.50, 0.68),
    moment_gain_range=(0.085, 0.145),
    oswald_gain_range=(0.55, 0.85),
    oswald_exponent_range=(0.001, 0.030),
    shape_floor_range=(0.030, 0.090),
    shed_threshold_range=(0.035, 0.055),
    shed_probability_range=(0.0005, 0.0015),
    shed_fraction_range=(0.20, 0.40),
    chord_range=(1.98, 1.98),
    leading_edge_scale_range=(0.070, 0.070),
)

#: Family ``C`` -- SLD-leaning, adhesion-dominated.
#: Strong large-droplet capture, high freezing-fraction bias, dense ice, very
#: reluctant shedding. Represents the Appendix-O-like regime where ice persists
#: and accretes aft of protected surfaces. OOD evaluation family.
_FAMILY_C = PhysicsFamily(
    name="C",
    description=(
        "SLD-leaning, adhesion-dominated regime. Strong large-droplet capture, "
        "high freezing fraction, dense persistent ice that sheds reluctantly. "
        "Represents Appendix-O-like conditions. OOD evaluation family."
    ),
    beta_scale_range=(1.25, 1.60),
    beta_mvd_exponent_range=(0.20, 0.40),
    freezing_fraction_bias_range=(0.08, 0.20),
    roughness_augmentation_range=(1.15, 1.45),
    rho_rime_range=(780.0, 900.0),
    drag_gain_range=(0.036, 0.060),
    drag_exponent_range=(0.50, 0.70),
    # C sits between A and B on magnitude but is the STEEPEST in thickness -
    # large-droplet capture keeps accreting where A saturates. Upper lift bound
    # held at 0.50 (the S&C measurement); the 80 % figure is NOT reachable by
    # leading-edge accretion and is carried by the SLD ridge mechanism instead.
    lift_gain_range=(0.40, 0.50),
    lift_exponent_range=(0.28, 0.40),
    alpha_gain_range=(5.00, 8.00),
    alpha_exponent_range=(0.80, 1.00),
    moment_gain_range=(0.055, 0.095),
    oswald_gain_range=(0.38, 0.62),
    oswald_exponent_range=(0.03, 0.10),
    shape_floor_range=(0.002, 0.015),
    shed_threshold_range=(0.055, 0.090),
    shed_probability_range=(0.0001, 0.0006),
    shed_fraction_range=(0.10, 0.25),
    chord_range=(1.98, 1.98),
    leading_edge_scale_range=(0.070, 0.070),
)

# NOTE (2026-09-04): ``chord_m`` and ``leading_edge_scale_m`` are properties of
# the *airframe*, not of the icing physics, and are now pinned to the DHC-6
# (mean chord 1.98 m; impingement scale 0.070 m = 2 x leading-edge radius at an
# assumed 3.5% chord for a STOL high-lift section). They previously varied
# across families, which meant the surrogate was calibrated against DHC-6 flight
# data while the simulator used a 1.69 m chord - see PROVENANCE List B, item B1.
# They remain in this structure only to avoid a wider refactor; they should move
# to AircraftConfig. Physics families now vary *icing physics* alone.
#
# Identifiability caveat: no source reports wing accretion thickness, so
# ``leading_edge_scale_m`` (which sets beta_0, hence thickness) and the
# surrogate gains are NOT separately identifiable from the available data -
# only their product is constrained. The length scale is therefore pinned on
# geometric grounds and the gains fitted, not both fitted.

#: Registry consumed by the episode generator and the novelty-split builder.
PHYSICS_FAMILIES: dict[str, PhysicsFamily] = {
    "A": _FAMILY_A,
    "B": _FAMILY_B,
    "C": _FAMILY_C,
}


# ---------------------------------------------------------------------------
# Ice protection
# ---------------------------------------------------------------------------


@dataclass
class IceProtectionConfig:
    r"""Ice protection system configuration and state.

    Two mechanisms with distinct telemetry signatures:

    * **Anti-ice** (thermal, evaporative/running-wet): reduces the effective
      impingement rate by ``antiice_effectiveness`` once warmed up. Smooth
      suppression.
    * **De-ice boots**: cyclic mechanical shedding. Fires when accreted
      thickness exceeds ``boot_trigger_thickness_m`` and at least
      ``boot_cycle_period_s`` has elapsed, removing ``boot_shed_fraction`` of
      the mass and leaving residual ice. Produces a sawtooth drag signature.

    The distinction matters for XAI: a boot cycle causes an abrupt, *benign*
    drop in drag and AoA, which a naive change detector will flag. The
    ``boot_fired`` event is exposed in :class:`IceState` so that downstream
    monitors can be evaluated on whether they correctly attribute it.
    """

    antiice_available: bool = True
    antiice_on: bool = False
    antiice_effectiveness: float = 0.85
    antiice_warmup_s: float = 20.0

    boots_available: bool = True
    boots_on: bool = False
    boot_cycle_period_s: float = 180.0
    boot_trigger_thickness_m: float = 0.006
    boot_shed_fraction: float = 0.80
    boot_residual_fraction: float = 0.15

    # -- internal state ---------------------------------------------------
    _antiice_elapsed_s: float = 0.0
    _last_boot_time_s: float = -1e9


# ---------------------------------------------------------------------------
# State and outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IceState:
    """Instantaneous accretion ground truth.

    Written directly from simulator internals. This is the label source for the
    whole pipeline -- there is no external annotation step and therefore no
    annotation noise.
    """

    time_s: float
    ice_mass_per_area_kg_m2: float
    ice_thickness_m: float
    accretion_rate_kg_m2_s: float
    thickness_rate_m_s: float
    collection_efficiency: float
    freezing_fraction: float
    ice_density_kg_m3: float
    ice_type: IceType
    convective_h_w_m2_k: float
    total_accreted_mass_kg_m2: float
    shed_event: bool
    boot_fired: bool
    antiice_active: bool
    ridge_height_m: float = 0.0
    ridge_active: bool = False
    ice_regime: str = "leading_edge"

    @property
    def ice_thickness_mm(self) -> float:
        """Accreted thickness, mm -- the unit crews actually use."""
        return self.ice_thickness_m * 1e3

    @property
    def thickness_rate_mm_min(self) -> float:
        """Accretion rate, mm/min. The dominant rate feature for UC-3."""
        return self.thickness_rate_m_s * 1e3 * 60.0

    def as_dict(self) -> dict[str, float | str]:
        return {
            "time_s": self.time_s,
            "ice_thickness_mm": self.ice_thickness_mm,
            "ice_mass_per_area_kg_m2": self.ice_mass_per_area_kg_m2,
            "accretion_rate_mm_min": self.thickness_rate_mm_min,
            "collection_efficiency": self.collection_efficiency,
            "freezing_fraction": self.freezing_fraction,
            "ice_density_kg_m3": self.ice_density_kg_m3,
            "ice_type": self.ice_type,
            "convective_h_w_m2_k": self.convective_h_w_m2_k,
            "total_accreted_mass_kg_m2": self.total_accreted_mass_kg_m2,
            "shed_event": float(self.shed_event),
            "boot_fired": float(self.boot_fired),
            "antiice_active": float(self.antiice_active),
        }


@dataclass(frozen=True)
class AeroDegradation:
    r"""Aerodynamic penalty imposed by the current ice accretion.

    Consumed by :mod:`rlea.simdata.aircraft`. All quantities are *increments*
    applied to the clean-airframe coefficients.

    Attributes
    ----------
    delta_cd : float
        :math:`\Delta C_D`, added to parasite drag, dimensionless.
    delta_cl_max : float
        :math:`\Delta C_{L,max}`, negative, dimensionless.
    delta_alpha_stall_rad : float
        :math:`\Delta \alpha_{stall}`, negative, rad.
    delta_cm : float
        :math:`\Delta C_m`, pitching-moment shift driving trim/elevator change.
    shape_factor : float
        :math:`f_{shape}` actually applied, dimensionless. Exposed because it
        is the mechanism by which ice *type* (not just amount) drives penalty,
        and is therefore a quantity XAI attribution should be able to surface.
    oswald_factor : float
        Multiplier on the clean Oswald span efficiency :math:`e`, in (0, 1].
        Ice degrades induced drag as well as parasite drag: NASA TM-83564
        measured :math:`e` falling from 0.764 to 0.405 (-47%) on the DHC-6 in
        glaze icing, while the rime case stayed within measurement scatter.
        Induced drag dominates at high :math:`C_L`, so omitting this term
        biases exactly the low-speed, high-AoA regime that determines stall
        margin (ASSUMPTIONS A-ICE-11).
    """

    delta_cd: float
    delta_cl_max: float
    delta_alpha_stall_rad: float
    delta_cm: float
    shape_factor: float
    oswald_factor: float = 1.0

    @property
    def delta_alpha_stall_deg(self) -> float:
        """Stall-angle reduction, degrees."""
        return math.degrees(self.delta_alpha_stall_rad)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class IcingModel:
    r"""Stagnation-line Messinger accretion model with sampled physics.

    Parameters
    ----------
    params : PhysicsParameters
        Episode physics draw. Obtain via
        ``PHYSICS_FAMILIES["A"].sample(rng)``.
    protection : IceProtectionConfig, optional
        Ice protection configuration. Defaults to available-but-off.
    rng : numpy.random.Generator, optional
        Used only for stochastic shedding triggers.

    Examples
    --------
    >>> import numpy as np
    >>> from rlea.simdata.atmosphere import AtmosphereModel, AtmosphereConfig, WeatherCell
    >>> rng = np.random.default_rng(1)
    >>> params = PHYSICS_FAMILIES["A"].sample(rng)
    >>> model = IcingModel(params, rng=rng)
    >>> atm = AtmosphereModel(AtmosphereConfig(sea_level_temperature_k=278.0), rng=rng)
    >>> _ = atm.seed_stratiform_layer(altitude_m=2000.0, thickness_m=900.0,
    ...                               lwc_g_m3=0.5, mvd_um=20.0)
    >>> atm.step(60.0)
    >>> st = atm.sample(0.0, 0.0, 2000.0, true_airspeed_ms=120.0)
    >>> ice, aero = model.step(st, true_airspeed_ms=120.0, dt_s=0.1)
    >>> ice.freezing_fraction > 0.0
    True
    """

    def __init__(
        self,
        params: PhysicsParameters,
        protection: IceProtectionConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.params = params
        self.protection = protection if protection is not None else IceProtectionConfig()
        self.rng = rng if rng is not None else np.random.default_rng()

        self.time_s: float = 0.0
        self.ice_mass_per_area_kg_m2: float = 0.0
        self.ice_thickness_m: float = 0.0
        self.ridge_height_m: float = 0.0
        self.total_accreted_mass_kg_m2: float = 0.0
        self._last_density: float = RHO_GLAZE

    # -- component physics -------------------------------------------------

    def collection_efficiency(
        self, state: AtmosphericState, true_airspeed_ms: float
    ) -> float:
        r"""Stagnation-line collection efficiency :math:`\beta_0`.

        Langmuir--Blodgett correlation with the family's ``beta_scale`` and
        ``beta_mvd_exponent`` applied. See module docstring for the equations.

        Returns
        -------
        float
            :math:`\beta_0 \in [0, 1]`, dimensionless. Zero when there are no
            droplets or the flow is too slow for impingement.
        """
        p = self.params
        d = state.mvd_m
        v = max(true_airspeed_ms, 1.0)
        if d <= 0.0 or state.lwc_kg_m3 <= 0.0:
            return 0.0

        mu = dynamic_viscosity(state.temperature_k)
        rho_a = state.density_kg_m3
        # Impingement length scale, NOT the wing chord: the inertia parameter
        # is set by the body dimension droplets must flow around.
        c = p.leading_edge_scale_m

        # Inertia parameter and droplet Reynolds number
        k = (RHO_WATER * d * d * v) / (18.0 * mu * c)
        re_delta = (rho_a * d * v) / mu

        # Langmuir-Blodgett range correction
        lam_ratio = 1.0 / (
            0.8388 + 0.001483 * re_delta + 0.1847 * math.sqrt(max(re_delta, 0.0))
        )
        k0 = 0.125 + (k - 0.125) * lam_ratio
        if k0 <= 0.125:
            return 0.0

        x = 1.40 * (k0 - 0.125) ** 0.84
        beta = x / (1.0 + x)

        # Family-specific scaling: overall gain plus an MVD-relative exponent
        # that lets family C exaggerate large-droplet capture.
        beta *= p.beta_scale
        if p.beta_mvd_exponent != 0.0:
            beta *= (d / 20e-6) ** p.beta_mvd_exponent

        return float(min(max(beta, 0.0), 1.0))

    def convective_coefficient(
        self, state: AtmosphericState, true_airspeed_ms: float
    ) -> float:
        r"""Convective heat transfer coefficient :math:`h_c`, W m^-2 K^-1.

        Turbulent flat-plate correlation
        :math:`\mathrm{Nu} = 0.0296\,\mathrm{Re}_c^{0.8}\mathrm{Pr}^{1/3}`,
        augmented by the family's ice-roughness factor. Higher :math:`h_c`
        removes heat faster, raising the freezing fraction -- the roughness
        feedback that accelerates early accretion.
        """
        p = self.params
        mu = dynamic_viscosity(state.temperature_k)
        # Chord-referenced: the flat-plate correlation below is defined on the
        # chord Reynolds number, so it keeps chord rather than the impingement
        # scale. Documented simplification.
        re_c = state.density_kg_m3 * max(true_airspeed_ms, 1.0) * p.chord_m / mu
        nu = 0.0296 * re_c**0.8 * PR_AIR ** (1.0 / 3.0)
        h = nu * K_AIR / p.chord_m
        return float(h * p.roughness_augmentation)

    def freezing_fraction(
        self,
        state: AtmosphericState,
        true_airspeed_ms: float,
        beta: float,
        h_c: float,
    ) -> float:
        r"""Messinger freezing fraction :math:`n \in [0, 1]`.

        Ruff/LEWICE algebraic form; see module docstring for
        :math:`\phi`, :math:`\theta`, :math:`b`.

        Returns
        -------
        float
            0 means nothing freezes (too warm / all runback);
            1 means every impinging droplet freezes on contact (rime).
        """
        p = self.params
        t_inf = state.temperature_k
        v = max(true_airspeed_ms, 1.0)
        lwc = state.lwc_kg_m3

        if lwc <= 0.0 or beta <= 0.0 or t_inf >= T_FREEZE:
            return 0.0

        # phi: droplet sensible + kinetic energy term
        phi = (T_FREEZE - t_inf) - v * v / (2.0 * CP_WATER)
        # theta: air energy transfer term with adiabatic recovery heating
        theta = (T_FREEZE - t_inf) - RECOVERY_R * v * v / (2.0 * CP_AIR)
        # b: relative heat factor
        b = (lwc * v * beta * CP_WATER) / max(h_c, 1e-6)

        n = (CP_WATER / L_FUSION) * (phi + theta / max(b, 1e-9))
        n += p.freezing_fraction_bias
        return float(min(max(n, 0.0), 1.0))

    @staticmethod
    def classify_ice(freezing_fraction: float, accreting: bool) -> IceType:
        """Map freezing fraction to the operational ice-type vocabulary.

        Thresholds follow common usage: :math:`n \\ge 0.9` rime,
        :math:`n \\le 0.4` glaze, mixed in between. These are display
        categories, not physics -- the continuous :math:`n` is what drives the
        aerodynamics.
        """
        if not accreting:
            return "none"
        if freezing_fraction >= 0.9:
            return "rime"
        if freezing_fraction <= 0.4:
            return "glaze"
        return "mixed"

    def ice_density(self, freezing_fraction: float) -> float:
        r"""Accreted ice density :math:`\rho_{ice}(n)`, kg m^-3.

        Linear interpolation between the family's rime density (low :math:`n`
        limit is glaze, high :math:`n` limit is rime):

        .. math::
            \rho_{ice}(n) = \rho_{glaze} - n\,(\rho_{glaze} - \rho_{rime})
        """
        p = self.params
        return RHO_GLAZE - freezing_fraction * (RHO_GLAZE - p.rho_rime)

    # -- protection --------------------------------------------------------

    def _apply_protection(self, dt_s: float) -> tuple[float, bool, bool]:
        """Advance ice-protection state.

        Returns
        -------
        tuple
            ``(impingement_suppression, antiice_active, boot_fired)`` where
            ``impingement_suppression`` in [0, 1] multiplies the accretion rate.
        """
        prot = self.protection
        suppression = 1.0
        antiice_active = False
        boot_fired = False

        if prot.antiice_available and prot.antiice_on:
            prot._antiice_elapsed_s += dt_s
            ramp = min(prot._antiice_elapsed_s / max(prot.antiice_warmup_s, 1e-6), 1.0)
            suppression *= 1.0 - prot.antiice_effectiveness * ramp
            antiice_active = True
        else:
            prot._antiice_elapsed_s = 0.0

        if prot.boots_available and prot.boots_on:
            due = (self.time_s - prot._last_boot_time_s) >= prot.boot_cycle_period_s
            thick_enough = self.ice_thickness_m >= prot.boot_trigger_thickness_m
            if due and thick_enough:
                retained = max(1.0 - prot.boot_shed_fraction, prot.boot_residual_fraction)
                self.ice_mass_per_area_kg_m2 *= retained
                self.ice_thickness_m *= retained
                prot._last_boot_time_s = self.time_s
                boot_fired = True

        return suppression, antiice_active, boot_fired

    def _maybe_shed(self, dt_s: float) -> bool:
        """Stochastic self-shedding above the family's thickness threshold.

        Hazard model: once :math:`t_{ice} > t_{shed}`, shedding occurs in
        ``dt`` with probability :math:`\\lambda\\,dt`. Removes
        ``shed_fraction`` of the accreted mass.
        """
        p = self.params
        if self.ice_thickness_m <= p.shed_threshold_m:
            return False
        if self.rng.random() >= p.shed_probability_per_s * dt_s:
            return False
        retained = 1.0 - p.shed_fraction
        self.ice_mass_per_area_kg_m2 *= retained
        self.ice_thickness_m *= retained
        return True

    # -- aerodynamic surrogate --------------------------------------------

    def ridge_clmax_loss(self) -> float:
        r"""C_L,max loss carried by the SLD ridge, as a positive fraction.

        Linear in ridge height up to the single published measurement, then
        saturated. Claiming nothing beyond the measured point is deliberate:
        Cao Figs. 10-11 show C_L,max falling with ridge height and then
        flattening, but the curve itself is not tabulated, so a linear
        interpolation to the measured value is the minimal defensible form.
        """
        if self.ridge_height_m <= 0.0:
            return 0.0
        kc = self.ridge_height_m / max(self.params.chord_m, 1e-6)
        return RIDGE_CLMAX_LOSS * min(kc / RIDGE_REF_KC, 1.0)

    def aero_degradation(self, freezing_fraction: float) -> AeroDegradation:
        r"""Map current accretion to aerodynamic increments.

        Parameterised power-law surrogate; see module docstring section 5.
        Non-dimensional thickness :math:`\tau = t_{ice}/c`.

        The exponents satisfy :math:`p < 1`, so :math:`d(\Delta C_D)/d\tau
        \to \infty` as :math:`\tau \to 0`. That is intentional and physically
        motivated: the first roughness elements cost disproportionately more
        than equivalent later smooth growth. It also means the *rate* of ice
        accretion, not just its accumulated amount, carries early signal --
        which is precisely the structure the UC-3 factor-attribution work needs
        to be able to recover.
        """
        p = self.params
        ridge_loss = self.ridge_clmax_loss()
        tau = self.ice_thickness_m / max(p.chord_m, 1e-6)
        if tau <= 0.0:
            if ridge_loss <= 0.0:
                return AeroDegradation(0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
            return AeroDegradation(0.0, -ridge_loss, 0.0, -ridge_loss * 0.10, 1.0, 1.0)

        # Ice-type shape function: peaks at n = 0.2, falls away on both
        # sides, normalised to 1.0 at the peak (Cao et al. 2018 Sec. 5.4.2).
        n_pk = 0.2
        _u = freezing_fraction / n_pk
        f_shape = p.shape_floor + (1.0 - p.shape_floor) * _u * math.exp(1.0 - _u)
        r = tau / TAU_REF

        delta_cd = p.drag_gain * r**p.drag_exponent * f_shape
        delta_cl_max = -p.lift_gain * r**p.lift_exponent * f_shape
        delta_alpha = -math.radians(p.alpha_gain * r**p.alpha_exponent * f_shape)
        delta_cm = -p.moment_gain * r**p.drag_exponent * f_shape

        # Leading-edge accretion and an SLD ridge are two C_L,max losses on the
        # same wing, so they are NOT additive; the more severe one binds.
        if ridge_loss > 0.0:
            delta_cl_max = min(delta_cl_max, -ridge_loss)
            # Cao Figs. 12-13: a ridge produces an abrupt pitching-moment change
            # whose magnitude grows with ridge height. Scaled off the same law.
            delta_cm = min(delta_cm, -ridge_loss * 0.10)
        # NOTE: the ridge's own drag contribution is NOT modelled. Drag is the
        # one channel inverted directly from TM-83564 wing measurements and is
        # left untouched; the omission understates drag in ridge episodes and is
        # therefore conservative for detection (A-ICE-17).
        # Span-efficiency loss, floored so e stays strictly positive.
        e_loss = min(p.oswald_gain * r**p.oswald_exponent * f_shape, 0.85)

        return AeroDegradation(
            delta_cd=float(delta_cd),
            delta_cl_max=float(delta_cl_max),
            delta_alpha_stall_rad=float(delta_alpha),
            delta_cm=float(delta_cm),
            shape_factor=float(f_shape),
            oswald_factor=float(1.0 - e_loss),
        )

    # -- integration -------------------------------------------------------

    def step(
        self,
        state: AtmosphericState,
        true_airspeed_ms: float,
        dt_s: float,
    ) -> tuple[IceState, AeroDegradation]:
        r"""Advance accretion by ``dt_s`` and return ground truth + penalty.

        Sequence:

        1. collection efficiency :math:`\beta_0` and convective :math:`h_c`,
        2. freezing fraction :math:`n` from the Messinger balance,
        3. impingement :math:`\dot{m}_{imp} = \beta_0\,\mathrm{LWC}\,V`,
           suppressed by anti-ice,
        4. accretion :math:`\dot{m}_{ice} = n\,\dot{m}_{imp}`, integrated,
        5. thickness via :math:`\rho_{ice}(n)`,
        6. de-ice boot cycle and stochastic shedding,
        7. aerodynamic surrogate evaluation.

        Parameters
        ----------
        state : AtmosphericState
            Clean atmospheric sample at the aircraft position.
        true_airspeed_ms : float
            True airspeed, m s^-1.
        dt_s : float
            Timestep, s. Nominally 0.1 (10 Hz).

        Returns
        -------
        tuple of (IceState, AeroDegradation)
        """
        p = self.params

        beta = self.collection_efficiency(state, true_airspeed_ms)
        h_c = self.convective_coefficient(state, true_airspeed_ms)
        n = self.freezing_fraction(state, true_airspeed_ms, beta, h_c)

        suppression, antiice_active, boot_fired = self._apply_protection(dt_s)

        # Only supercooled water accretes.
        lwc_effective = state.supercooled_lwc_kg_m3
        m_imp = beta * lwc_effective * max(true_airspeed_ms, 0.0) * suppression
        m_ice_rate = n * m_imp

        rho_ice = self.ice_density(n) if m_ice_rate > 0.0 else self._last_density
        thickness_rate = m_ice_rate / max(rho_ice, 1.0)

        self.ice_mass_per_area_kg_m2 += m_ice_rate * dt_s
        self.ice_thickness_m += thickness_rate * dt_s
        self.total_accreted_mass_kg_m2 += m_ice_rate * dt_s
        self._last_density = rho_ice

        shed_event = self._maybe_shed(dt_s)

        # -- SLD ridge aft of the protected surface ------------------------
        # Grows only while all three hold: SLD droplets present, protection
        # actively keeping the leading edge clear, and supercooled water in the
        # airstream. The fraction of impingement falling aft of the protected
        # region ramps from zero at the Appendix C/O threshold of 50 um to unity
        # at RIDGE_MVD_FULL_UM. Ridge ice is runback glaze, hence RHO_GLAZE.
        #
        # Boots do not remove it: it forms aft of the protected span, which is
        # the entire mechanism. Nothing in this model removes it at all
        # (A-ICE-03, no sublimation), so a ridge persists for the episode.
        mvd_um = state.mvd_m * 1e6
        protection_running = antiice_active or self.protection.boots_on
        ridge_active = bool(
            state.sld_flag
            and protection_running
            and lwc_effective > 0.0
            and mvd_um > SLD_THRESHOLD_UM
        )
        if ridge_active:
            f_aft = min(
                (mvd_um - SLD_THRESHOLD_UM)
                / max(RIDGE_MVD_FULL_UM - SLD_THRESHOLD_UM, 1e-6),
                1.0,
            )
            ridge_rate = (
                f_aft * beta * lwc_effective
                * max(true_airspeed_ms, 0.0) / RHO_GLAZE
            )
            self.ridge_height_m += ridge_rate * dt_s

        self.time_s += dt_s

        ice_state = IceState(
            time_s=self.time_s,
            ice_mass_per_area_kg_m2=self.ice_mass_per_area_kg_m2,
            ice_thickness_m=self.ice_thickness_m,
            accretion_rate_kg_m2_s=m_ice_rate,
            thickness_rate_m_s=thickness_rate,
            collection_efficiency=beta,
            freezing_fraction=n,
            ice_density_kg_m3=rho_ice,
            ice_type=self.classify_ice(n, accreting=m_ice_rate > 1e-9),
            convective_h_w_m2_k=h_c,
            total_accreted_mass_kg_m2=self.total_accreted_mass_kg_m2,
            shed_event=shed_event,
            boot_fired=boot_fired,
            antiice_active=antiice_active,
            ridge_height_m=self.ridge_height_m,
            ridge_active=ridge_active,
            ice_regime=("sld_ridge" if self.ridge_height_m > 0.0 else "leading_edge"),
        )
        return ice_state, self.aero_degradation(n)

    def reset(self) -> None:
        """Clear accreted ice and protection state (new episode, same physics)."""
        self.time_s = 0.0
        self.ice_mass_per_area_kg_m2 = 0.0
        self.ice_thickness_m = 0.0
        self.ridge_height_m = 0.0
        self.total_accreted_mass_kg_m2 = 0.0
        self._last_density = RHO_GLAZE
        self.protection._antiice_elapsed_s = 0.0
        self.protection._last_boot_time_s = -1e9
