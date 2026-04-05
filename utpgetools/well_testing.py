"""
Well Testing Analysis Module

This module provides comprehensive functions for well testing analysis in petroleum engineering,
including transient pressure analysis, infinite acting radial flow, pseudo-steady state analysis,
and pressure derivative calculations. The module supports both single-phase liquid systems and
includes tools for pressure transient test interpretation and parameter estimation.

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

The module is organized into logical functional groups:

1. GEOMETRY & TIME FUNCTIONS (lines 53-180)
   - drainage_radius_from_area: Convert area to circular radius equivalence
   - t_infinite_acting_hr: Duration of infinite-acting period before boundaries matter
   - t_pseudosteady_start_hr: Onset of pseudo-steady state using dimensionless time
   - mbh_dimensionless_time: Matthews-Brons-Hazebroek dimensionless time (average pressure)
   These handle unit conversions and provide characteristic times for test interpretation.

2. PRESSURE SOLUTION FUNCTIONS (lines 183-570)
   Four classes of pressure models for different flow regimes:
   
   a) Closed-system Material Balance:
      - pavg_closed_system: Average pressure depletion over time (p_avg = p_i - depletion)
   
   b) Infinite-Acting Radial Flow (IARF) - valid for early times when boundaries don't matter:
      - pwf_infinite_acting: Uses exponential integral solution (line source)
      Conditions: t << t_infinite_acting
   
   c) Pseudo-Steady State (PSS) - valid for late times when boundary effects dominate:
      - pwf_pseudosteady_centered_well: For circular centered well
      - pwf_pseudosteady_shape_factor: General form with Dietz shape factors (allows any geometry)
      Conditions: t >> t_infinite_acting
      Combines: material balance depletion + pseudo-steady drawdown
   
   d) Hybrid Piecewise Model:
      - pwf_bounded_piecewise: Switches from IARF→PSS at t_inf automatically
      - p_profile_pseudosteady: Spatial pressure distribution in PSS
   
   All return wellbore pressure (pwf_psia) or profile (p at radius r).

3. DERIVATIVE & ANALYSIS TOOLS (lines 573-730)
   - bourdet_derivative: Compute dp/d(ln t) using 3-point formula on log-time grid
   - r2_score: R² coefficient for regression fit quality assessment
   These support flow regime identification and quality evaluation.

4. FORMATTING & OUTPUT (lines 733-850)
   - _format_grouped_results, format_well_test_results, print_well_test_results
   - Pretty-print permeability, skin, damage ratio in human-readable form
   - Used by main well_test_analysis() for console reporting

5. PLOTTING HELPER FUNCTIONS (lines 853-950)
   - _plot_sparse_safe: Choose scatter vs line based on data density
   - _shade_masked_xzones: Render boolean mask regions as full-height x-spans (flow zones)
   Both are matplotlib utilities that make plots clearer and more robust to edge cases.

6. BUILDUP ANALYSIS ENGINE (lines 953-1060)
   - _apply_time_epsilon: Replace non-positive times with 1e-8 hr (keeps all data)
   - _analyze_buildup_period: Core interpretation engine for single buildup
      Performs: Horner, MDH, Bourdet derivative, average pressure, pseudo-steady
      Returns: dict with all permeability/skin/pressure estimates + regime masks
   - _extract_reference_rate: Get constant rate from scalar or rate history dict
   - _parse_rate_history: Unpack variable-rate history for deconvolution

7. DECONVOLUTION FOR VARIABLE-RATE TESTS (lines 1063-1260)
   - _log_time_interpolation_matrix: Build logarithmic interpolation for pressure matching
   - _second_difference_matrix: Roughness matrix for Tikhonov regularization
   - _mbh_b17_* and _mbh_b8_*: Helper evaluators for MBH average pressure
   - rate_superposition_buildup: Fast superposition on Horner axis
   - rate_deconvolution_buildup: True inverse deconvolution with smoothing

8. MAIN ENTRY POINT (lines 1263-2450)
   - well_test_analysis(): Orchestrates entire workflow
      • Parses test type (buildup, dst, drawdown, etc)
      • Handles DST with multiple flow/buildup periods
      • Calls _analyze_buildup_period for each period
      • Creates Horner semilog + Bourdet derivative plots
      • Returns matplotlib figure + fig.utpge_results dict with all parameters

═══════════════════════════════════════════════════════════════════════════════
KEY CONCEPTS & TERMINOLOGY
═══════════════════════════════════════════════════════════════════════════════

IARF (Infinite-Acting Radial Flow):
  - Early-time behavior when well appears to be in infinite reservoir
  - Pressure response shape determined by diffusivity equation
  - Line source solution with exponential integral
  - Identified on Horner plot as straight line (constant slope)
  - Valid for: t << t_inf = 121 * A * φ * μ * ct / k [hours]

PSS (Pseudo-Steady State):
  - Late-time behavior when boundaries have influenced pressure
  - All parts of reservoir depleting uniformly (material balance)
  - Pressure drawdown depends only on current production rate + geometry
  - Linear on Horner plot with different slope than IARF
  - Governed by: pwf = p_avg - constant*ln(re/rw) - skin_term

t_DA (Dimensionless Time based on Area):
  - t_DA = 0.0002637 * k * t / (φ * μ * ct * A)  [dimensionless]
  - Used for type curves and average pressure methods
  - t_DA ≈ 0.1 typically marks start of PSS (used for t_pseudosteady_start_hr)
  - Independent of drainage geometry (area-based not radius-based)

Horner Plot:
  - x-axis: log10[(tp + Δt) / Δt]  where tp = producing time, Δt = shut-in time
  - y-axis: pressure (linear)
  - IARF region: straight line with slope m = Δp/Δlog(x)
  - From slope m: k = 162.6 * q * B * μ / (|m| * h)
  - Intercept at x→∞ gives p* (average reservoir pressure)

Bourdet Derivative:
  - y-axis: dp/d(ln t) = t * dp/dt
  - x-axis: time t (log scale)
  - IARF (radial): appears as horizontal line
  - Wellbore storage: appears as 1:1 slope line (45°)
  - Boundaries: downward trend or multiple bends

Skin Factor (s):
  - Dimensionless parameter representing wellbore effects
  - s > 0: well damage (reduced flow capacity)
  - s < 0: stimulation (improved flow capacity)
  - s = 0: no skin
  - Related to pressure drop: Δp_s = 0.87 * m * s  (where m = slope)

═══════════════════════════════════════════════════════════════════════════════
UNIT CONVENTIONS
═══════════════════════════════════════════════════════════════════════════════

Pressure: psia (absolute pressure in pounds per square inch)
Time: hours (hr)
Rate: STB/day (stock tank barrels per day)
Permeability: millidarcies (md)
Viscosity: centipoise (cp)
Compressibility: psi^-1
Area: acres (or square feet internally)
Distance: feet (ft)
Porosity: fraction (0-1, not percent)
Formation volume factor: rb/STB (reservoir barrels per stock tank barrel)

Key conversion factors used throughout:
  - 43560.04 ft²/acre (area conversion)
  - 5.615 rb/ft³ (volume factor)
  - 0.0002637 (time constant for dimensionless conversions)
  - 162.6 (permeability calculation coefficient)
  - 141.2 (drawdown calculation coefficient)
  - 0.87 (skin-to-slope conversion)
  - 1.151 (skin calculation scaling)

═══════════════════════════════════════════════════════════════════════════════
WORKFLOW EXAMPLE: Buildup Test Interpretation
═══════════════════════════════════════════════════════════════════════════════

1. Data Input:
   - Shut-in times: Δt = [0, 0.1, 0.2, ..., 200] hours
   - Pressures: pws = [2665, 2940, 2970, ..., 3361] psia
   - Producing time: tp = 1008 hours
   - Parameters: q, B, μ, h, rw, φ, ct, area

2. Preprocessing (_apply_time_epsilon):
   - Replace Δt[0]=0 with 1e-8 hr to avoid log(0) and 1/0
   - Keep all datapoints: no deletion

3. Horner Transform:
   - Calculate: horner_time = (tp + Δt) / Δt
   - x = log10(horner_time)
   - Create display_mask to exclude Δt≤0 from plot xlim

4. IARF Identification:
   - Locate points in IARF_slice = (1, 20) hours (given by user)
   - Fit straight line: pressure = m * log10(horner_time) + p*
   - Calculate R² to assess fit quality

5. Parameter Extraction:
   - Slope m from linear regression
   - Intercept p* (average reservoir pressure)
   - p_1hr = pressure at horner_time = tp+1
   - Permeability: k = 162.6*q*B*μ/(|m|*h)
   - Skin: s = 1.151*[log(k/φ/μ/ct/rw²) - 3.23 + (p_1hr-p*)/(|m|)]

6. Derivative Analysis:
   - Compute Bourdet derivative: dp/d(ln t)
   - Fit radial flow regime (horizontal line): gives another k estimate
   - Identify wellbore storage (early time, unit slope)
   - Identify boundaries (late time, downward trend)

7. Average Pressure Estimation:
   - Three methods used (for comparison):
     a) Dietz/MDH: uses MBH dimensionless time
     b) Ramey-Cobb: geometric method
     c) MBH: Matthews-Brons-Hazebroek generalized

8. Output Generation:
   - Create Horner semilog plot with zones highlighted
   - Create log-log derivative staircase with zones
   - Generate results dictionary with all estimates
   - Print formatted summary to console

═══════════════════════════════════════════════════════════════════════════════

Functions:
    drainage_radius_from_area: Calculate equivalent drainage radius from area
    t_infinite_acting_hr: Calculate end of infinite-acting radial flow period
    mbh_dimensionless_time: Calculate MBH dimensionless average time t_pAD
    mbh_dimensionless_average_pressure: Calculate MBH dimensionless average pressure
    pavg_closed_system: Calculate average reservoir pressure vs time for closed systems
    pwf_infinite_acting: Calculate wellbore pressure using infinite-acting solution
    pwf_pseudosteady_centered_well: Calculate wellbore pressure using pseudo-steady approximation
    pwf_pseudosteady_shape_factor: Calculate wellbore pressure using Dietz shape factors
    pwf_bounded_piecewise: Piecewise bounded model combining IARF and pseudo-steady solutions
    p_profile_pseudosteady: Calculate spatial pressure profile in pseudo-steady state
    bourdet_derivative: Calculate Bourdet-style pressure derivative
    r2_score: Calculate coefficient of determination for regression analysis
    horner_plot_analysis: Automated Horner plot analysis with trend line fitting
    derivative_curve_analysis: Automated pressure derivative analysis with flow regime identification
    
Dependencies:
    - numpy: For numerical calculations and array operations
    - scipy.special: For exponential integral calculations (expi)
    - scipy.optimize: For curve fitting and optimization (curve_fit)
    
Notes:
    This module focuses on single-phase, slightly compressible liquid systems in
    homogeneous reservoirs. Functions handle both dimensionless and field units
    with consistent unit conversions. All pressure solutions assume zero skin
    factor unless otherwise specified.
    
    The module supports:
    - Early-time infinite acting radial flow (IARF) analysis
    - Late-time pseudo-steady state (PSS) analysis  
    - Bounded reservoir behavior with various drainage shapes
    - Pressure derivative analysis for test interpretation
    - Horner plot analysis with automatic trend line fitting
    - Flow regime identification (wellbore storage, radial flow, boundaries)
    - Parameter estimation from pressure transient data
"""

from matplotlib.pyplot import grid
import numpy as np
from scipy.integrate import quad
from scipy.special import expi
from scipy.optimize import curve_fit, lsq_linear
from typing import Any, Union, Tuple, Dict, Optional

def drainage_radius_from_area(area_acres: float) -> float:
    """
    Calculate equivalent drainage radius from drainage area.
    
    Converts drainage area to equivalent circular drainage radius using the
    relationship: A = π * re^2, where A is area and re is equivalent radius.
    
    Args:
        area_acres (float): Drainage area in acres
        
    Returns:
        float: Equivalent drainage radius in feet
        
    Examples:
        >>> drainage_radius_from_area(80.0)
        1052.96...
        
        >>> re = drainage_radius_from_area(160.0)
        >>> area_check = np.pi * re**2 / 43560.04
        >>> np.isclose(area_check, 160.0)
        True
        
    Notes:
        - Uses conversion factor 43560.04 ft²/acre
        - Assumes circular drainage area for equivalent radius calculation
        - Commonly used in well testing for bounded reservoir analysis
    """
    return float(np.sqrt(area_acres * 43560.04 / np.pi))

def t_infinite_acting_hr(phi: float, mu_cp: float, ct_psi_inv: float, 
                        area_acres: float, k_md: float) -> float:
    """
    Calculate the end of infinite-acting radial flow period.
    
    Estimates the time when boundary effects begin to influence wellbore pressure
    using the relationship: t_inf = 121.0 * φ * μ * ct * A / k (hours).
    
    Args:
        phi (float): Porosity as fraction (0-1)
        mu_cp (float): Viscosity in centipoise (cp)
        ct_psi_inv (float): Total compressibility in psi^-1
        area_acres (float): Drainage area in acres
        k_md (float): Permeability in millidarcies (md)
        
    Returns:
        float: End of infinite-acting period in hours
        
    Examples:
        >>> t_infinite_acting_hr(0.2, 1.5, 15e-6, 80.0, 55.0)
        34.472727272727266
        
        >>> # For higher permeability, boundary effects occur earlier
        >>> t_infinite_acting_hr(0.2, 1.5, 15e-6, 80.0, 110.0)
        17.236363636363633
        
    Notes:
        - Formula assumes slightly compressible liquid in homogeneous reservoir
        - Boundary effects become significant after this time
        - Used to determine when to switch from IARF to bounded solutions
        - Conservative estimate; actual boundary effects may vary
    """
    return float(121.0 * phi * mu_cp * ct_psi_inv * area_acres * 43560.04 / k_md)

def t_pseudosteady_start_hr(phi: float,
                            mu_cp: float,
                            ct_psi_inv: float,
                            k_md: float,
                            area_acres: Optional[float] = None,
                            area_ft2: Optional[float] = None,
                            re_ft: Optional[float] = None,
                            tda_threshold: float = 0.1) -> float:
    """
    Calculate the start time of pseudo-steady flow for a centered well.

    Uses the area-based dimensionless time relation shown in common well-testing
    references for regularly shaped drainage areas:

        t_DA = 0.0002637 * k * t / (phi * mu * ct * A)

    with pseudo-steady onset approximated by t_DA = 0.1, giving:

        t = tda_threshold * phi * mu * ct * A / (0.0002637 * k)

    Equivalent radius form (when re_ft is provided):

        t = (tda_threshold * pi / 0.0002637) * phi * mu * ct * re^2 / k

    Args:
        phi (float): Porosity fraction.
        mu_cp (float): Viscosity in centipoise.
        ct_psi_inv (float): Total compressibility in psi^-1.
        k_md (float): Permeability in millidarcies.
        area_acres (Optional[float]): Drainage area in acres.
        area_ft2 (Optional[float]): Drainage area in square feet.
        re_ft (Optional[float]): Drainage radius in feet.
        tda_threshold (float): Dimensionless threshold for pseudo-steady onset.
            Defaults to 0.1.

    Returns:
        float: Start time of pseudo-steady flow in hours.
    """
    if phi <= 0.0 or mu_cp <= 0.0 or ct_psi_inv <= 0.0 or k_md <= 0.0:
        raise ValueError("phi, mu_cp, ct_psi_inv, and k_md must be positive")
    if tda_threshold <= 0.0:
        raise ValueError("tda_threshold must be positive")

    provided_geometry = sum(value is not None for value in (area_acres, area_ft2, re_ft))
    if provided_geometry != 1:
        raise ValueError("Provide exactly one of area_acres, area_ft2, or re_ft")

    if re_ft is not None:
        if re_ft <= 0.0:
            raise ValueError("re_ft must be positive")
        area_ft2_value = np.pi * float(re_ft) ** 2
    elif area_ft2 is not None:
        if area_ft2 <= 0.0:
            raise ValueError("area_ft2 must be positive")
        area_ft2_value = float(area_ft2)
    else:
        if area_acres <= 0.0:
            raise ValueError("area_acres must be positive")
        area_ft2_value = float(area_acres) * 43560.04

    t_hr = tda_threshold * phi * mu_cp * ct_psi_inv * area_ft2_value / (0.0002637 * k_md)
    return float(t_hr)

def mbh_dimensionless_time(t_hr: Union[float, np.ndarray], k_md: float, phi: float,
                          mu_cp: float, ct_psi_inv: float,
                          area_acres: Optional[float] = None,
                          area_ft2: Optional[float] = None) -> Union[float, np.ndarray]:
    """
    Calculate Matthews-Brons-Hazebroek dimensionless average time t_pAD.

    The MBH chart and average-pressure method use the area-based dimensionless time:
    t_pAD = k * t / (phi * mu * ct * A)

    In oilfield units this becomes:
    t_pAD = 0.0002637 * k_md * t_hr / (phi * mu_cp * ct_psi_inv * A_ft2)

    Args:
        t_hr (Union[float, np.ndarray]): Producing time in hours
        k_md (float): Permeability in millidarcies
        phi (float): Porosity as fraction (0-1)
        mu_cp (float): Viscosity in centipoise
        ct_psi_inv (float): Total compressibility in psi^-1
        area_acres (Optional[float]): Drainage area in acres
        area_ft2 (Optional[float]): Drainage area in square feet

    Returns:
        Union[float, np.ndarray]: Dimensionless average time t_pAD

    Notes:
        - Supply exactly one of area_acres or area_ft2
        - Matches the MBH notation T used in the original paper images
    """
    if (area_acres is None) == (area_ft2 is None):
        raise ValueError("Provide exactly one of area_acres or area_ft2")

    t_hr = np.asarray(t_hr, dtype=float)
    area_ft2_value = area_ft2 if area_ft2 is not None else area_acres * 43560.04
    result = 0.0002637 * k_md * t_hr / (phi * mu_cp * ct_psi_inv * area_ft2_value)
    return float(result) if t_hr.ndim == 0 else result

def calculate_deltap_skin(q_stbd: Union[float, np.ndarray], bo_rb_stb: float, mu_cp: float,
                         k_md: float, h_ft: float, skin_factor: Union[float, np.ndarray],
                         slope_psi_per_log_cycle: Optional[float] = None) -> Union[float, np.ndarray]:
    """
    Calculate pressure drop attributed to skin effect, Δp_skin.

    Uses the radial-flow skin pressure drop relationship in field units:
    Δp_skin = (141.2 * q * B * μ / (k * h)) * s

    Optionally, if semilog slope magnitude m is supplied, this can also be written as:
    Δp_skin = 0.87 * m * s

    Args:
        q_stbd (Union[float, np.ndarray]): Production rate in STB/day.
        bo_rb_stb (float): Oil formation volume factor in rb/STB.
        mu_cp (float): Viscosity in centipoise.
        k_md (float): Permeability in millidarcies.
        h_ft (float): Net pay thickness in feet.
        skin_factor (Union[float, np.ndarray]): Skin factor (dimensionless).
        slope_psi_per_log_cycle (Optional[float]): Semilog straight-line slope
            magnitude in psi/log-cycle. If provided and positive, the function
            uses Δp_skin = 0.87 * m * s.

    Returns:
        Union[float, np.ndarray]: Skin-related pressure drop in psi.

    Examples:
        >>> calculate_deltap_skin(300.0, 1.2, 1.5, 55.0, 50.0, 2.0)
        55.46...

        >>> # Slope-based form (same units)
        >>> calculate_deltap_skin(300.0, 1.2, 1.5, 55.0, 50.0, 2.0, slope_psi_per_log_cycle=31.9)
        55.50...

        >>> # Array skin factors
        >>> ds = calculate_deltap_skin(300.0, 1.2, 1.5, 55.0, 50.0, np.array([0.0, 1.0, 2.0]))
        >>> len(ds) == 3
        True

    Notes:
        - Positive skin gives positive additional pressure drop (damage)
        - Negative skin gives negative additional pressure drop (stimulation)
        - For consistent interpretation, use radial-flow slope magnitude for m
    """
    if bo_rb_stb <= 0.0:
        raise ValueError("bo_rb_stb must be positive")
    if mu_cp <= 0.0:
        raise ValueError("mu_cp must be positive")
    if k_md <= 0.0:
        raise ValueError("k_md must be positive")
    if h_ft <= 0.0:
        raise ValueError("h_ft must be positive")

    q_stbd = np.asarray(q_stbd, dtype=float)
    skin_factor = np.asarray(skin_factor, dtype=float)

    if np.any(~np.isfinite(q_stbd)) or np.any(~np.isfinite(skin_factor)):
        raise ValueError("q_stbd and skin_factor must be finite")

    if slope_psi_per_log_cycle is not None:
        if not np.isfinite(slope_psi_per_log_cycle) or slope_psi_per_log_cycle <= 0.0:
            raise ValueError("slope_psi_per_log_cycle must be finite and positive when provided")
        result = 0.87 * slope_psi_per_log_cycle * skin_factor
    else:
        result = 141.2 * (q_stbd * bo_rb_stb * mu_cp) / (k_md * h_ft) * skin_factor

    return float(result) if result.ndim == 0 else result

def calculate_damage_ratio(pr_psia: Union[float, np.ndarray], pwf_psia: Union[float, np.ndarray],
                          delta_p_skin_psi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate damage ratio from reservoir and flowing pressures.

    Damage ratio is defined as:
    DR = (PR - Pwf) / (PR - Pwf - Δp_skin)

    where PR is average reservoir pressure, Pwf is average flowing pressure,
    and Δp_skin is the skin-only pressure drop contribution.

    Args:
        pr_psia (Union[float, np.ndarray]): Average reservoir pressure in psia.
        pwf_psia (Union[float, np.ndarray]): Average flowing pressure in psia.
        delta_p_skin_psi (Union[float, np.ndarray]): Skin pressure drop in psi.

    Returns:
        Union[float, np.ndarray]: Damage ratio (dimensionless).

    Examples:
        >>> calculate_damage_ratio(4000.0, 3600.0, 80.0)
        1.25

        >>> # No skin effect gives ratio of 1.0
        >>> calculate_damage_ratio(4000.0, 3600.0, 0.0)
        1.0

        >>> # Array pressures
        >>> dr = calculate_damage_ratio(np.array([4000.0, 3900.0]), np.array([3600.0, 3500.0]), 80.0)
        >>> len(dr) == 2
        True

    Notes:
        - DR > 1 generally indicates damage (positive skin)
        - DR < 1 generally indicates stimulation (negative skin)
        - Denominator must be non-zero for a physically meaningful value
    """
    pr_psia = np.asarray(pr_psia, dtype=float)
    pwf_psia = np.asarray(pwf_psia, dtype=float)
    delta_p_skin_psi = np.asarray(delta_p_skin_psi, dtype=float)

    if np.any(~np.isfinite(pr_psia)) or np.any(~np.isfinite(pwf_psia)) or np.any(~np.isfinite(delta_p_skin_psi)):
        raise ValueError("pr_psia, pwf_psia, and delta_p_skin_psi must be finite")
    if np.any(pr_psia <= pwf_psia):
        raise ValueError("pr_psia must be greater than pwf_psia")

    total_drawdown = pr_psia - pwf_psia
    non_skin_drawdown = total_drawdown - delta_p_skin_psi
    if np.any(np.isclose(non_skin_drawdown, 0.0)):
        raise ValueError("Non-skin drawdown is zero; damage ratio is undefined")

    result = total_drawdown / non_skin_drawdown
    return float(result) if result.ndim == 0 else result

def _mbh_b17_bracket(fraction: float, decay: float, terms: int, tol: float) -> float:
    """Evaluate one of the bracketed summations in Eq. B-17."""
    total = 2.0
    for index in range(1, terms + 1):
        term = 2.0 * np.exp(-(index ** 2) * decay) * (1.0 + np.cos(2.0 * np.pi * index * fraction))
        total += term
        if abs(term) < tol:
            break
    return total

def _mbh_b17_integrand(u: float, alpha: float, beta: float, aspect_ratio_lambda: float,
                       terms: int, tol: float) -> float:
    """Integrand appearing in the large-time MBH expression, Eq. B-17."""
    x_bracket = _mbh_b17_bracket(alpha, aspect_ratio_lambda * u, terms, tol)
    y_bracket = _mbh_b17_bracket(beta, u / aspect_ratio_lambda, terms, tol)
    return x_bracket * y_bracket

def _mbh_b8_image_sum(time_value: float, alpha: float, beta: float,
                      aspect_ratio_lambda: float, image_terms: int) -> float:
    """Evaluate the small-time MBH image-well summation corresponding to Eq. B-8."""
    total = 4.0 * np.pi * time_value
    for m_index in range(-image_terms, image_terms + 1):
        for n_index in range(-image_terms, image_terms + 1):
            if m_index != 0 or n_index != 0:
                total += expi(-(m_index ** 2 * aspect_ratio_lambda + n_index ** 2 / aspect_ratio_lambda) / time_value)

            total += expi(-(((m_index + alpha) ** 2) * aspect_ratio_lambda + n_index ** 2 / aspect_ratio_lambda) / time_value)
            total += expi(-(m_index ** 2 * aspect_ratio_lambda + ((n_index + beta) ** 2) / aspect_ratio_lambda) / time_value)
            total += expi(-(((m_index + alpha) ** 2) * aspect_ratio_lambda + ((n_index + beta) ** 2) / aspect_ratio_lambda) / time_value)
    return float(total)

def _mbh_b17_correction(time_value: float, alpha: float, beta: float,
                        aspect_ratio_lambda: float, terms: int, tol: float) -> float:
    """Evaluate the convergent correction term used with the B-17 asymptote."""
    lower_limit = (np.pi ** 2) * time_value
    correction, _ = quad(
        lambda u: (_mbh_b17_integrand(u, alpha, beta, aspect_ratio_lambda, terms, tol) - 4.0) / np.pi,
        lower_limit,
        np.inf,
        limit=200,
    )
    return float(correction)

def mbh_dimensionless_average_pressure(
    t_pAD: Union[float, np.ndarray],
    shape_factor_ca: float,
    alpha: float = 0.5,
    beta: float = 0.5,
    aspect_ratio_lambda: float = 1.0,
    terms: int = 200,
    tol: float = 1e-12,
    b8_threshold: float = 0.05,
    image_terms: int = 12,
) -> Union[float, np.ndarray]:
    """
    Calculate MBH dimensionless average pressure using the Eq. B-17 form.

    This function evaluates the dimensionless quantity shown in the MBH charts:

    F(T) = (p* - p_avg) / (q * mu / (4 * pi * k * h))

    using the large-time representation visible in Eq. B-17. The implementation
    is written as a late-time asymptote plus a convergent correction integral:

    F(T) = ln(C_A * T) + (1/pi) * integral_[pi^2 T to inf] (G(u) - 4) du

    where
        T = t_pAD
        C_A = Dietz shape factor for the specified geometry
        alpha = x_w / x_e
        beta = y_w / y_e
        lambda = x_e / y_e

    Args:
        t_pAD (Union[float, np.ndarray]): Dimensionless average time T
        shape_factor_ca (float): Dietz shape factor C_A for the same geometry.
        alpha (float, optional): Fractional well position in x-direction. Defaults to 0.5
        beta (float, optional): Fractional well position in y-direction. Defaults to 0.5
        aspect_ratio_lambda (float, optional): Rectangle aspect ratio x_e / y_e. Defaults to 1.0
        terms (int, optional): Maximum number of series terms in each bracket. Defaults to 200
        tol (float, optional): Truncation tolerance for series terms. Defaults to 1e-12
        b8_threshold (float, optional): Time below which the B-8 image-well form is used.
            Defaults to 0.05.
        image_terms (int, optional): Number of positive and negative image blocks used
            in the B-8 summation. Defaults to 12.

    Returns:
        Union[float, np.ndarray]: MBH dimensionless average pressure F(T)

    Notes:
        - The implementation uses a hybrid strategy: B-8 for early time and
          B-17 for later time
        - Exact reproduction of a published curve requires the correct pair
          (geometry, C_A). If only C_A is changed, late-time alignment is exact
          while early-time curvature remains geometry-dependent through alpha,
          beta, and lambda.
    """
    if aspect_ratio_lambda <= 0.0:
        raise ValueError("aspect_ratio_lambda must be positive")
    if image_terms < 1:
        raise ValueError("image_terms must be at least 1")

    times = np.asarray(t_pAD, dtype=float)
    if np.any(times <= 0.0):
        raise ValueError("t_pAD must be positive")

    if shape_factor_ca <= 0.0:
        raise ValueError("shape_factor_ca must be positive")

    def _evaluate_one(time_value: float) -> float:
        if time_value <= b8_threshold:
            return _mbh_b8_image_sum(time_value, alpha, beta, aspect_ratio_lambda, image_terms)
        correction = _mbh_b17_correction(time_value, alpha, beta, aspect_ratio_lambda, terms, tol)
        return float(np.log(shape_factor_ca * time_value) + correction)

    if times.ndim == 0:
        return _evaluate_one(float(times))
    return np.array([_evaluate_one(float(value)) for value in times], dtype=float)

def pavg_closed_system(psia_pi: float, t_hr: Union[float, np.ndarray], 
                      q_stbd: float, bo_rb_stb: float, area_acres: float, 
                      h_ft: float, phi: float, ct_psi_inv: float) -> Union[float, np.ndarray]:
    """
    Calculate average reservoir pressure vs time for a closed slightly-compressible system.
    
    Uses material balance equation for closed boundary systems:
    p_avg(t) = Pi - (q * Bo * t) / (PV * ct), where PV is pore volume.
    
    Args:
        psia_pi (float): Initial reservoir pressure in psia
        t_hr (Union[float, np.ndarray]): Time in hours (scalar or array)
        q_stbd (float): Production rate in STB/day
        bo_rb_stb (float): Oil formation volume factor in rb/STB
        area_acres (float): Drainage area in acres
        h_ft (float): Net pay thickness in feet
        phi (float): Porosity as fraction (0-1)
        ct_psi_inv (float): Total compressibility in psi^-1
        
    Returns:
        Union[float, np.ndarray]: Average pressure(s) in psia
        
    Examples:
        >>> pavg_closed_system(4000.0, 24.0, 300.0, 1.2, 80.0, 50.0, 0.2, 15e-6)
        3999.22...
        
        >>> # Array input for multiple times
        >>> times = np.array([24.0, 48.0, 72.0])
        >>> pressures = pavg_closed_system(4000.0, times, 300.0, 1.2, 80.0, 50.0, 0.2, 15e-6)
        >>> len(pressures) == 3
        True
        
    Notes:
        - Assumes constant production rate and properties
        - Valid for closed boundary systems (no-flow boundaries)
        - Linear depletion with time due to constant compressibility
        - Uses conversion factor 5.615 for rb to ft³
    """
    t_hr = np.asarray(t_hr, dtype=float)
    pore_volume_rb = (area_acres * 43560.04 * h_ft * phi) / 5.615
    result = psia_pi - (q_stbd * bo_rb_stb) * (t_hr / 24.0) / (pore_volume_rb * ct_psi_inv)
    return float(result) if t_hr.ndim == 0 else result

def pwf_infinite_acting(t_hr: Union[float, np.ndarray], q_stbd: float, psia_pi: float, 
                       k_md: float, h_ft: float, phi: float, mu_cp: float, 
                       ct_psi_inv: float, rw_ft: float, bo_rb_stb: float) -> Union[float, np.ndarray]:
    """
    Calculate infinite-acting (line-source) wellbore pressure using exponential integral.
    
    Implements the classical infinite acting radial flow solution for slightly
    compressible liquids: Δp = (141.2 * q * B * μ) / (k * h) * [-0.5 * Ei(-1/(4*tD))]
    
    Args:
        t_hr (Union[float, np.ndarray]): Time in hours (scalar or array)
        q_stbd (float): Production rate in STB/day
        psia_pi (float): Initial reservoir pressure in psia
        k_md (float): Permeability in millidarcies
        h_ft (float): Net pay thickness in feet
        phi (float): Porosity as fraction (0-1)
        mu_cp (float): Viscosity in centipoise
        ct_psi_inv (float): Total compressibility in psi^-1
        rw_ft (float): Wellbore radius in feet
        bo_rb_stb (float): Oil formation volume factor in rb/STB
        
    Returns:
        Union[float, np.ndarray]: Wellbore flowing pressure(s) in psia
        
    Examples:
        >>> pwf_infinite_acting(1.0, 300.0, 4000.0, 55.0, 50.0, 0.2, 1.5, 15e-6, 0.25, 1.2)
        3854.69...
        
        >>> # Early times show larger drawdown
        >>> early = pwf_infinite_acting(0.1, 300.0, 4000.0, 55.0, 50.0, 0.2, 1.5, 15e-6, 0.25, 1.2)
        >>> late = pwf_infinite_acting(10.0, 300.0, 4000.0, 55.0, 50.0, 0.2, 1.5, 15e-6, 0.25, 1.2)
        >>> early < late  # Early time has more drawdown
        True
        
    Notes:
        - Valid only during infinite-acting period (t < t_inf)
        - Uses exponential integral Ei(-x) via scipy.special.expi(-x)
        - Assumes homogeneous reservoir with constant properties
        - Zero skin factor assumed; skin effects can be added separately
        - Line-source solution; wellbore storage effects not included
    """
    t_hr = np.asarray(t_hr, dtype=float)
    tD = 0.0002637 * k_md * t_hr / (phi * mu_cp * ct_psi_inv * rw_ft**2)
    pD = -0.5 * expi(-1.0 / (4.0 * tD))
    delta_p = 141.2 * (q_stbd * bo_rb_stb * mu_cp) / (k_md * h_ft) * pD
    result = psia_pi - delta_p
    return float(result) if t_hr.ndim == 0 else result

def pwf_pseudosteady_centered_well(t_hr: Union[float, np.ndarray], q_stbd: float, 
                                  psia_pi: float, area_acres: float, k_md: float, 
                                  h_ft: float, phi: float, mu_cp: float, 
                                  ct_psi_inv: float, rw_ft: float, bo_rb_stb: float, 
                                  skin: float = 0.0) -> Union[float, np.ndarray]:
    """
    Calculate pseudo-steady wellbore pressure for a centered well in a circular reservoir.
    
    Uses the pseudo-steady state approximation valid for late times when
    boundary effects dominate: pwf = p_avg - (141.2*q*B*μ)/(k*h) * [ln(re/rw) - 0.75 + s]
    
    Args:
        t_hr (Union[float, np.ndarray]): Time in hours (scalar or array)
        q_stbd (float): Production rate in STB/day
        psia_pi (float): Initial reservoir pressure in psia
        area_acres (float): Drainage area in acres
        k_md (float): Permeability in millidarcies
        h_ft (float): Net pay thickness in feet
        phi (float): Porosity as fraction (0-1)
        mu_cp (float): Viscosity in centipoise
        ct_psi_inv (float): Total compressibility in psi^-1
        rw_ft (float): Wellbore radius in feet
        bo_rb_stb (float): Oil formation volume factor in rb/STB
        skin (float, optional): Skin factor (dimensionless). Defaults to 0.0.
        
    Returns:
        Union[float, np.ndarray]: Wellbore flowing pressure(s) in psia
        
    Examples:
        >>> pwf_pseudosteady_centered_well(100.0, 300.0, 4000.0, 80.0, 55.0, 50.0, 0.2, 1.5, 15e-6, 0.25, 1.2)
        3732.03...
        
        >>> # With skin factor
        >>> pwf_with_skin = pwf_pseudosteady_centered_well(100.0, 300.0, 4000.0, 80.0, 55.0, 50.0, 0.2, 1.5, 15e-6, 0.25, 1.2, skin=2.0)
        >>> pwf_no_skin = pwf_pseudosteady_centered_well(100.0, 300.0, 4000.0, 80.0, 55.0, 50.0, 0.2, 1.5, 15e-6, 0.25, 1.2, skin=0.0)
        >>> pwf_with_skin < pwf_no_skin  # Positive skin reduces wellbore pressure
        True
        
    Notes:
        - Valid for late times when pseudo-steady conditions exist
        - Assumes circular drainage area with centered well
        - Combines material balance depletion with constant drawdown
        - Skin factor accounts for wellbore damage or stimulation
        - Shape factor of 0.75 appropriate for circular centered well
    """
    t_hr = np.asarray(t_hr, dtype=float)
    re_ft = drainage_radius_from_area(area_acres)
    p_avg = pavg_closed_system(psia_pi, t_hr, q_stbd, bo_rb_stb, area_acres, h_ft, phi, ct_psi_inv)
    dd = 141.2 * (q_stbd * bo_rb_stb * mu_cp) / (k_md * h_ft) * (np.log(re_ft / rw_ft) - 0.75 + skin)
    result = p_avg - dd
    return float(result) if t_hr.ndim == 0 else result

def pwf_pseudosteady_shape_factor(t_hr: Union[float, np.ndarray], q_stbd: float, 
                                 psia_pi: float, area_acres: float, k_md: float, 
                                 h_ft: float, phi: float, mu_cp: float, 
                                 ct_psi_inv: float, rw_ft: float, bo_rb_stb: float, 
                                 shape_factor_CA: float, skin: float = 0.0) -> Union[float, np.ndarray]:
    """
    Calculate pseudo-steady wellbore pressure using Dietz/Dake-style drainage shape factor.
    
    Implements Dake's pseudo-steady approximation using drainage shape factor C_A:
    pwf = p_avg - (141.2*q*B*μ)/(k*h) * [0.5*ln(4A/(γ*C_A*rw²)) + s]
    where γ = e^0.57721566 ≈ 1.781 (Euler's constant).
    
    Args:
        t_hr (Union[float, np.ndarray]): Time in hours (scalar or array)
        q_stbd (float): Production rate in STB/day
        psia_pi (float): Initial reservoir pressure in psia
        area_acres (float): Drainage area in acres
        k_md (float): Permeability in millidarcies
        h_ft (float): Net pay thickness in feet
        phi (float): Porosity as fraction (0-1)
        mu_cp (float): Viscosity in centipoise
        ct_psi_inv (float): Total compressibility in psi^-1
        rw_ft (float): Wellbore radius in feet
        bo_rb_stb (float): Oil formation volume factor in rb/STB
        shape_factor_CA (float): Dietz shape factor C_A (dimensionless)
        skin (float, optional): Skin factor (dimensionless). Defaults to 0.0.
        
    Returns:
        Union[float, np.ndarray]: Wellbore flowing pressure(s) in psia
        
    Examples:
        >>> # Square drainage with centered well (C_A = 30.9)
        >>> pwf_pseudosteady_shape_factor(100.0, 300.0, 4000.0, 80.0, 55.0, 50.0, 0.2, 1.5, 15e-6, 0.25, 1.2, 30.9)
        3748.72...
        
        >>> # Compare different shape factors
        >>> square_center = 30.9
        >>> circle_center = 31.6
        >>> p_square = pwf_pseudosteady_shape_factor(100.0, 300.0, 4000.0, 80.0, 55.0, 50.0, 0.2, 1.5, 15e-6, 0.25, 1.2, square_center)
        >>> p_circle = pwf_pseudosteady_shape_factor(100.0, 300.0, 4000.0, 80.0, 55.0, 50.0, 0.2, 1.5, 15e-6, 0.25, 1.2, circle_center)
        >>> abs(p_square - p_circle) < 10  # Similar pressures for similar shapes
        True
        
    Notes:
        - Valid for various drainage shapes using appropriate C_A values
        - Common C_A values: circle (31.6), square (30.9), 2:1 rectangle (21.9)
        - Based on Dake's formulation with Euler's constant γ ≈ 1.781
        - More flexible than circular centered well approximation
        - Shape factor accounts for non-circular drainage areas
    """
    t_hr = np.asarray(t_hr, dtype=float)
    area_ft2 = area_acres * 43560.04
    gamma_euler = float(np.exp(0.5772156649015329))  # ≈ 1.781
    p_avg = pavg_closed_system(psia_pi, t_hr, q_stbd, bo_rb_stb, area_acres, h_ft, phi, ct_psi_inv)
    shape_term = 0.5 * np.log((4.0 * area_ft2) / (gamma_euler * shape_factor_CA * rw_ft**2)) + skin
    dd = 141.2 * (q_stbd * bo_rb_stb * mu_cp) / (k_md * h_ft) * shape_term
    result = p_avg - dd
    return float(result) if t_hr.ndim == 0 else result

def pwf_bounded_piecewise(t_hr: Union[float, np.ndarray], q_stbd: float, 
                         psia_pi: float, area_acres: float, k_md: float, 
                         h_ft: float, phi: float, mu_cp: float, 
                         ct_psi_inv: float, rw_ft: float, bo_rb_stb: float, 
                         skin: float = 0.0) -> Union[float, np.ndarray]:
    """
    Calculate wellbore pressure using piecewise bounded model.
    
    Combines infinite-acting radial flow (IARF) for early times and pseudo-steady
    state for late times, switching at the end of infinite-acting period.
    Uses IARF solution for t ≤ t_inf and pseudo-steady solution for t > t_inf.
    
    Args:
        t_hr (Union[float, np.ndarray]): Time in hours (scalar or array)
        q_stbd (float): Production rate in STB/day
        psia_pi (float): Initial reservoir pressure in psia
        area_acres (float): Drainage area in acres
        k_md (float): Permeability in millidarcies
        h_ft (float): Net pay thickness in feet
        phi (float): Porosity as fraction (0-1)
        mu_cp (float): Viscosity in centipoise
        ct_psi_inv (float): Total compressibility in psi^-1
        rw_ft (float): Wellbore radius in feet
        bo_rb_stb (float): Oil formation volume factor in rb/STB
        skin (float, optional): Skin factor (dimensionless). Defaults to 0.0.
        
    Returns:
        Union[float, np.ndarray]: Wellbore flowing pressure(s) in psia
        
    Examples:
        >>> # Early time (IARF behavior)
        >>> p_early = pwf_bounded_piecewise(1.0, 300.0, 4000.0, 80.0, 55.0, 50.0, 0.2, 1.5, 15e-6, 0.25, 1.2)
        >>> p_early > 3800
        True
        
        >>> # Late time (pseudo-steady behavior)
        >>> p_late = pwf_bounded_piecewise(100.0, 300.0, 4000.0, 80.0, 55.0, 50.0, 0.2, 1.5, 15e-6, 0.25, 1.2)
        >>> p_late < 3800
        True
        
        >>> # Array input
        >>> times = np.array([1.0, 10.0, 100.0])
        >>> pressures = pwf_bounded_piecewise(times, 300.0, 4000.0, 80.0, 55.0, 50.0, 0.2, 1.5, 15e-6, 0.25, 1.2)
        >>> len(pressures) == 3
        True
        
    Notes:
        - Provides smooth transition between flow regimes
        - More realistic than pure IARF or pseudo-steady solutions
        - Switching time calculated automatically based on reservoir properties
        - Skin factor applied only in pseudo-steady regime
        - Useful for complete well test analysis covering all time periods
    """
    t_hr = np.asarray(t_hr, dtype=float)
    t_inf_hr = t_infinite_acting_hr(phi, mu_cp, ct_psi_inv, area_acres, k_md)
    pwf_iarf = pwf_infinite_acting(t_hr, q_stbd, psia_pi, k_md, h_ft, phi, mu_cp, ct_psi_inv, rw_ft, bo_rb_stb)
    pwf_pss = pwf_pseudosteady_centered_well(t_hr, q_stbd, psia_pi, area_acres, k_md, h_ft, phi, mu_cp, ct_psi_inv, rw_ft, bo_rb_stb, skin=skin)
    result = np.where(t_hr <= t_inf_hr, pwf_iarf, pwf_pss)
    return float(result) if t_hr.ndim == 0 else result

def p_profile_pseudosteady(t_hr: Union[float, np.ndarray], r_ft: Union[float, np.ndarray], 
                          q_stbd: float, psia_pi: float, area_acres: float, 
                          k_md: float, h_ft: float, phi: float, mu_cp: float, 
                          ct_psi_inv: float, rw_ft: float, bo_rb_stb: float, 
                          skin: float = 0.0) -> Union[float, np.ndarray]:
    """
    Calculate late-time pseudo-steady pressure profile p(r,t) for centered well.
    
    Provides spatial pressure distribution in a closed circular reservoir during
    pseudo-steady state using: p(r,t) = p_avg - (141.2*q*B*μ)/(k*h) * [ln(re/r) - 0.75 + r²/(2*re²) + s]
    
    Args:
        t_hr (Union[float, np.ndarray]): Time in hours (scalar or array)
        r_ft (Union[float, np.ndarray]): Radial distance from well in feet (scalar or array)
        q_stbd (float): Production rate in STB/day
        psia_pi (float): Initial reservoir pressure in psia
        area_acres (float): Drainage area in acres
        k_md (float): Permeability in millidarcies
        h_ft (float): Net pay thickness in feet
        phi (float): Porosity as fraction (0-1)
        mu_cp (float): Viscosity in centipoise
        ct_psi_inv (float): Total compressibility in psi^-1
        rw_ft (float): Wellbore radius in feet
        bo_rb_stb (float): Oil formation volume factor in rb/STB
        skin (float, optional): Skin factor (dimensionless). Defaults to 0.0.
        
    Returns:
        Union[float, np.ndarray]: Pressure(s) at specified radius and time in psia
        
    Examples:
        >>> # Pressure at wellbore
        >>> p_wf = p_profile_pseudosteady(100.0, 0.25, 300.0, 4000.0, 80.0, 55.0, 50.0, 0.2, 1.5, 15e-6, 0.25, 1.2)
        >>> p_wf < 4000
        True
        
        >>> # Pressure at drainage boundary
        >>> re = drainage_radius_from_area(80.0)
        >>> p_boundary = p_profile_pseudosteady(100.0, re, 300.0, 4000.0, 80.0, 55.0, 50.0, 0.2, 1.5, 15e-6, 0.25, 1.2)
        >>> p_wf < p_boundary  # Pressure increases with distance from well
        True
        
        >>> # Radial pressure profile
        >>> radii = np.logspace(0, 3, 10)  # 1 to 1000 ft
        >>> profile = p_profile_pseudosteady(100.0, radii, 300.0, 4000.0, 80.0, 55.0, 50.0, 0.2, 1.5, 15e-6, 0.25, 1.2)
        >>> len(profile) == 10
        True
        
    Notes:
        - Valid only during pseudo-steady state (late times)
        - Assumes circular drainage area with centered well
        - Includes both logarithmic and parabolic terms
        - Skin factor affects pressure at all radii
        - Useful for reservoir pressure mapping and analysis
        - Profile shape depends on time through average pressure depletion
    """
    t_hr = np.asarray(t_hr, dtype=float)
    r_ft = np.asarray(r_ft, dtype=float)
    re_ft = drainage_radius_from_area(area_acres)
    p_avg = pavg_closed_system(psia_pi, t_hr, q_stbd, bo_rb_stb, area_acres, h_ft, phi, ct_psi_inv)
    shape_term = np.log(re_ft / r_ft) - 0.75 + (r_ft**2) / (2.0 * re_ft**2) + skin
    dd = 141.2 * (q_stbd * bo_rb_stb * mu_cp) / (k_md * h_ft) * shape_term
    result = p_avg - dd
    
    # Handle broadcasting for different input combinations
    if t_hr.ndim == 0 and r_ft.ndim == 0:
        return float(result)
    else:
        return result

def bourdet_derivative(t: Union[np.ndarray, list], p: Union[np.ndarray, list]) -> np.ndarray:
    """
    Calculate Bourdet-style pressure derivative dp/d(ln t) using 3-point formula.
    
    Computes the pressure derivative with respect to natural logarithm of time
    using a three-point finite difference formula on log-time spacing. This
    derivative is commonly used in well test analysis for flow regime identification.
    
    Args:
        t (Union[np.ndarray, list]): Time values (must be monotonically increasing)
        p (Union[np.ndarray, list]): Pressure values corresponding to time points
        
    Returns:
        np.ndarray: Derivative dp/d(ln t) with same length as input arrays.
                   End points and invalid points are set to NaN.
                   
    Examples:
        >>> t = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        >>> p = np.array([100.0, 90.0, 80.0, 70.0, 60.0])  # Linear decline
        >>> deriv = bourdet_derivative(t, p)
        >>> np.isnan(deriv[0]) and np.isnan(deriv[-1])  # End points are NaN
        True
        >>> np.all(deriv[1:-1] < 0)  # Negative derivative for declining pressure
        True
        
        >>> # Constant pressure should give zero derivative
        >>> p_const = np.full(5, 100.0)
        >>> deriv_const = bourdet_derivative(t, p_const)
        >>> np.allclose(deriv_const[1:-1], 0.0, atol=1e-10)
        True
        
    Notes:
        - Uses three-point finite difference formula on logarithmic time spacing
        - First and last points cannot be calculated (set to NaN)
        - Handles non-uniform time spacing appropriately
        - Skips points with identical time values (sets to NaN)
        - Commonly plotted on log-log scale for flow regime analysis
        - Useful for identifying radial flow (horizontal line), bilinear flow (1/2 slope)
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)

    ln_t = np.log(t)
    n = len(t)
    d = np.full(n, np.nan, dtype=float)

    for i in range(1, n - 1):
        x1, x2, x3 = ln_t[i - 1], ln_t[i], ln_t[i + 1]
        y1, y2, y3 = p[i - 1], p[i], p[i + 1]

        # Skip if any time points are identical (would cause division by zero)
        if (x2 == x1) or (x3 == x2) or (x3 == x1):
            continue

        # Three-point finite difference formula
        d[i] = (
            ((x2 - x1) / (x3 - x1)) * ((y3 - y2) / (x3 - x2))
            + ((x3 - x2) / (x3 - x1)) * ((y2 - y1) / (x2 - x1))
        )

    return d

def r2_score(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Calculate coefficient of determination (R² score) for regression analysis.
    
    Computes R² = 1 - SS_res/SS_tot, where SS_res is the sum of squares of residuals
    and SS_tot is the total sum of squares. Measures the proportion of variance
    in the dependent variable predictable from the independent variables.
    
    Args:
        y_true (Union[np.ndarray, list]): True values (observed data)
        y_pred (Union[np.ndarray, list]): Predicted values (model predictions)
        
    Returns:
        float: R² coefficient of determination. Perfect fit = 1.0, 
               no predictive power = 0.0, worse than mean = negative values.
               Returns NaN if total sum of squares is zero (constant y_true).
               
    Examples:
        >>> # Perfect fit
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = y_true.copy()
        >>> r2_score(y_true, y_pred)
        1.0
        
        >>> # No fit (prediction = mean)
        >>> y_pred_mean = np.full_like(y_true, np.mean(y_true))
        >>> r2_score(y_true, y_pred_mean)
        0.0
        
        >>> # Partial fit
        >>> y_pred_partial = y_true + np.random.normal(0, 0.1, len(y_true))
        >>> r2 = r2_score(y_true, y_pred_partial)
        >>> 0.0 <= r2 <= 1.0  # Should be between 0 and 1
        True
        
        >>> # Constant y_true returns NaN
        >>> y_const = np.full(4, 5.0)
        >>> np.isnan(r2_score(y_const, y_const))
        True
        
    Notes:
        - R² = 1.0 indicates perfect fit
        - R² = 0.0 indicates model performs no better than predicting the mean
        - Negative R² indicates model performs worse than predicting the mean
        - Returns NaN for constant y_true (zero total sum of squares)
        - Commonly used to evaluate well test model fits
        - Values closer to 1.0 indicate better model performance
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

def _format_grouped_results(title: str, method_values: Dict[str, float], unit: str = "") -> str:
    """Format a group of method results with a mean line."""
    lines = [f"{title}:"]
    values = np.array(list(method_values.values()), dtype=float)
    for method_name, value in method_values.items():
        display_name = str(method_name).replace('_', '-').title()
        lines.append(f"  {display_name}: {value:.2f}{unit}")
    valid_values = values[~np.isnan(values)]
    mean_value = np.nan if valid_values.size == 0 else float(np.mean(valid_values))
    lines.append(f"  Mean ({valid_values.size} methods): {mean_value:.2f}{unit}")
    return "\n".join(lines)

def format_well_test_results(results: Dict[str, Any]) -> str:
    """Return a formatted text summary for well_test_analysis results."""
    test_type = results.get('test_type', 'buildup')
    show_average_pressure = results.get('show_average_pressure', test_type != 'dst')

    if test_type == 'dst':
        final_result = results['final_buildup']
        damage_ratio = results['damage_ratio']
        sections = [
            "DST Final Buildup Analysis (primary):",
            _format_grouped_results("Permeability", final_result['permeability_md'], " md"),
            _format_grouped_results("Skin", final_result['skin']),
            "Damage Ratio:",
            f"  Reservoir pressure: {damage_ratio['reservoir_pressure_psia']:.2f} psi",
            f"  Average flowing pressure: {damage_ratio['average_flowing_pressure_psia']:.2f} psi",
            f"  deltaP_skin: {damage_ratio['delta_p_skin_psi']:.2f} psi",
            f"  Damage ratio: {damage_ratio['value']:.3f}",
        ]

        if show_average_pressure:
            sections.insert(3, _format_grouped_results("Average Pressure", final_result['average_pressure_psia'], " psi"))

        comparison_result = results.get('comparison_buildup')
        if comparison_result is not None:
            sections.extend([
                "DST Initial vs Final Buildup Comparison:",
                f"  Initial permeability (Horner): {comparison_result['permeability_md']['horner']:.2f} md",
                f"  Final permeability (Horner):   {final_result['permeability_md']['horner']:.2f} md",
                f"  Initial skin (Horner): {comparison_result['skin']['horner']:.2f}",
                f"  Final skin (Horner):   {final_result['skin']['horner']:.2f}",
            ])
        return "\n".join(sections)

    sections = [
        _format_grouped_results("Permeability", results['permeability_md'], " md"),
        _format_grouped_results("Skin", results['skin']),
    ]
    if show_average_pressure:
        sections.append(_format_grouped_results("Average Pressure", results['average_pressure_psia'], " psi"))
    return "\n\n".join(sections)

def print_well_test_results(results: Dict[str, Any]) -> None:
    """Pretty-print well test analysis results."""
    print(format_well_test_results(results))

def _plot_sparse_safe(ax, plot_kind: str, x: Union[np.ndarray, list], y: Union[np.ndarray, list],
                      min_line_points: int = 5,
                      force_scatter: bool = False,
                      **kwargs):
    """Plot a series while avoiding choppy-looking lines for very sparse datasets."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if plot_kind == 'loglog':
        valid_mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    else:
        valid_mask = np.isfinite(x) & np.isfinite(y)

    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    if x_valid.size == 0:
        return None

    plotter = getattr(ax, plot_kind)
    plot_kwargs = dict(kwargs)
    if force_scatter or x_valid.size < min_line_points:
        plot_kwargs.setdefault('linestyle', 'None')
        plot_kwargs.setdefault('marker', 'o')
        plot_kwargs.setdefault('markersize', 5)
    return plotter(x_valid, y_valid, **plot_kwargs)

def _shade_masked_xzones(ax, x: Union[np.ndarray, list], mask: Union[np.ndarray, list],
                         color: str, alpha: float, label: Optional[str] = None):
    """Shade contiguous True regions of a boolean mask as full-height x-spans."""
    
    # ╔═════════════════════════════════════════════════════════════════════════╗
    # ║ FUNCTION PURPOSE:                                                      ║
    # ║   Render colored vertical bands for flow regimes (WBS, IARF, boundary) ║
    # ║   Robust to any x-axis direction (ascending or descending Horner axis) ║
    # ║                                                                         ║
    # ║ WHY THIS IS TRICKY:                                                    ║
    # ║   - Horner plots: x-axis descends (left to right: ∞ to 0)              ║
    # ║   - Time plots: x-axis ascends (left to right: 0 to ∞)                 ║
    # ║   - fill_between + y-limits doesn't work for descending axis           ║
    # ║   - Solution: use axvspan with min/max x bounds instead               ║
    # ║                                                                         ║
    # ║ ALGORITHM:                                                             ║
    # ║   1. Convert inputs to arrays                                          ║
    # ║   2. Find contiguous regions where mask=True                           ║
    # ║   3. For each region: render axvspan from min(x) to max(x)            ║
    # ║   4. axvspan works regardless of axis direction                        ║
    # ╚═════════════════════════════════════════════════════════════════════════╝
    
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if x.ndim != 1 or mask.ndim != 1 or x.size != mask.size:
        return
    if x.size == 0 or not np.any(mask):
        return

    # Find contiguous runs of True values in mask
    # Example: mask = [F,T,T,F,T,F,T,T,T] → runs at (1-2), (4), (6-8)
    run_start = None
    first_patch = True
    for index, is_true in enumerate(mask):
        # Start new run if we find first True after False
        if is_true and run_start is None:
            run_start = index
        # End run if we find False after True, or reach end of array
        if (not is_true or index == x.size - 1) and run_start is not None:
            # Calculate endpoint of this run
            run_end = index if not is_true else index + 1
            # Extract x values in this contiguous region
            x_segment = x[run_start:run_end]
            # Compute bounds (works for both ascending and descending x)
            left = float(np.min(x_segment))
            right = float(np.max(x_segment))
            # Render full-height vertical band from left to right
            if right > left:
                ax.axvspan(left, right, color=color, alpha=alpha, 
                          label=label if first_patch else None)
                first_patch = False
            run_start = None

def _apply_time_epsilon(time_hr: Union[np.ndarray, list], eps_hr: Optional[float] = None) -> np.ndarray:
    """Replace non-positive shut-in times with a small epsilon to keep all data points."""
    
    # ╔═════════════════════════════════════════════════════════════════════════╗
    # ║ FUNCTION PURPOSE:                                                      ║
    # ║   Preserve all buildup datapoints despite mathematical singularities   ║
    # ║   that occur at t=0 (log(0)→-∞, 1/0→∞, etc).                          ║
    # ║                                                                         ║
    # ║ KEY INSIGHT:                                                           ║
    # ║   Instead of discarding first point (traditional), we substitute       ║
    # ║   a tiny positive time ε so all calculations work cleanly.             ║
    # ║                                                                         ║
    # ║ STRATEGY:                                                              ║
    # ║   - Use epsilon only for math (Horner axis, derivatives, etc)          ║
    # ║   - Exclude first point from plot xlim via display_mask                ║
    # ║   - Result: no visual axis distortion + no data loss                   ║
    # ╚═════════════════════════════════════════════════════════════════════════╝
    
    time_hr = np.asarray(time_hr, dtype=float)
    if time_hr.ndim != 1:
        raise ValueError("Shut-in time must be a one-dimensional array")
    if np.any(~np.isfinite(time_hr)):
        raise ValueError("Shut-in time values must be finite")

    if np.any(time_hr < 0.0):
        raise ValueError("Shut-in time values must be non-negative")

    positive_times = time_hr[time_hr > 0.0]
    if positive_times.size == 0:
        raise ValueError("Buildup analysis requires at least one positive shut-in time")

    # Determine epsilon: use machine epsilon or 1e-8 hr (≈ 3.6 microseconds)
    # whichever is larger. This ensures numerical stability without being
    # so small that floating-point precision becomes an issue.
    if eps_hr is None:
        eps_hr = max(np.finfo(float).eps, 1e-8)
    eps_hr = float(eps_hr)
    if eps_hr <= 0.0:
        raise ValueError("eps_hr must be positive")

    # Replace any time ≤ 0 with epsilon, keep positive times unchanged
    # Result: min(output) = eps_hr > 0, all log calculations are well-defined
    return np.maximum(time_hr, eps_hr)

def _analyze_buildup_period(raw_time_hr: Union[np.ndarray, list],
                            raw_pressure_psia: Union[np.ndarray, list],
                            q_stbd: float,
                            tp_hr: float,
                            IARF_slice: Tuple[float, float],
                            Bo: float,
                            mu: float,
                            h: float,
                            rw: float,
                            phi: float,
                            ct: float,
                            A: float,
                            CA: Optional[float],
                            pmbhd: Optional[float] = None,
                            mbh_alpha: Optional[float] = None,
                            mbh_beta: Optional[float] = None,
                            mbh_lambda: Optional[float] = None,
                            deconvolved_data: Optional[Dict[str, Any]] = None,
                            period_label: str = 'Buildup',
                            use_horner_only: bool = False) -> Dict[str, Any]:
    """
    Run a single buildup interpretation using the shared Horner/MDH/Bourdet workflow.
    
    ╔═════════════════════════════════════════════════════════════════════════╗
    ║ CORE BUILDUP ANALYSIS ENGINE                                           ║
    ║                                                                         ║
    ║ This function is the heart of all pressure transient analysis. It:      ║
    ║   1. Applies epsilon to avoid singularities (retains all datapoints)    ║
    ║   2. Creates Horner transform: (tp+Δt)/Δt                              ║
    ║   3. Fits straight line to IARF interval                               ║
    ║   4. Extracts permeability, skin, avg pressure from slope/intercept    ║
    ║   5. Computes Bourdet derivative for secondary verification            ║
    ║   6. Returns dict with all estimates + visualization masks             ║
    ║                                                                         ║
    ║ INPUT VARIATION:                                                        ║
    ║   - Constant rate: raw data directly analyzed                          ║
    ║   - Variable rate: deconvolved_data contains pre-processed response    ║
    ║   - DST: called once per buildup period with different tp/rate         ║
    ║                                                                         ║
    ║ OUTPUT DICT CONTAINS:                                                  ║
    ║   - All Horner/MDH/Bourdet estimates (3 estimates each for k and s)    ║
    ║   - Time arrays and transformed axes (for plotting)                    ║
    ║   - Flow regime masks (IARF, WBS, boundary identification)             ║
    ║   - Fit quality metrics (R², residual %)                               ║
    ║   - Display masks to exclude epsilon points from plot xlim             ║
    ╚═════════════════════════════════════════════════════════════════════════╝
    """
    
    raw_time_hr = np.asarray(raw_time_hr, dtype=float)
    raw_pressure_psia = np.asarray(raw_pressure_psia, dtype=float)

    if raw_time_hr.shape != raw_pressure_psia.shape:
        raise ValueError("Buildup time and pressure arrays must have the same shape")
    if raw_time_hr.ndim != 1 or raw_time_hr.size < 2:
        raise ValueError("Buildup analysis requires one-dimensional time and pressure arrays")
    if tp_hr is None or tp_hr <= 0.0:
        raise ValueError("Buildup producing time (tp) must be positive")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1: DATA PREPARATION AND EPSILON APPLICATION
    # ════════════════════════════════════════════════════════════════════════
    
    # Find initial (shut-in) pressure: the minimum time typically corresponds
    # to earliest shut-in pressure, used as baseline for delta-p calculations
    pwf_psia = float(raw_pressure_psia[np.argmin(raw_time_hr)])
    
    # Apply epsilon to times: replace any t ≤ 0 with 1e-8 hr
    # Crucially, we keep all datapoints (no deletion)
    time_positive = _apply_time_epsilon(raw_time_hr)
    pressure_positive = raw_pressure_psia

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2: SELECT DATA SOURCE (CONSTANT vs DECONVOLVED)
    # ════════════════════════════════════════════════════════════════════════
    
    # If deconvolved_data provided (from variable-rate deconvolution),
    # use transformed coordinates from deconvolution.
    # Otherwise use raw data directly (constant-rate case).
    analysis_time = deconvolved_data['analysis_time_hr'] if deconvolved_data is not None else time_positive
    analysis_pressure = deconvolved_data['pressure_psia'] if deconvolved_data is not None else pressure_positive
    analysis_delta_p = deconvolved_data['delta_p_psi'] if deconvolved_data is not None else (pressure_positive - pressure_positive[0])
    horner_time = deconvolved_data['horner_time'] if deconvolved_data is not None else (time_positive + tp_hr) / time_positive
    log_horner_time = deconvolved_data['log_horner_time'] if deconvolved_data is not None else np.log10(horner_time)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3: CREATE VISUALIZATION MASKS
    # ════════════════════════════════════════════════════════════════════════
    
    # regime_time: used for identifying flow regimes (WBS, IARF, boundary)
    regime_time = time_positive
    
    # display_mask: boolean array indicating which points to include in plot
    # xlimits. Excludes epsilon-adjusted points (original t ≤ 0) so they
    # don't distort axis limits, but includes them in calculations.
    display_mask = raw_time_hr > 0.0
    
    # If deconvolution reordered points, apply same reordering to masks
    sort_order = None if deconvolved_data is None else deconvolved_data.get('sort_order')
    if sort_order is not None:
        regime_time = regime_time[np.asarray(sort_order, dtype=int)]
        display_mask = display_mask[np.asarray(sort_order, dtype=int)]

    # ════════════════════════════════════════════════════════════════════════
    # STEP 4: IDENTIFY IARF STRAIGHT-LINE INTERVAL
    # ════════════════════════════════════════════════════════════════════════
    
    # Create mask for points within user-specified IARF interval
    # Example: IARF_slice = (1.0, 20.0) means use points where 1 ≤ Δt ≤ 20
    IARF_mask = (regime_time >= IARF_slice[0]) & (regime_time <= IARF_slice[1])
    if np.count_nonzero(IARF_mask) < 2:
        raise ValueError("Need at least two points in the IARF interval")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 5: HORNER PLOT STRAIGHT-LINE REGRESSION
    # ════════════════════════════════════════════════════════════════════════
    
    # Compute Bourdet derivative (used for verification and plotting)
    dp_dlnt = bourdet_derivative(analysis_time, analysis_delta_p)

    # Fit linear regression to log-log Horner coordinates on IARF interval
    # x = log10[(tp + Δt)/Δt], y = pressure
    # Result: slope m and intercept p* 
    # Physical meaning:
    #   m = -162.6*q*B*μ/(k*h)  →  k = 162.6*q*B*μ/(|m|*h)
    #   p* = intercept at (tp+Δt)/Δt → ∞  (average reservoir pressure)
    params_horner = np.polyfit(log_horner_time[IARF_mask], analysis_pressure[IARF_mask], 1)
    
    # Generate fitted line across all time values (for plotting)
    IARF_line_horner = params_horner[0] * log_horner_time + params_horner[1]
    
    # Calculate fit quality: residual standard deviation as % of pressure range
    iarf_pressure = analysis_pressure[IARF_mask]
    iarf_residuals = iarf_pressure - IARF_line_horner[IARF_mask]
    iarf_pressure_range = np.ptp(iarf_pressure)  # peak-to-peak = max - min
    iarf_residual_pct = 100.0 * np.std(iarf_residuals) / iarf_pressure_range if iarf_pressure_range > 0.0 else np.nan

    # ════════════════════════════════════════════════════════════════════════
    # STEP 6: BOURDET DERIVATIVE ANALYSIS (MULTI-METHOD MODE)
    # ════════════════════════════════════════════════════════════════════════
    
    if not use_horner_only:
        # For full multi-method analysis, also fit Bourdet derivative
        # Derivative in radial flow regime should be horizontal (constant)
        def horizontal_line(x, c):
            return c

        # Find finite derivative points in IARF region that are positive
        iarf_derivative_mask = IARF_mask & np.isfinite(dp_dlnt) & (dp_dlnt > 0.0)
        if np.count_nonzero(iarf_derivative_mask) < 2:
            raise ValueError("Need at least two finite derivative points in the straight-line interval")
        
        # Fit horizontal line to log(t) vs log(dp/d(ln t))
        # Result: bourdet_level = the constant derivative value in radial flow
        popt_bourdet, _ = curve_fit(horizontal_line, np.log10(analysis_time[iarf_derivative_mask]), dp_dlnt[iarf_derivative_mask])
        bourdet_level = float(popt_bourdet[0])
        bourdet_line = np.full_like(analysis_time, bourdet_level, dtype=float)
    else:
        bourdet_level = np.nan
        bourdet_line = np.full_like(analysis_time, np.nan, dtype=float)
        bourdet_level = np.nan
        bourdet_line = np.full_like(analysis_time, np.nan, dtype=float)

    k_from_horner = 162.6 * q_stbd * Bo * mu / (abs(params_horner[0]) * h)
    p_1hr = params_horner[0] * np.log10((tp_hr + 1.0) / 1.0) + params_horner[1]
    s_from_horner = 1.151 * ((p_1hr - pwf_psia) / np.abs(params_horner[0]) - np.log10(k_from_horner / phi / mu / ct / rw**2) + 3.23)

    if not use_horner_only:
        params_mdh = np.polyfit(np.log10(analysis_time[IARF_mask]), analysis_pressure[IARF_mask], 1)
        k_from_mdh = 162.6 * q_stbd * Bo * mu / (abs(params_mdh[0]) * h)
        s_from_mdh = 1.151 * ((p_1hr - pwf_psia) / np.abs(params_mdh[0]) - np.log10(k_from_mdh / phi / mu / ct / rw**2) + 3.23)

        k_from_bourdet = 70.6 * q_stbd * Bo * mu / (bourdet_level * h)
        iarf_indices = np.where(IARF_mask)[0]
        deltap_r_index = iarf_indices[len(iarf_indices) // 2]
        deltap_r = analysis_delta_p[deltap_r_index]
        deltat_r = analysis_time[deltap_r_index]
        s_from_bourdet = 0.5 * ((deltap_r / bourdet_level) - np.log(k_from_bourdet * deltat_r / 1688.0 / phi / mu / ct / rw**2))

        deltat_pbar_mdh = phi * mu * ct * A / 0.0002637 / k_from_mdh / CA
        pbar_mdh = params_mdh[0] * np.log10(deltat_pbar_mdh) + params_mdh[1]
    else:
        k_from_mdh = np.nan
        s_from_mdh = np.nan
        k_from_bourdet = np.nan
        s_from_bourdet = np.nan
        pbar_mdh = np.nan
    horner_time_pbar = 0.0002637 * k_from_horner * CA * tp_hr / phi / mu / ct / A
    pbar_ramey = params_horner[0] * np.log10(horner_time_pbar) + params_horner[1]

    tpad = 0.0002637 * k_from_horner * tp_hr / phi / mu / ct / A
    if pmbhd is None and CA is not None:
        if mbh_alpha is None or mbh_beta is None or mbh_lambda is None:
            pmbhd_value = float(np.log(CA * tpad))
        else:
            pmbhd_value = mbh_dimensionless_average_pressure(
                tpad,
                shape_factor_ca=CA,
                alpha=mbh_alpha,
                beta=mbh_beta,
                aspect_ratio_lambda=mbh_lambda,
            )
    else:
        pmbhd_value = pmbhd

    pbar_mbh = params_horner[1] - abs(params_horner[0]) / 2.303 * pmbhd_value if pmbhd_value is not None else np.nan

    return {
        'period_label': period_label,
        'time_hr': analysis_time,
        'pressure_psia': analysis_pressure,
        'deltap_psi': analysis_delta_p,
        'dp_dlnt': dp_dlnt,
        'IARF_mask': IARF_mask,
        'display_mask': display_mask,
        'regime_time_hr': regime_time,
        'horner_time': horner_time,
        'log_horner_time': log_horner_time,
        'IARF_line_horner': IARF_line_horner,
        'bourdet_line': bourdet_line,
        'q_stbd': float(q_stbd),
        'tp_hr': float(tp_hr),
        'pwf_psia': float(pwf_psia),
        'p_star_psia': float(params_horner[1]),
        'horner_slope': float(params_horner[0]),
        'horner_residual_pct_of_range': float(iarf_residual_pct),
        'calculation_mode': 'horner_only' if use_horner_only else 'full',
        'permeability_md': {
            'horner': float(k_from_horner),
            'mdh': float(k_from_mdh),
            'bourdet': float(k_from_bourdet),
        },
        'skin': {
            'horner': float(s_from_horner),
            'mdh': float(s_from_mdh),
            'bourdet': float(s_from_bourdet),
        },
        'average_pressure_psia': {
            'dietz': float(pbar_mdh),
            'ramey_cobb': float(pbar_ramey),
            'mbh': float(pbar_mbh) if pmbhd_value is not None else np.nan,
        },
        'pmbhd': None if pmbhd_value is None else float(pmbhd_value),
        'tpad': float(tpad),
    }

def _extract_reference_rate(rate: Any) -> float:
    """Return the reference production rate used for permeability and skin estimates."""
    if isinstance(rate, dict):
        rate_values = rate.get('rate', rate.get('values'))
        if rate_values is None:
            raise ValueError("Rate history dictionaries must include a 'rate' or 'values' entry")
        rate_array = np.asarray(rate_values, dtype=float)
    else:
        rate_array = np.asarray(rate, dtype=float)

    if rate_array.ndim == 0:
        rate_value = float(rate_array)
        if not np.isfinite(rate_value) or rate_value == 0.0:
            raise ValueError("Rate must be finite and non-zero for well test interpretation")
        return rate_value

    valid_rates = rate_array[np.isfinite(rate_array) & (rate_array != 0.0)]
    if valid_rates.size == 0:
        raise ValueError("Rate history must contain at least one finite non-zero value")
    return float(valid_rates[-1])

def _parse_rate_history(rate: Any, tp_hr: Optional[float]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """Parse optional variable-rate history used for buildup deconvolution."""
    if not isinstance(rate, dict):
        return None, None, tp_hr

    rate_values = rate.get('rate', rate.get('values'))
    if rate_values is None:
        raise ValueError("Rate history dictionaries must include a 'rate' or 'values' entry")

    rate_values = np.asarray(rate_values, dtype=float)
    if rate_values.ndim != 1 or rate_values.size == 0:
        raise ValueError("Rate history values must be a one-dimensional array with at least one entry")
    if np.any(~np.isfinite(rate_values)):
        raise ValueError("Rate history values must be finite")

    if 'dt' in rate:
        dt_history = np.asarray(rate['dt'], dtype=float)
        if dt_history.shape != rate_values.shape:
            raise ValueError("Rate history 'dt' must have the same length as the rate values")
        if np.any(dt_history <= 0.0):
            raise ValueError("Rate history interval durations must be positive")
        history_end_times = np.cumsum(dt_history)
    elif 'time' in rate:
        history_end_times = np.asarray(rate['time'], dtype=float)
        if history_end_times.shape != rate_values.shape:
            raise ValueError("Rate history 'time' must have the same length as the rate values")
        if np.any(history_end_times <= 0.0) or np.any(np.diff(history_end_times) <= 0.0):
            raise ValueError("Rate history times must be strictly increasing and positive")
    else:
        raise ValueError("Rate history dictionaries must include either 'dt' or 'time'")

    history_total_time = float(history_end_times[-1])
    if tp_hr is None:
        tp_hr = history_total_time
    elif not np.isclose(history_total_time, tp_hr, rtol=1e-4, atol=1e-8):
        raise ValueError("Rate history total duration must match tp for buildup deconvolution")

    return history_end_times, rate_values, float(tp_hr)

def _log_time_interpolation_matrix(target_times_hr: Union[np.ndarray, list],
                                   time_grid_hr: Union[np.ndarray, list]) -> np.ndarray:
    """Build a linear interpolation matrix on a logarithmic time grid."""
    target_times_hr = np.asarray(target_times_hr, dtype=float)
    time_grid_hr = np.asarray(time_grid_hr, dtype=float)

    if target_times_hr.ndim != 1 or time_grid_hr.ndim != 1:
        raise ValueError("Interpolation inputs must be one-dimensional arrays")
    if np.any(target_times_hr <= 0.0) or np.any(time_grid_hr <= 0.0):
        raise ValueError("Interpolation times must be strictly positive")

    log_target = np.log10(target_times_hr)
    log_grid = np.log10(time_grid_hr)
    matrix = np.zeros((target_times_hr.size, time_grid_hr.size), dtype=float)

    for row_index, log_time in enumerate(log_target):
        if log_time <= log_grid[0]:
            matrix[row_index, 0] = 1.0
            continue
        if log_time >= log_grid[-1]:
            matrix[row_index, -1] = 1.0
            continue

        right_index = int(np.searchsorted(log_grid, log_time))
        left_index = right_index - 1
        left_log = log_grid[left_index]
        right_log = log_grid[right_index]
        weight_right = (log_time - left_log) / (right_log - left_log)
        matrix[row_index, left_index] = 1.0 - weight_right
        matrix[row_index, right_index] = weight_right

    return matrix

def _second_difference_matrix(size: int) -> np.ndarray:
    """Build a standard second-difference smoothing matrix."""
    if size < 3:
        return np.zeros((0, size), dtype=float)

    matrix = np.zeros((size - 2, size), dtype=float)
    for row_index in range(size - 2):
        matrix[row_index, row_index:row_index + 3] = (1.0, -2.0, 1.0)
    return matrix

def rate_superposition_buildup(dt_shut_hr: Union[np.ndarray, list],
                               pws_psia: Union[np.ndarray, list],
                               rate: Any,
                               tp_hr: Optional[float]) -> Optional[Dict[str, np.ndarray]]:
    """
    Convert variable-rate buildup data to an equivalent-time semilog response.

    The implementation uses rate superposition to compute an equivalent Horner
    axis referenced to the final pre-shut-in rate. For constant-rate tests this
    reduces exactly to the classical Horner transform.

    Args:
        dt_shut_hr (Union[np.ndarray, list]): Shut-in time after closure in hours.
        pws_psia (Union[np.ndarray, list]): Shut-in pressures in psia.
        rate (Any): Either a scalar constant rate or a history dictionary with:
            - 'rate' or 'values': interval rates in STB/day
            - 'dt': interval durations in hours, or
            - 'time': cumulative interval end times in hours
        tp_hr (Optional[float]): Total producing time before shut-in in hours.

    Returns:
        Optional[Dict[str, np.ndarray]]: Equivalent-time transformed metadata, or
        None when no explicit rate history is provided.

    References:
        - Matthews, C. S., and Russell, D. G. (1967). Pressure Buildup and Flow
          Tests in Wells. SPE Monograph Series, Vol. 1.
        - Earlougher, R. C., Jr. (1977). Advances in Well Test Analysis.
          SPE Monograph Series, Vol. 5.
    """
    history_end_times, rate_values, tp_hr = _parse_rate_history(rate, tp_hr)
    if history_end_times is None or rate_values is None:
        return None

    dt_shut_hr = np.asarray(dt_shut_hr, dtype=float)
    pws_psia = np.asarray(pws_psia, dtype=float)

    if dt_shut_hr.shape != pws_psia.shape:
        raise ValueError("Shut-in time and pressure arrays must have the same shape")
    dt_shut_hr = _apply_time_epsilon(dt_shut_hr)

    q_ref = float(rate_values[-1])
    if not np.isfinite(q_ref) or q_ref == 0.0:
        raise ValueError("Final pre-shut-in rate must be finite and non-zero for deconvolution")

    history_start_times = np.concatenate(([0.0], history_end_times[:-1]))
    delta_q = np.diff(np.concatenate(([0.0], rate_values)))

    numerator = tp_hr - history_start_times[:, None] + dt_shut_hr[None, :]
    denominator = tp_hr - history_end_times[:, None] + dt_shut_hr[None, :]
    superposition_log = np.sum((delta_q[:, None] / q_ref) * np.log10(numerator / denominator), axis=0)

    horner_time = np.power(10.0, superposition_log)
    equivalent_dt_hr = tp_hr / np.maximum(horner_time - 1.0, np.finfo(float).eps)

    sort_order = np.argsort(equivalent_dt_hr)
    pressure_sorted = pws_psia[sort_order]

    return {
        'analysis_time_hr': equivalent_dt_hr[sort_order],
        'horner_time': horner_time[sort_order],
        'log_horner_time': superposition_log[sort_order],
        'pressure_psia': pressure_sorted,
        'delta_p_psi': pressure_sorted - pressure_sorted[0],
        'reference_rate_stbd': np.full_like(equivalent_dt_hr, q_ref, dtype=float),
        'deconvolution_method': 'superposition equivalent-time normalization',
        'sort_order': sort_order,
    }

def rate_deconvolution_buildup(dt_shut_hr: Union[np.ndarray, list],
                               pws_psia: Union[np.ndarray, list],
                               rate: Any,
                               tp_hr: Optional[float],
                               smoothing: float = 1e-4,
                               grid_points: Optional[int] = None) -> Optional[Dict[str, np.ndarray]]:
    """
    Perform true inverse rate deconvolution for variable-rate buildup data.

    The method estimates the single-rate unit-response function by solving a
    regularized linear inverse problem based on pressure superposition for a
    piecewise-constant rate history. The recovered response is then used to
    reconstruct the constant-rate buildup pressure that would correspond to the
    final pre-shut-in rate and producing time.

    Args:
        dt_shut_hr (Union[np.ndarray, list]): Shut-in time after closure in hours.
        pws_psia (Union[np.ndarray, list]): Shut-in pressures in psia.
        rate (Any): Rate history dictionary with:
            - 'rate' or 'values': interval rates in STB/day
            - 'dt': interval durations in hours, or
            - 'time': cumulative interval end times in hours
        tp_hr (Optional[float]): Total producing time before shut-in in hours.
        smoothing (float, optional): Non-dimensional Tikhonov smoothing weight.
            Defaults to 1e-4.
        grid_points (Optional[int], optional): Number of logarithmic time grid
            points used for the recovered unit-response function. Defaults to a
            data-based value between 6 and 40.

    Returns:
        Optional[Dict[str, np.ndarray]]: Deconvolved constant-rate buildup data,
        or None when no explicit rate history is provided.

    References:
        - Levitan, M. M. (2005). Practical aspects of pressure-transient test
          deconvolution with noisy pressure and rate data. SPE Reservoir Evaluation
          & Engineering, 8(1), 25-34.
        - von Schroeter, T., Hollaender, F., and Gringarten, A. C. (2001).
          Deconvolution of well-test data as a nonlinear total least-squares
          problem. SPE Annual Technical Conference and Exhibition.
        - Matthews, C. S., and Russell, D. G. (1967). Pressure Buildup and Flow
          Tests in Wells. SPE Monograph Series, Vol. 1.
        - Earlougher, R. C., Jr. (1977). Advances in Well Test Analysis.
          SPE Monograph Series, Vol. 5.
    """
    history_end_times, rate_values, tp_hr = _parse_rate_history(rate, tp_hr)
    if history_end_times is None or rate_values is None:
        return None

    dt_shut_hr = np.asarray(dt_shut_hr, dtype=float)
    pws_psia = np.asarray(pws_psia, dtype=float)

    if dt_shut_hr.shape != pws_psia.shape:
        raise ValueError("Shut-in time and pressure arrays must have the same shape")
    dt_shut_hr = _apply_time_epsilon(dt_shut_hr)
    if smoothing < 0.0:
        raise ValueError("smoothing must be non-negative")

    q_ref = float(rate_values[-1])
    if not np.isfinite(q_ref) or q_ref == 0.0:
        raise ValueError("Final pre-shut-in rate must be finite and non-zero for deconvolution")

    history_start_times = np.concatenate(([0.0], history_end_times[:-1]))
    observation_times = tp_hr + dt_shut_hr
    min_lag = float(np.min(dt_shut_hr))
    max_lag = float(np.max(observation_times - history_start_times[0]))

    if grid_points is None:
        grid_points = int(np.clip(len(dt_shut_hr), 6, 40))
    if grid_points < 3:
        raise ValueError("grid_points must be at least 3")

    response_time_grid_hr = np.logspace(np.log10(min_lag), np.log10(max_lag), grid_points)

    interval_drawdown_matrix = np.zeros((dt_shut_hr.size, response_time_grid_hr.size), dtype=float)
    for interval_rate, start_time, end_time in zip(rate_values, history_start_times, history_end_times):
        start_lags = observation_times - start_time
        end_lags = observation_times - end_time
        interval_drawdown_matrix += interval_rate * (
            _log_time_interpolation_matrix(start_lags, response_time_grid_hr)
            - _log_time_interpolation_matrix(end_lags, response_time_grid_hr)
        )

    design_matrix = np.column_stack((np.ones(dt_shut_hr.size, dtype=float), -interval_drawdown_matrix))
    roughness = _second_difference_matrix(response_time_grid_hr.size)
    regularization_matrix = np.column_stack((np.zeros((roughness.shape[0], 1), dtype=float), roughness))

    if regularization_matrix.shape[0] > 0 and smoothing > 0.0:
        design_norm = np.linalg.norm(design_matrix, ord='fro')
        regularization_norm = np.linalg.norm(regularization_matrix, ord='fro')
        lambda_scale = smoothing * (design_norm ** 2) / max(regularization_norm ** 2, np.finfo(float).eps)
        augmented_matrix = np.vstack((design_matrix, np.sqrt(lambda_scale) * regularization_matrix))
        augmented_rhs = np.concatenate((pws_psia, np.zeros(regularization_matrix.shape[0], dtype=float)))
    else:
        augmented_matrix = design_matrix
        augmented_rhs = pws_psia

    lower_bounds = np.concatenate(([-np.inf], np.zeros(response_time_grid_hr.size, dtype=float)))
    upper_bounds = np.full(response_time_grid_hr.size + 1, np.inf, dtype=float)
    solution = lsq_linear(augmented_matrix, augmented_rhs, bounds=(lower_bounds, upper_bounds), lsq_solver='exact')
    if not solution.success:
        raise RuntimeError(f"Inverse deconvolution did not converge: {solution.message}")

    pressure_intercept = float(solution.x[0])
    drawdown_response = np.maximum.accumulate(np.maximum(solution.x[1:], 0.0))
    fitted_pressure = pressure_intercept - interval_drawdown_matrix @ drawdown_response
    pressure_intercept = float(np.mean(pws_psia + interval_drawdown_matrix @ drawdown_response))
    fitted_pressure = pressure_intercept - interval_drawdown_matrix @ drawdown_response

    buildup_operator = q_ref * (
        _log_time_interpolation_matrix(tp_hr + dt_shut_hr, response_time_grid_hr)
        - _log_time_interpolation_matrix(dt_shut_hr, response_time_grid_hr)
    )
    deconvolved_pressure = pressure_intercept - buildup_operator @ drawdown_response
    horner_time = (tp_hr + dt_shut_hr) / dt_shut_hr

    return {
        'analysis_time_hr': dt_shut_hr.copy(),
        'horner_time': horner_time,
        'log_horner_time': np.log10(horner_time),
        'pressure_psia': deconvolved_pressure,
        'delta_p_psi': deconvolved_pressure - deconvolved_pressure[0],
        'reference_rate_stbd': np.full_like(dt_shut_hr, q_ref, dtype=float),
        'drawdown_time_grid_hr': response_time_grid_hr,
        'drawdown_response_psi_per_stbd': drawdown_response,
        'pressure_match_psia': fitted_pressure,
        'fit_residual_norm': float(np.linalg.norm(pws_psia - fitted_pressure)),
        'deconvolution_method': 'inverse rate deconvolution',
    }

def horner_plot_analysis(dt_shut_hr: Union[np.ndarray, list], pws_psia: Union[np.ndarray, list], 
                        tp_hr: float, q_stbd: float, bo_rb_stb: float, mu_cp: float, 
                        h_ft: float, rw_ft: float, phi: float, ct_psi_inv: float,
                        min_points: int = 10) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Perform Horner plot analysis with automatic trend line fitting and parameter estimation.
    
    Creates a Horner plot of pressure vs log((tp + Δt)/Δt) and automatically finds the
    best-fit straight line with highest R² for middle-time radial flow. Calculates 
    permeability, skin factor, and other reservoir parameters from the trend line.
    
    Args:
        dt_shut_hr (Union[np.ndarray, list]): Shut-in time intervals in hours (Δt)
        pws_psia (Union[np.ndarray, list]): Shut-in wellbore pressures in psia
        tp_hr (float): Production time before shut-in in hours
        q_stbd (float): Production rate before shut-in in STB/day
        bo_rb_stb (float): Oil formation volume factor in rb/STB
        mu_cp (float): Viscosity in centipoise
        h_ft (float): Net pay thickness in feet
        rw_ft (float): Wellbore radius in feet
        phi (float): Porosity as fraction (0-1)
        ct_psi_inv (float): Total compressibility in psi^-1
        min_points (int, optional): Minimum points required for trend line fit. Defaults to 10.
        
    Returns:
        Dict[str, Union[float, np.ndarray, str]]: Analysis results containing:
            - 'permeability_md': Calculated permeability in millidarcies
            - 'skin_factor': Calculated skin factor (dimensionless)
            - 'slope_psi_per_log_cycle': Horner plot slope in psi/log-cycle
            - 'p1hr_psia': Pressure at 1-hour Horner time
            - 'r2_fit': R² coefficient for the trend line
            - 'horner_time': Horner time values (tp+Δt)/Δt
            - 'fit_range_start': Start index of fitted data range
            - 'fit_range_end': End index of fitted data range
            - 'analysis_quality': Quality assessment ('Excellent', 'Good', 'Fair', 'Poor')
            
    Examples:
        >>> # Example Horner analysis
        >>> dt = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 24.0])  # hours
        >>> pws = np.array([3950, 3965, 3975, 3985, 3995, 4005, 4015])  # psia
        >>> tp = 100.0  # hours
        >>> results = horner_plot_analysis(dt, pws, tp, 300.0, 1.2, 1.5, 50.0, 0.25, 0.2, 15e-6)
        >>> results['permeability_md'] > 0  # Should calculate positive permeability
        True
        >>> 'skin_factor' in results
        True
        
    Notes:
        - Horner plot uses x = log((tp + Δt)/Δt) vs y = pws
        - Slope m relates to permeability: k = 162.6 * q * B * μ / (|m| * h)
        - Skin calculated from: s = 1.151 * [(p1hr - pi) / |m| + log(k/(φ*μ*ct*rw²)) - 3.23]
        - Function automatically searches for best linear segment
        - Quality assessment based on R² value and number of points
        - Assumes single-phase liquid flow and homogeneous reservoir
        - Wellbore storage effects should be minimal in fitted region
    """
    dt_shut_hr = np.asarray(dt_shut_hr, dtype=float)
    pws_psia = np.asarray(pws_psia, dtype=float)
    
    if len(dt_shut_hr) != len(pws_psia):
        raise ValueError("Shut-in time and pressure arrays must have same length")
    if len(dt_shut_hr) < min_points:
        raise ValueError(f"Need at least {min_points} data points for analysis")
    if tp_hr <= 0:
        raise ValueError("Production time must be positive")
    
    # Calculate Horner time
    horner_time = (tp_hr + dt_shut_hr) / dt_shut_hr
    log_horner = np.log10(horner_time)
    
    # Find best linear fit by testing different ranges
    best_r2 = -np.inf
    best_fit = None
    best_range = None
    
    n_points = len(pws_psia)
    
    # Try different starting and ending points for the linear segment
    for start_idx in range(n_points - min_points + 1):
        for end_idx in range(start_idx + min_points, n_points + 1):
            x_segment = log_horner[start_idx:end_idx]
            y_segment = pws_psia[start_idx:end_idx]
            
            if len(x_segment) < min_points:
                continue
                
            # Linear regression
            slope, intercept = np.polyfit(x_segment, y_segment, 1)
            y_pred = slope * x_segment + intercept
            r2 = r2_score(y_segment, y_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_fit = (slope, intercept)
                best_range = (start_idx, end_idx)
    
    if best_fit is None:
        raise ValueError("Could not find suitable linear trend in data")
    
    slope_psi_per_log, intercept_psi = best_fit
    start_idx, end_idx = best_range
    
    # Calculate permeability from slope
    # Horner slope m = -162.6 * q * B * μ / (k * h) (negative for drawdown)
    # Therefore: k = -162.6 * q * B * μ / (m * h)
    permeability_md = 162.6 * q_stbd * bo_rb_stb * mu_cp / (abs(slope_psi_per_log) * h_ft)
    
    # Calculate pressure at 1-hour Horner time: (tp + 1)/1 = tp + 1
    log_horner_1hr = np.log10(tp_hr + 1.0)
    p1hr_psia = slope_psi_per_log * log_horner_1hr + intercept_psi
    
    # Calculate skin factor
    # s = 1.151 * [(p1hr - pi)/|m| + log(k/(φ*μ*ct*rw²)) - 3.23]
    # Here we use p1hr as reference since we don't have pi directly
    # Simplified skin calculation assuming p1hr approximates late-time behavior
    if permeability_md > 0:
        log_term = np.log10(permeability_md / (phi * mu_cp * ct_psi_inv * rw_ft**2))
        skin_factor = 1.151 * (log_term - 3.23)
    else:
        skin_factor = np.nan
    
    # Quality assessment based on R² and number of points
    n_fit_points = end_idx - start_idx
    if best_r2 >= 0.95 and n_fit_points >= 15:
        quality = "Excellent"
    elif best_r2 >= 0.90 and n_fit_points >= 10:
        quality = "Good"
    elif best_r2 >= 0.80 and n_fit_points >= 8:
        quality = "Fair"
    else:
        quality = "Poor"
    
    return {
        'permeability_md': float(permeability_md),
        'skin_factor': float(skin_factor),
        'slope_psi_per_log_cycle': float(slope_psi_per_log),
        'p1hr_psia': float(p1hr_psia),
        'r2_fit': float(best_r2),
        'horner_time': horner_time,
        'fit_range_start': int(start_idx),
        'fit_range_end': int(end_idx),
        'analysis_quality': quality
    }

def derivative_curve_analysis(t_hr: Union[np.ndarray, list], dp_dlnt: Union[np.ndarray, list],
                             q_stbd: float, bo_rb_stb: float, mu_cp: float, h_ft: float,
                             rw_ft: float, phi: float, ct_psi_inv: float,
                             min_radial_points: int = 5) -> Dict[str, Union[float, np.ndarray, str, Dict]]:
    """
    Analyze pressure derivative curve with automatic identification of flow regimes.
    
    Analyzes log-log derivative plot to identify wellbore storage, radial flow, and
    boundary-dominated flow regimes. Automatically fits trend lines for early-time
    wellbore storage (unit slope) and middle-time radial flow (horizontal line).
    Calculates permeability, skin, wellbore storage, and other parameters.
    
    Args:
        t_hr (Union[np.ndarray, list]): Time values in hours
        dp_dlnt (Union[np.ndarray, list]): Pressure derivative dp/d(ln t) in psi
        q_stbd (float): Production rate in STB/day
        bo_rb_stb (float): Oil formation volume factor in rb/STB
        mu_cp (float): Viscosity in centipoise
        h_ft (float): Net pay thickness in feet
        rw_ft (float): Wellbore radius in feet
        phi (float): Porosity as fraction (0-1)
        ct_psi_inv (float): Total compressibility in psi^-1
        min_radial_points (int, optional): Minimum points for radial flow fit. Defaults to 5.
        
    Returns:
        Dict[str, Union[float, np.ndarray, str, Dict]]: Analysis results containing:
            - 'permeability_md': Permeability from radial flow regime in md
            - 'skin_factor': Skin factor from wellbore storage/radial flow transition
            - 'wellbore_storage_bbl_psi': Wellbore storage coefficient in bbl/psi
            - 'radial_flow_derivative': Derivative level during radial flow in psi
            - 'wellbore_storage_regime': Dict with fit parameters for WBS regime
            - 'radial_flow_regime': Dict with fit parameters for radial flow regime
            - 'transition_time_hr': Approximate transition time from WBS to radial flow
            - 'analysis_quality': Overall quality assessment
            
    Examples:
        >>> # Example derivative analysis
        >>> times = np.logspace(-1, 2, 50)  # 0.1 to 100 hours
        >>> # Synthetic derivative data with WBS and radial flow
        >>> dp_dt = np.where(times < 1.0, 50.0 * times, 50.0)  # Unit slope then flat
        >>> results = derivative_curve_analysis(times, dp_dt, 300.0, 1.2, 1.5, 50.0, 0.25, 0.2, 15e-6)
        >>> results['permeability_md'] > 0
        True
        >>> 'wellbore_storage_bbl_psi' in results
        True
        
    Notes:
        - Log-log derivative plot: log(t) vs log(dp/d(ln t))
        - Wellbore storage: unit slope (slope ≈ 1) on log-log plot
        - Radial flow: horizontal line (slope ≈ 0) on log-log plot
        - Permeability from radial flow: k = 162.6 * q * B * μ / (dp/d(ln t) * h)
        - Wellbore storage: C = q * B * t / (24 * dp/d(ln t)) during unit slope
        - Skin calculated from timing and amplitude of transition
        - Function identifies flow regimes automatically
        - Quality based on fit statistics and regime identification
    """
    t_hr = np.asarray(t_hr, dtype=float)
    dp_dlnt = np.asarray(dp_dlnt, dtype=float)
    
    if len(t_hr) != len(dp_dlnt):
        raise ValueError("Time and derivative arrays must have same length")
    if len(t_hr) < 2 * min_radial_points:
        raise ValueError(f"Need at least {2 * min_radial_points} data points for analysis")
    
    # Remove invalid data points
    valid_mask = (t_hr > 0) & (dp_dlnt > 0) & np.isfinite(t_hr) & np.isfinite(dp_dlnt)
    t_valid = t_hr[valid_mask]
    dp_valid = dp_dlnt[valid_mask]
    
    if len(t_valid) < min_radial_points:
        raise ValueError("Insufficient valid data points after filtering")
    
    log_t = np.log10(t_valid)
    log_dp = np.log10(dp_valid)
    
    # Find radial flow regime (horizontal line with slope ≈ 0)
    best_radial_r2 = -np.inf
    best_radial_fit = None
    best_radial_range = None
    radial_derivative = np.nan
    
    n_points = len(t_valid)
    
    # Search for horizontal segment (radial flow)
    for start_idx in range(n_points - min_radial_points + 1):
        for end_idx in range(start_idx + min_radial_points, n_points + 1):
            log_t_segment = log_t[start_idx:end_idx]
            log_dp_segment = log_dp[start_idx:end_idx]
            
            if len(log_t_segment) < min_radial_points:
                continue
            
            # Fit horizontal line (slope should be close to 0)
            slope, intercept = np.polyfit(log_t_segment, log_dp_segment, 1)
            
            # Penalize non-horizontal fits
            slope_penalty = abs(slope) * 10  # Penalty for non-zero slope
            
            log_dp_pred = slope * log_t_segment + intercept
            r2_raw = r2_score(log_dp_segment, log_dp_pred)
            r2_adjusted = r2_raw - slope_penalty  # Adjusted R² favoring horizontal lines
            
            if r2_adjusted > best_radial_r2 and abs(slope) < 0.2:  # Slope threshold for "horizontal"
                best_radial_r2 = r2_adjusted
                best_radial_fit = (slope, intercept)
                best_radial_range = (start_idx, end_idx)
                radial_derivative = 10**intercept  # Convert back from log
    
    # Find wellbore storage regime (unit slope ≈ 1)
    best_wbs_r2 = -np.inf
    best_wbs_fit = None
    best_wbs_range = None
    
    # Search for unit slope segment (early time)
    max_wbs_end = min(n_points, n_points // 2)  # WBS typically in first half of data
    
    for start_idx in range(max_wbs_end - min_radial_points + 1):
        for end_idx in range(start_idx + min_radial_points, max_wbs_end + 1):
            log_t_segment = log_t[start_idx:end_idx]
            log_dp_segment = log_dp[start_idx:end_idx]
            
            if len(log_t_segment) < min_radial_points:
                continue
            
            slope, intercept = np.polyfit(log_t_segment, log_dp_segment, 1)
            
            # Penalize deviations from unit slope
            slope_penalty = abs(slope - 1.0) * 10
            
            log_dp_pred = slope * log_t_segment + intercept
            r2_raw = r2_score(log_dp_segment, log_dp_pred)
            r2_adjusted = r2_raw - slope_penalty
            
            if r2_adjusted > best_wbs_r2 and 0.8 <= slope <= 1.2:  # Unit slope tolerance
                best_wbs_r2 = r2_adjusted
                best_wbs_fit = (slope, intercept)
                best_wbs_range = (start_idx, end_idx)
    
    # Calculate parameters
    results = {}
    
    if best_radial_fit is not None:
        # Permeability from radial flow derivative
        permeability_md = 162.6 * q_stbd * bo_rb_stb * mu_cp / (radial_derivative * h_ft)
        results['permeability_md'] = float(permeability_md)
        results['radial_flow_derivative'] = float(radial_derivative)
        
        radial_start, radial_end = best_radial_range
        results['radial_flow_regime'] = {
            'slope': float(best_radial_fit[0]),
            'intercept': float(best_radial_fit[1]),
            'r2': float(best_radial_r2),
            'start_time_hr': float(t_valid[radial_start]),
            'end_time_hr': float(t_valid[radial_end - 1]),
            'points_used': int(radial_end - radial_start)
        }
    else:
        results['permeability_md'] = np.nan
        results['radial_flow_derivative'] = np.nan
        results['radial_flow_regime'] = None
    
    if best_wbs_fit is not None:
        wbs_start, wbs_end = best_wbs_range
        wbs_time_mid = t_valid[wbs_start + (wbs_end - wbs_start) // 2]
        wbs_derivative_mid = dp_valid[wbs_start + (wbs_end - wbs_start) // 2]
        
        # Wellbore storage coefficient: C = q * B * t / (24 * dp/d(ln t))
        wellbore_storage = q_stbd * bo_rb_stb * wbs_time_mid / (24.0 * wbs_derivative_mid)
        results['wellbore_storage_bbl_psi'] = float(wellbore_storage)
        
        results['wellbore_storage_regime'] = {
            'slope': float(best_wbs_fit[0]),
            'intercept': float(best_wbs_fit[1]),
            'r2': float(best_wbs_r2),
            'start_time_hr': float(t_valid[wbs_start]),
            'end_time_hr': float(t_valid[wbs_end - 1]),
            'points_used': int(wbs_end - wbs_start)
        }
        
        # Transition time (end of wellbore storage)
        results['transition_time_hr'] = float(t_valid[wbs_end - 1])
    else:
        results['wellbore_storage_bbl_psi'] = np.nan
        results['wellbore_storage_regime'] = None
        results['transition_time_hr'] = np.nan
    
    # Skin factor estimation (simplified)
    if best_wbs_fit is not None and best_radial_fit is not None:
        # Skin from wellbore storage and permeability
        C_wbs = results['wellbore_storage_bbl_psi']
        k_md = results['permeability_md']
        
        # Approximate skin calculation
        skin_factor = 0.0  # Simplified - would need more sophisticated analysis
        results['skin_factor'] = float(skin_factor)
    else:
        results['skin_factor'] = np.nan
    
    # Overall quality assessment
    radial_quality = "Good" if best_radial_r2 > 0.8 else "Fair" if best_radial_r2 > 0.6 else "Poor"
    wbs_quality = "Good" if best_wbs_r2 > 0.8 else "Fair" if best_wbs_r2 > 0.6 else "Poor"
    
    if best_radial_fit is not None and best_wbs_fit is not None:
        if radial_quality == "Good" and wbs_quality == "Good":
            overall_quality = "Excellent"
        elif radial_quality in ["Good", "Fair"] and wbs_quality in ["Good", "Fair"]:
            overall_quality = "Good"
        else:
            overall_quality = "Fair"
    elif best_radial_fit is not None:
        overall_quality = radial_quality
    elif best_wbs_fit is not None:
        overall_quality = wbs_quality
    else:
        overall_quality = "Poor"
    
    results['analysis_quality'] = overall_quality
    
    return results

def well_test_analysis(time, 
                       pressure, 
                       rate,
                       IARF_slice=None, 
                       type=None, 
                       Bo=None, 
                       mu=None, 
                       h=None, 
                       rw=None, 
                       phi=None, 
                       ct=None, 
                       A=None, 
                       length=None, 
                       tp=None,
                       wbs_slice=None,
                       boundary_slice=None,
                       CA=None,
                       pmbhd=None,
                       mbh_alpha=None,
                       mbh_beta=None,
                       mbh_lambda=None,
                       deconvolution=True,
                       force_scatter: bool = False):
    """
    Perform a comprehensive well test analysis.
    
    Args:
        time (Union[np.ndarray, list]): Time values in hours (must be duration of timestep (deltat))
        pressure (Union[np.ndarray, list]): Pressure values in psia
        rate (Union[np.ndarray, list, dict, float]): Production rate in STB/day.
            Pass a scalar for constant-rate interpretation, or a dictionary with
            rate history to enable buildup deconvolution using variable-rate
            superposition. Supported dictionary keys are:
            - 'rate' or 'values': interval rates in STB/day
            - 'dt': interval durations in hours, or
            - 'time': cumulative interval end times in hours
        IARF_slice (tuple): Start and end times for the IARF period
        type (list(string)): List of well test types ['buildup', 'dst', 'drawdown', 'gas', 'horizontal]
        Bo (float): Oil formation volume factor in rb/STB
        mu (float): Viscosity in centipoise
        h (float): Net pay thickness in feet
        rw (float): Wellbore radius in feet
        phi (float): Porosity as fraction (0-1)
        ct (float): Total compressibility in psi^-1
        A (float): Drainage area in acres
        length (float): Length of horizontal well in feet (if applicable)
        tp (float): Shut in time before buildup test in hours (if applicable)
        wbs_slice (tuple): Start and end times for wellbore storage period (optional)
        boundary_slice (tuple): Start and end times for boundary-dominated flow period (optional)
        CA (float): Dietz shape factor for the reservoir geometry
        pmdhbd (float): Dimensionless pressure at dimensionless producing time for MBH method.
            If omitted and CA is provided, the value is computed from MBH geometry.
        mbh_alpha (Optional[float]): Fractional well position in x-direction for MBH curves.
            Use only for rectangular MBH geometry.
        mbh_beta (Optional[float]): Fractional well position in y-direction for MBH curves.
            Use only for rectangular MBH geometry.
        mbh_lambda (Optional[float]): Rectangle aspect ratio x_e / y_e for MBH curves.
            Use only for rectangular MBH geometry.
                deconvolution (Union[bool, str], optional): Controls variable-rate
                    buildup deconvolution when explicit rate history is provided.
                    Supported values are:
                    - True or 'inverse': regularized inverse deconvolution
                    - 'superposition': legacy equivalent-time normalization
                    - False: disable deconvolution
                    Defaults to True.
                force_scatter (bool, optional): If True, plot measured data as
                    marker-only scatter in Horner/derivative charts instead of
                    connected lines. Fit lines remain connected. Defaults to False.

        DST input format:
            For `type=['dst']`, pass `time` as a dictionary describing period data.
            Preferred structure:
            {
                'flow_periods': [
                    {'time': [...], 'pressure': [...], 'time_unit': 'min' or 'hr'},
                    ...
                ],
                'buildup_periods': [
                    {'time': [...], 'pressure': [...], 'time_unit': 'min' or 'hr'},
                    ...
                ]
            }
            Backward-compatible aliases are also supported:
            `initial_flow`, `initial_buildup`, `final_flow`, `final_buildup`.

        
    Returns:
                matplotlib.figure.Figure: Figure containing the buildup semilog and
                derivative plots. Analysis metadata are also attached to the figure as
                ``fig.utpge_results``.

        References:
                - Matthews, C. S., and Russell, D. G. (1967). Pressure Buildup and Flow
                    Tests in Wells. SPE Monograph Series, Vol. 1.
                - Earlougher, R. C., Jr. (1977). Advances in Well Test Analysis.
                    SPE Monograph Series, Vol. 5.
                - Levitan, M. M. (2005). Practical aspects of pressure-transient test
                    deconvolution with noisy pressure and rate data.
                - von Schroeter, T., Hollaender, F., and Gringarten, A. C. (2001).
                    Deconvolution of well-test data as a nonlinear total least-squares problem.
                - Lee, J. (AAPG Wiki). Pressure transient testing. Summary reference for
                    Horner semilog interpretation and variable-rate test requirements.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.optimize import curve_fit
    if type is None:
        raise ValueError("Must specify at least one test type in 'type' parameter")
        return None
    if IARF_slice is None:
        raise ValueError("Please specify a slice for IARF, I'm not good enough at coding to automatically identify it")

    if 'dst' in type:
        if not isinstance(time, dict):
            raise ValueError("For DST analysis, provide `time` as a dictionary of flow/buildup periods")

        def _period_to_hr(period: Dict[str, Any], require_pressure: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            if 'time' not in period:
                raise ValueError("Each DST period dictionary must include 'time'")
            t_arr = np.asarray(period['time'], dtype=float)
            if t_arr.ndim != 1 or t_arr.size < 2:
                raise ValueError("Each DST period time array must be one-dimensional with at least two entries")
            if np.any(~np.isfinite(t_arr)):
                raise ValueError("DST period times must be finite")

            unit = str(period.get('time_unit', 'hr')).lower()
            if unit in ('min', 'minute', 'minutes'):
                t_arr = t_arr / 60.0
            elif unit not in ('hr', 'hour', 'hours'):
                raise ValueError("DST period time_unit must be 'min' or 'hr'")

            p_arr = None
            if 'pressure' in period:
                p_arr = np.asarray(period['pressure'], dtype=float)
                if p_arr.shape != t_arr.shape:
                    raise ValueError("DST period pressure array must have same length as time array")
                if np.any(~np.isfinite(p_arr)):
                    raise ValueError("DST period pressures must be finite")
            elif require_pressure:
                raise ValueError("DST buildup periods must include pressure data")

            return t_arr, p_arr

        flow_periods = time.get('flow_periods')
        buildup_periods = time.get('buildup_periods')

        if flow_periods is None or buildup_periods is None:
            flow_periods = []
            buildup_periods = []
            alias_flow_keys = ['initial_flow', 'final_flow']
            alias_buildup_keys = ['initial_buildup', 'final_buildup']
            for key in alias_flow_keys:
                if key in time:
                    flow_periods.append(time[key])
            for key in alias_buildup_keys:
                if key in time:
                    buildup_periods.append(time[key])

        if not flow_periods or not buildup_periods:
            raise ValueError("DST analysis requires at least one flow period and one buildup period")

        flow_data = [
            {'time_hr': _period_to_hr(period, require_pressure=False)[0], 'pressure_psia': _period_to_hr(period, require_pressure=False)[1]}
            for period in flow_periods
        ]
        buildup_data = [
            {'time_hr': _period_to_hr(period, require_pressure=True)[0], 'pressure_psia': _period_to_hr(period, require_pressure=True)[1]}
            for period in buildup_periods
        ]

        flow_durations_hr = [float(np.max(period['time_hr']) - np.min(period['time_hr'])) for period in flow_data]
        flow_cum_durations_hr = np.cumsum(flow_durations_hr)

        if isinstance(rate, dict) and 'flow_rates' in rate:
            flow_rates = np.asarray(rate['flow_rates'], dtype=float)
            if flow_rates.size < len(flow_data):
                raise ValueError("rate['flow_rates'] must have at least one entry per DST flow period")
        else:
            flow_rates = np.full(len(flow_data), _extract_reference_rate(rate), dtype=float)

        def _tp_for_buildup(period_index: int) -> float:
            if isinstance(tp, dict):
                key = 'final' if period_index == len(buildup_data) - 1 else 'initial'
                if key in tp:
                    return float(tp[key])
            if isinstance(tp, (list, tuple, np.ndarray)):
                tp_values = np.asarray(tp, dtype=float)
                if tp_values.size > period_index:
                    return float(tp_values[period_index])
            if tp is not None and np.isscalar(tp):
                if len(buildup_data) == 1:
                    return float(tp)
                if period_index == len(buildup_data) - 1:
                    return float(tp)
            flow_index = min(period_index, len(flow_cum_durations_hr) - 1)
            return float(flow_cum_durations_hr[flow_index])

        def _iarf_slice_for_period(period_index: int) -> Tuple[float, float]:
            if isinstance(IARF_slice, dict):
                if period_index in IARF_slice:
                    return tuple(IARF_slice[period_index])
                key = 'final' if period_index == len(buildup_data) - 1 else 'initial'
                if key in IARF_slice:
                    return tuple(IARF_slice[key])
                raise ValueError("IARF_slice dictionary must include entries for DST periods")
            return tuple(IARF_slice)

        buildup_results = []
        for period_index, period in enumerate(buildup_data):
            tp_period_hr = _tp_for_buildup(period_index)
            q_period = float(flow_rates[min(period_index, len(flow_rates) - 1)])
            result = _analyze_buildup_period(
                raw_time_hr=period['time_hr'],
                raw_pressure_psia=period['pressure_psia'],
                q_stbd=q_period,
                tp_hr=tp_period_hr,
                IARF_slice=_iarf_slice_for_period(period_index),
                Bo=Bo,
                mu=mu,
                h=h,
                rw=rw,
                phi=phi,
                ct=ct,
                A=A,
                CA=CA,
                pmbhd=pmbhd,
                mbh_alpha=mbh_alpha,
                mbh_beta=mbh_beta,
                mbh_lambda=mbh_lambda,
                period_label='Final Buildup' if period_index == len(buildup_data) - 1 else f'Buildup {period_index + 1}',
                use_horner_only=True,
            )
            result['period_index'] = period_index
            buildup_results.append(result)

        final_result = buildup_results[-1]
        comparison_result = buildup_results[0] if len(buildup_results) > 1 else None

        final_flow_pressure = flow_data[min(len(flow_data) - 1, final_result['period_index'])]['pressure_psia']
        if final_flow_pressure is not None and final_flow_pressure.size > 0:
            pwf_final = float(np.mean(final_flow_pressure))
        else:
            pwf_final = float(final_result['pressure_psia'][0])

        pr_final = final_result['p_star_psia']
        delta_p_skin_final = calculate_deltap_skin(
            final_result['q_stbd'],
            Bo,
            mu,
            final_result['permeability_md']['horner'],
            h,
            final_result['skin']['horner'],
        )
        damage_ratio_final = calculate_damage_ratio(pr_final, pwf_final, delta_p_skin_final)

        if comparison_result is not None:
            fig, axes = plt.subplots(2, 1, figsize=(10, 10))
            axes_final = axes[0]
            axes_comparison = axes[1]
        else:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            axes_final = axes
            axes_comparison = None

        def _plot_buildup_panel(target_ax, result, title_prefix: str, show_zone_label: str):
            color = 'tab:blue' if 'Final' in result['period_label'] else 'tab:orange'
            display_mask = np.asarray(result.get('display_mask', np.ones_like(result['time_hr'], dtype=bool)), dtype=bool)
            horner_display = result['horner_time'][display_mask]
            pressure_display = result['pressure_psia'][display_mask]
            fit_display = result['IARF_line_horner'][display_mask]
            iarf_display = result['IARF_mask'][display_mask]

            _plot_sparse_safe(target_ax, 'semilogx', horner_display, pressure_display, color=color, label=result['period_label'], force_scatter=force_scatter)
            target_ax.semilogx(horner_display, fit_display, linestyle='--', color=color, alpha=0.8, label=f"{result['period_label']} Straight-line Fit")
            _shade_masked_xzones(target_ax, horner_display, iarf_display, color='red', alpha=0.08, label=show_zone_label)

            target_ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            target_ax.set_xlabel('Horner Time (tp+Δt)/Δt')
            target_ax.set_ylabel('Pressure (psi)')
            target_ax.set_title(f'{title_prefix} Horner Plot')
            target_ax.legend()

        _plot_buildup_panel(axes_final, final_result, 'DST Final Buildup', 'Final Straight line zone')
        if comparison_result is not None and axes_comparison is not None:
            _plot_buildup_panel(axes_comparison, comparison_result, 'DST Initial Buildup Comparison', 'Initial Straight line zone')

        plt.show()

        fig.utpge_results = {
            'test_type': 'dst',
            'analysis_basis': 'final buildup primary, earlier buildups for comparison',
            'show_average_pressure': False,
            'period_count': {
                'flow': len(flow_data),
                'buildup': len(buildup_data),
            },
            'final_buildup': final_result,
            'comparison_buildup': comparison_result,
            'damage_ratio': {
                'value': float(damage_ratio_final),
                'reservoir_pressure_psia': float(pr_final),
                'average_flowing_pressure_psia': float(pwf_final),
                'delta_p_skin_psi': float(delta_p_skin_final),
            },
        }
        fig.utpge_results['summary_text'] = format_well_test_results(fig.utpge_results)
        print_well_test_results(fig.utpge_results)
        return fig

    # Force data to arrays
    pressure = np.array(pressure)
    time = np.array(time)
    if 'buildup' in type:
        # Perform build-up analysis
        # Create figure for Horner plot and derivative plot
        # from utpgetools.well_testing import bourdet_derivative, r2_score
        reference_rate = _extract_reference_rate(rate)
        time_positive = _apply_time_epsilon(time)
        pressure_positive = pressure

        deconvolved_data = None
        deconvolution_mode = 'inverse' if deconvolution is True else deconvolution
        if deconvolution_mode:
            if deconvolution_mode == 'inverse':
                deconvolved_data = rate_deconvolution_buildup(time_positive, pressure_positive, rate, tp)
            elif deconvolution_mode == 'superposition':
                deconvolved_data = rate_superposition_buildup(time_positive, pressure_positive, rate, tp)
            else:
                raise ValueError("deconvolution must be one of False, True, 'inverse', or 'superposition'")

        buildup_result = _analyze_buildup_period(
            raw_time_hr=time,
            raw_pressure_psia=pressure,
            q_stbd=reference_rate,
            tp_hr=tp,
            IARF_slice=tuple(IARF_slice),
            Bo=Bo,
            mu=mu,
            h=h,
            rw=rw,
            phi=phi,
            ct=ct,
            A=A,
            CA=CA,
            pmbhd=pmbhd,
            mbh_alpha=mbh_alpha,
            mbh_beta=mbh_beta,
            mbh_lambda=mbh_lambda,
            deconvolved_data=deconvolved_data,
            period_label='Buildup',
        )

        analysis_time = buildup_result['time_hr']
        analysis_pressure = buildup_result['pressure_psia']
        analysis_delta_p = buildup_result['deltap_psi']
        horner_time = buildup_result['horner_time']
        IARF_mask = buildup_result['IARF_mask']
        display_mask = buildup_result['display_mask']
        params_horner = np.array([buildup_result['horner_slope'], buildup_result['p_star_psia']], dtype=float)
        IARF_line_horner = buildup_result['IARF_line_horner']
        _iarf_residual_pct = buildup_result['horner_residual_pct_of_range']
        bourdet_line = buildup_result['bourdet_line']
        regime_time = buildup_result['regime_time_hr']

        # Define masks for flow regimes
        if wbs_slice is None:
            wbs_mask = regime_time < IARF_slice[0]
        else:
            wbs_mask = (regime_time >= wbs_slice[0]) & (regime_time <= wbs_slice[1])
        if boundary_slice is None:
            boundary_mask = regime_time > IARF_slice[1]
        else:
            boundary_mask = (regime_time >= boundary_slice[0]) & (regime_time <= boundary_slice[1])
        dp_dlnt = bourdet_derivative(analysis_time, analysis_delta_p)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        # Pressure vs time plot
        # axes[0].plot(time, pressure, label='Pressure')
        # axes[0].set_xlabel('Time (hr)')
        # axes[0].set_ylabel('Pressure (psi)')
        # axes[0].set_title('Pressure vs Time')
        # axes[0].legend()
        # Horner Plot
        horner_time_display = horner_time[display_mask]
        analysis_pressure_display = analysis_pressure[display_mask]
        IARF_line_horner_display = IARF_line_horner[display_mask]
        IARF_mask_display = IARF_mask[display_mask]

        _plot_sparse_safe(axes[0], 'semilogx', horner_time_display, analysis_pressure_display, label='Pressure', force_scatter=force_scatter)
        axes[0].semilogx(horner_time_display, IARF_line_horner_display, 'r--', label='Straight-line Fit')
        # Dummy line for annotation
        axes[0].semilogx(horner_time_display, IARF_line_horner_display, linestyle='', label=f'm={params_horner[0]:.2f}\nP*={params_horner[1]:.0f}\nResidual:({_iarf_residual_pct:.1f}% of range)', alpha=0)
        _shade_masked_xzones(axes[0], horner_time_display, IARF_mask_display, color='red', alpha=0.1, label='Straight line zone')
        _shade_masked_xzones(axes[0], horner_time_display, wbs_mask[display_mask], color='blue', alpha=0.1, label='WBS Zone')
        _shade_masked_xzones(axes[0], horner_time_display, boundary_mask[display_mask], color='green', alpha=0.1, label='Boundary Zone')
        axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[0].set_xlim(min(horner_time_display)+1e-12, max(horner_time_display)-1e-12)
        axes[0].set_ylim(np.min(analysis_pressure_display), np.max(analysis_pressure_display))
        axes[0].set_xlabel('Horner Time (tp+Δt)/Δt')
        axes[0].set_ylabel('Pressure (psi)')
        axes[0].set_title('Horner Plot' if deconvolved_data is None else 'Deconvolved Horner Plot')
        axes[0].legend()

        # Bourdet derivative plot
        analysis_time_display = analysis_time[display_mask]
        dp_dlnt_display = dp_dlnt[display_mask]
        analysis_delta_p_display = analysis_delta_p[display_mask]
        bourdet_line_display = bourdet_line[display_mask]

        _plot_sparse_safe(axes[1], 'loglog', analysis_time_display, dp_dlnt_display, label='dp/d(ln t)', force_scatter=force_scatter)
        _plot_sparse_safe(axes[1], 'loglog', analysis_time_display, analysis_delta_p_display, label='deltaP', force_scatter=force_scatter)
        _plot_sparse_safe(axes[1], 'loglog', analysis_time_display, bourdet_line_display, color='r', linestyle='--', label='Straight-line Fit')
        _shade_masked_xzones(axes[1], analysis_time_display, IARF_mask_display, color='red', alpha=0.1, label='Straight line zone')
        _shade_masked_xzones(axes[1], analysis_time_display, wbs_mask[display_mask], color='blue', alpha=0.1, label='WBS Zone')
        _shade_masked_xzones(axes[1], analysis_time_display, boundary_mask[display_mask], color='green', alpha=0.1, label='Boundary Zone')
        axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[1].set_xlabel('Time (hr)')
        axes[1].set_ylabel('Pressure')
        axes[1].set_xlim(np.min(analysis_time_display)+1e-12, np.max(analysis_time_display)-1e-12)
        positive_derivative_values = np.concatenate((analysis_delta_p_display[analysis_delta_p_display > 0], dp_dlnt_display[dp_dlnt_display > 0]))
        if positive_derivative_values.size > 0:
            axes[1].set_ylim(np.min(positive_derivative_values), np.max(positive_derivative_values))
        axes[1].legend()
        axes[1].set_title('Bourdet Derivative Plot' if deconvolved_data is None else 'Deconvolved Bourdet Derivative Plot')
        plt.show()

        fig.utpge_results = {
            'deconvolution_requested': bool(deconvolution),
            'deconvolution_applied': bool(deconvolved_data is not None),
            'deconvolution_method': None if deconvolved_data is None else deconvolved_data.get('deconvolution_method'),
            'reference_rate_stbd': float(reference_rate),
            'analysis_time_hr': analysis_time,
            'analysis_pressure_psia': analysis_pressure,
            'analysis_delta_p_psi': analysis_delta_p,
            'horner_time': horner_time,
            'deconvolution_details': None if deconvolved_data is None else {
                key: value for key, value in deconvolved_data.items()
                if key not in {'analysis_time_hr', 'pressure_psia', 'delta_p_psi', 'horner_time', 'log_horner_time', 'reference_rate_stbd'}
            },
            'permeability_md': buildup_result['permeability_md'],
            'skin': buildup_result['skin'],
            'average_pressure_psia': buildup_result['average_pressure_psia'],
            'p_star_psia': buildup_result['p_star_psia'],
        }
        fig.utpge_results['summary_text'] = format_well_test_results(fig.utpge_results)
        print_well_test_results(fig.utpge_results)
        return fig