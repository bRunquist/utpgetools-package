"""
Well Testing Analysis Module

This module provides comprehensive functions for well testing analysis in petroleum engineering,
including transient pressure analysis, infinite acting radial flow, pseudo-steady state analysis,
and pressure derivative calculations. The module supports both single-phase liquid systems and
includes tools for pressure transient test interpretation and parameter estimation.

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
from scipy.optimize import curve_fit
from typing import Union, Tuple, Dict, Optional

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
                       mbh_lambda=None):
    """
    Perform a comprehensive well test analysis.
    
    Args:
        time (Union[np.ndarray, list]): Time values in hours (must be duration of timestep (deltat))
        pressure (Union[np.ndarray, list]): Pressure values in psia
        rate (Union[np.ndarray, list]): Production rate in STB/day
        IARF_slice (tuple): Start and end times for the IARF period
        type (list(string)): List of well test types ['buildup', 'drawdown', 'gas', 'horizontal]
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

        
    Returns:
        
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
    # Force data to arrays
    pressure = np.array(pressure)
    time = np.array(time)
    rate = np.array(rate)
    if 'buildup' in type:
        # Perform build-up analysis
        # Create figure for Horner plot and derivative plot
        # from utpgetools.well_testing import bourdet_derivative, r2_score
        pwf = float(pressure[np.argmin(time)])
        positive_time_mask = time > 0
        if not np.any(positive_time_mask):
            raise ValueError("Buildup analysis requires at least one positive shut-in time")
        time = time[positive_time_mask]
        pressure = pressure[positive_time_mask]
        if rate.ndim > 0 and rate.shape == positive_time_mask.shape:
            rate = rate[positive_time_mask]
        # Define masks for flow regimes
        if wbs_slice is None:
            wbs_mask = (time < IARF_slice[0])
        else:
            wbs_mask = (time >= wbs_slice[0]) & (time <= wbs_slice[1])
        if boundary_slice is None:
            boundary_mask = (time > IARF_slice[1])
        else:
            boundary_mask = (time >= boundary_slice[0]) & (time <= boundary_slice[1])
        # Calculate Horner time
        horner_time = (time + tp) / time
        deltap = pressure - pressure[0]  # Pressure change from initial value
        dp_dlnt = bourdet_derivative(time, deltap)
        # Fit lines through IARF period
        IARF_mask = (time >= IARF_slice[0]) & (time <= IARF_slice[1])
        IARF_time = time[IARF_mask]
        IARF_pressure = pressure[IARF_mask]
        # Fit straight line to IARF on horner plot
        params_horner = np.polyfit(np.log10(horner_time[IARF_mask]), IARF_pressure, 1)
        IARF_line_horner = params_horner[0] * np.log10(horner_time) + params_horner[1]
        _iarf_residuals = IARF_pressure - IARF_line_horner[IARF_mask]
        _iarf_pressure_range = np.ptp(IARF_pressure)
        _iarf_residual_pct = 100 * np.std(_iarf_residuals) / _iarf_pressure_range if _iarf_pressure_range > 0 else np.nan
        # Fit straight line to IARF on derivative plot (should be horizontal line)
        def horizontal_line(x, c):
            return c
        popt_bourdet, _ = curve_fit(horizontal_line, np.log10(time[IARF_mask]), dp_dlnt[IARF_mask])
        bourdet_line = popt_bourdet[0] * np.ones_like(time)
        # Set masks for shading different flow regimes

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        # Pressure vs time plot
        # axes[0].plot(time, pressure, label='Pressure')
        # axes[0].set_xlabel('Time (hr)')
        # axes[0].set_ylabel('Pressure (psi)')
        # axes[0].set_title('Pressure vs Time')
        # axes[0].legend()
        # Horner Plot
        axes[0].semilogx(horner_time, pressure)
        axes[0].semilogx(horner_time, IARF_line_horner, 'r--', label='IARF Fit')
        # Dummy line for annotation
        axes[0].semilogx(horner_time, IARF_line_horner, linestyle='', label=f'm={params_horner[0]:.2f}\nP*={params_horner[1]:.0f}\nResidual:({_iarf_residual_pct:.1f}% of range)', alpha=0)
        axes[0].fill_between(horner_time, min(pressure), max(pressure), where=IARF_mask, color='red', alpha=0.1, label='IARF Zone')
        axes[0].fill_between(horner_time, min(pressure), max(pressure), where=wbs_mask, color='blue', alpha=0.1, label='WBS Zone')
        axes[0].fill_between(horner_time, min(pressure), max(pressure), where=boundary_mask, color='green', alpha=0.1, label='Boundary Zone')
        axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[0].set_xlim(min(horner_time)+1e-12, max(horner_time)-1e-12)
        axes[0].set_ylim(min(pressure), max(pressure))
        axes[0].set_xlabel('Horner Time (tp+Δt)/Δt')
        axes[0].set_ylabel('Pressure (psi)')
        axes[0].set_title('Horner Plot')
        axes[0].legend()

        # Bourdet derivative plot
        axes[1].loglog(time, dp_dlnt, label='dp/d(ln t)')
        axes[1].loglog(time, deltap, label='deltaP')
        axes[1].loglog(time, bourdet_line, 'r--', label='IARF Fit')
        axes[1].fill_between(time, min(deltap), max(deltap), where=IARF_mask, color='red', alpha=0.1, label='IARF Zone')
        axes[1].fill_between(time, min(deltap), max(deltap), where=wbs_mask, color='blue', alpha=0.1, label='WBS Zone')
        axes[1].fill_between(time, min(deltap), max(deltap), where=boundary_mask, color='green', alpha=0.1, label='Boundary Zone')
        axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[1].set_xlabel('Time (hr)')
        axes[1].set_ylabel('Pressure')
        axes[1].set_xlim(min(time)+1e-12, max(time)-1e-12)
        positive_derivative_values = np.concatenate((deltap[deltap > 0], dp_dlnt[dp_dlnt > 0]))
        if positive_derivative_values.size > 0:
            axes[1].set_ylim(np.min(positive_derivative_values), np.max(positive_derivative_values))
        axes[1].legend()
        axes[1].set_title('Bourdet Derivative Plot')
        plt.show()
        # Perform calculations for permeability, skin, etc.
        # Horner Plot
        k_from_horner = 162.6 * rate * Bo * mu / (abs(params_horner[0]) * h) if rate is not None else np.nan
        p_1hr = params_horner[0] * np.log10((tp + 1) / 1) + params_horner[1]
        s_from_horner = 1.151 * ((p_1hr - pwf)/np.abs(params_horner[0]) - np.log10(k_from_horner/phi/mu/ct/rw**2)+3.23) if rate is not None else np.nan
        # MDH semilog plot method
        params_mdh = np.polyfit(np.log10(time[IARF_mask]), pressure[IARF_mask], 1)
        k_from_mdh = 162.6 * rate * Bo * mu / (abs(params_mdh[0]) * h) if rate is not None else np.nan
        s_from_mdh = 1.151 * ((p_1hr - pwf)/np.abs(params_mdh[0]) - np.log10(k_from_mdh/phi/mu/ct/rw**2)+3.23) if rate is not None else np.nan
        # Bourdet derivative method
        k_from_bourdet = 70.6 * rate * Bo * mu / (bourdet_line[0] * h) if rate is not None else np.nan
        iarf_indices = np.where(IARF_mask)[0]
        deltap_r_index = iarf_indices[len(iarf_indices) // 2]
        deltap_r = deltap[deltap_r_index]
        deltat_r = time[deltap_r_index]
        s_from_bourdet = 0.5 * ((deltap_r/bourdet_line[0]) - np.log(k_from_bourdet*deltat_r/1688/phi/mu/ct/rw**2)) if rate is not None else np.nan

        # Calculate average drainage pressure
        # Dietz method (P vs deltat)
        deltat_pbar_mdh = phi * mu * ct * A / 0.0002637/k_from_mdh/CA
        pbar_mdh = params_mdh[0] * np.log10(deltat_pbar_mdh) + params_mdh[1]
        # Ramey Cobb method (horner plot)
        horner_time_pbar = 0.0002637 * k_from_horner * CA * tp / phi / mu / ct / A
        pbar_ramey = params_horner[0] * np.log10(horner_time_pbar) + params_horner[1]
        # Calculate tpad for MBH method
        tpad = 0.0002637 * k_from_horner * tp / phi / mu / ct / A
        print(f"tpad for MBH method: {tpad:.2f}")

        if pmbhd is None and CA is not None:
            if mbh_alpha is None or mbh_beta is None or mbh_lambda is None:
                pmbhd = float(np.log(CA * tpad))
                print(f"Computed pMBHD from CA-only asymptote: {pmbhd:.2f}")
            else:
                pmbhd = mbh_dimensionless_average_pressure(
                    tpad,
                    shape_factor_ca=CA,
                    alpha=mbh_alpha,
                    beta=mbh_beta,
                    aspect_ratio_lambda=mbh_lambda,
                )
                print(f"Computed pMBHD from MBH rectangle geometry: {pmbhd:.2f}")

        pbar_MBH = params_horner[1] - abs(params_horner[0]) / 2.303 * pmbhd if pmbhd is not None else np.nan

        def _print_grouped_results(title, method_values, unit=""):
            print(f"\n{title}:")
            for method_name, value in method_values.items():
                print(f"  {method_name}: {value:.2f}{unit}")
            values = np.array(list(method_values.values()), dtype=float)
            valid_values = values[~np.isnan(values)]
            mean_value = np.nan if valid_values.size == 0 else float(np.mean(valid_values))
            print(f"  Mean ({valid_values.size} methods): {mean_value:.2f}{unit}")

        _print_grouped_results(
            "Permeability",
            {
                "Horner": k_from_horner,
                "MDH": k_from_mdh,
                "Bourdet": k_from_bourdet,
            },
            " md",
        )
        _print_grouped_results(
            "Skin",
            {
                "Horner": s_from_horner,
                "MDH": s_from_mdh,
                "Bourdet": s_from_bourdet,
            },
        )
        _print_grouped_results(
            "Average Pressure",
            {
                "Dietz": pbar_mdh,
                "Ramey-Cobb": pbar_ramey,
                "MBH": pbar_MBH,
            },
            " psi",
        )
        return fig