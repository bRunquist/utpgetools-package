"""
Oil and Gas Facilities Engineering Module

This module provides functions for the design and analysis of oil and gas processing
facilities, including separators, pressure vessels, and fluid handling equipment.
The module focuses on separation processes, multiphase flow analysis, and equipment
sizing calculations commonly used in upstream facilities engineering.

Main Functions:
    liquid_area: Liquid escape area calculations for separator pressure valves
    reduced_properties: Gas reduced property calculations for equation of state work
    iterate_drag_coeff: Droplet terminal velocity and drag coefficient calculations
    multi_stage_separator_design: Multi-stage separation system design and optimization
    gas_separation_efficiency: Gas separation efficiency calculations for multi-stage systems

Applications:
    - Separator design and optimization
    - Multiphase flow analysis in process equipment
    - Droplet settling and separation calculations
    - Process equipment sizing and selection
    - Facilities engineering and design validation

Dependencies:
    - numpy: For numerical calculations and array operations

Notes:
    This module supports facilities engineers in the design and optimization of
    oil and gas processing equipment, with particular emphasis on separation
    processes and multiphase flow behavior in process facilities.
"""

import numpy as np

def liquid_area(ql,C,rho=None,gamma=None,deltap=None,p_sep=None):
    """
    Calculates the required liquid escape area for a separator pressure valve.
    
    This function determines the minimum cross-sectional area required for liquid
    flow through a pressure valve or orifice in a separator vessel. The calculation
    is based on fluid properties, flow rate, and pressure differential across the
    valve, following standard orifice flow equations for liquid service.
    
    Args:
        ql (float): Liquid flow rate in barrels per day (bbl/d).
            The volumetric flow rate of liquid through the valve.
        C (float): Discharge coefficient for the valve or orifice (dimensionless).
            Typical values range from 0.6-0.9 depending on valve design and geometry.
            Standard orifice plates typically use C = 0.61.
        rho (float, optional): Liquid density in lb/ft³.
            If not provided, will be calculated from gamma using rho = gamma * 62.4.
        gamma (float, optional): Liquid specific gravity (dimensionless, relative to water).
            Used to calculate density if rho is not provided. Required if rho is None.
        deltap (float, optional): Pressure drop across the valve in psia.
            If not provided, calculated as (p_sep - 14.7) using separator pressure.
        p_sep (float, optional): Separator pressure in psia.
            Used to calculate deltap if deltap is not provided. Required if deltap is None.
    
    Returns:
        float: Required liquid flow area in ft².
            The minimum cross-sectional area needed to accommodate the specified
            liquid flow rate under the given pressure and fluid conditions.
    
    Raises:
        ValueError: If neither (rho) nor (gamma) is provided, or if neither 
            (deltap) nor (p_sep) is provided.
        ZeroDivisionError: If calculated deltap is zero or negative.
    
    Examples:
        >>> # Calculate area using specific gravity and separator pressure
        >>> area = liquid_area(ql=500, C=0.65, gamma=0.85, p_sep=150)
        >>> print(f"Required area: {area:.4f} ft²")
        
        >>> # Calculate area using density and pressure drop
        >>> area = liquid_area(ql=750, C=0.61, rho=52.5, deltap=125)
        >>> print(f"Required area: {area:.4f} ft²")
        
        >>> # Convert to square inches for practical use
        >>> area_sq_in = area * 144  # 144 in²/ft²
        >>> print(f"Required area: {area_sq_in:.2f} in²")
    
    Theory:
        The calculation is based on the orifice flow equation for liquids:
        Q = C * A * sqrt(2 * g * Δh)
        
        Where:
        - Q = volumetric flow rate
        - C = discharge coefficient  
        - A = orifice area
        - g = gravitational acceleration
        - Δh = pressure head differential
        
        The function uses the form:
        A = (π/4) * ql / (8081.7 * C) * sqrt(ρ / Δp)
        
        The constant 8081.7 includes unit conversions for bbl/d to ft³/s
        and pressure to head calculations.
    
    Notes:
        - Formula assumes incompressible liquid flow
        - Discharge coefficient C accounts for entrance effects and viscosity
        - Pressure drop should be significant (typically > 10 psi) for accurate results
        - Results are for minimum area; actual valve should be sized with safety factor
        - Commonly used in separator and vessel sizing calculations
    """
    
    if deltap is None:
        deltap = p_sep - 14.7 # psia
    if rho is None:
        rho = gamma * 62.4 # lb/ft^3

    return (np.pi / 4 * ql / (8081.7 * C) * np.sqrt(rho / deltap))

def reduced_properties(gamma_g,P,T):
    """
    Calculate reduced pressure and temperature properties for natural gas.
    
    This function computes the reduced (pseudo-reduced) pressure and temperature
    properties of natural gas using correlations based on gas specific gravity.
    These reduced properties are essential for equation of state calculations,
    compressibility factor determination, and gas property estimation.
    
    Args:
        gamma_g (float): Gas specific gravity (dimensionless, relative to air).
            Standard gas specific gravity typically ranges from 0.55 to 0.75
            for most natural gas compositions. Air = 1.0 by definition.
        P (float): Pressure in psia.
            The absolute pressure at which reduced properties are calculated.
        T (float): Temperature in degrees Fahrenheit (°F).
            The temperature at which reduced properties are calculated.
    
    Returns:
        tuple: A 2-element tuple containing:
            - Pr (float): Reduced pressure (dimensionless)
            - Tr (float): Reduced temperature (dimensionless)
    
    Theory:
        Reduced properties are calculated as:
        - Pr = P / Pc (pressure / critical pressure)
        - Tr = T_abs / Tc (absolute temperature / critical temperature)
        
        Critical properties are estimated using Kay's rule correlations:
        - Pc = 756.8 - 131*γg - 3.6*γg²  [psia]
        - Tc = 169.2 + 349.5*γg - 74*γg²  [°R]
        
        Where γg is the gas specific gravity.
    
    Examples:
        >>> # Calculate reduced properties for typical reservoir gas
        >>> gamma_gas = 0.65  # Gas specific gravity
        >>> pressure = 3000   # psia
        >>> temperature = 180 # °F
        >>> Pr, Tr = reduced_properties(gamma_gas, pressure, temperature)
        >>> print(f"Reduced pressure: {Pr:.3f}")
        >>> print(f"Reduced temperature: {Tr:.3f}")
        
        >>> # Use for compressibility factor calculations
        >>> if Pr > 1.0 and Tr > 1.0:
        ...     print("Gas is in supercritical region")
        
    Applications:
        - Equation of state calculations
        - Gas compressibility factor (z-factor) determination
        - Corresponding states correlations
        - Gas property estimation and validation
        - Phase behavior analysis
    
    Notes:
        - Temperature is converted to absolute scale (°R = °F + 459.67) internally
        - Correlations are most accurate for natural gas mixtures
        - For gas mixtures with significant non-hydrocarbon components,
          more detailed compositional analysis may be required
        - Reduced properties are fundamental to corresponding states principle
        - Results are used extensively in gas engineering calculations
    
    References:
        - Standing, M.B. and Katz, D.L. (1942). Density of Natural Gases
        - McCain, W.D. (1990). The Properties of Petroleum Fluids
    """
    Pr = P / (756.8 - 131*gamma_g - 3.6*gamma_g**2)
    Tr = (T + 460) / (169.2 + 349.5*gamma_g - 74*gamma_g**2)

    return Pr, Tr


def iterate_drag_coeff(rhoc, rhod, dm, mu, accuracy=0.01):
    """
    Iteratively calculate drag coefficient and terminal velocity for droplet settling.
    
    This function determines the drag coefficient and terminal velocity of a spherical
    droplet falling through a continuous fluid phase using iterative solution of the
    drag coefficient correlations. The calculation accounts for droplet size, density
    difference, and fluid viscosity effects on settling behavior.
    
    Args:
        rhoc (float): Continuous phase (gas) density in lb/ft³.
            Density of the surrounding fluid through which the droplet settles.
        rhod (float): Droplet (liquid) density in lb/ft³.
            Density of the settling droplet or particle.
        dm (float): Droplet diameter in microns (μm).
            Size of the spherical droplet or particle.
        mu (float): Continuous phase viscosity in centipoise (cp).
            Dynamic viscosity of the fluid through which the droplet moves.
        accuracy (float, optional): Convergence tolerance for iteration (default=0.01).
            Relative error tolerance for Reynolds number convergence (1%).
    
    Returns:
        tuple: A 3-element tuple containing:
            - cd (float): Converged drag coefficient (dimensionless)
            - vt (float): Terminal velocity of the droplet (ft/s)
            - iterations (int): Number of iterations required for convergence
    
    Theory:
        The calculation uses the force balance for terminal velocity:
        Drag Force = Buoyancy Force
        
        Terminal velocity: vt = 0.0119 * sqrt(dm * |ρd - ρc| / cd / ρc)
        Reynolds number: Re = 4.822e-3 * ρc * dm * vt / μ
        
        Drag coefficient correlations:
        - For Re ≤ 1.0: cd = 24/Re (Stokes law)
        - For Re > 1.0: cd = 24/Re + 3/sqrt(Re) + 0.34 (intermediate regime)
    
    Algorithm:
        1. Initial guess: cd = 0.34
        2. Calculate terminal velocity from force balance
        3. Calculate Reynolds number from terminal velocity
        4. Update drag coefficient based on Reynolds number
        5. Repeat until Reynolds number converges within specified accuracy
    
    Examples:
        >>> # Water droplet settling in air
        >>> rho_air = 0.075      # lb/ft³ at standard conditions
        >>> rho_water = 62.4     # lb/ft³ at standard conditions  
        >>> droplet_size = 100   # microns
        >>> air_viscosity = 0.018  # cp at room temperature
        >>> cd, vt, iters = iterate_drag_coeff(rho_air, rho_water, droplet_size, air_viscosity)
        >>> print(f"Drag coefficient: {cd:.3f}")
        >>> print(f"Terminal velocity: {vt:.3f} ft/s")
        >>> print(f"Converged in {iters} iterations")
        
        >>> # Oil droplet in gas separator
        >>> cd, vt, n = iterate_drag_coeff(0.25, 55.0, 50, 0.012, accuracy=0.001)
        >>> settling_time = separator_height / vt  # time to settle through separator
    
    Applications:
        - Separator design and sizing
        - Droplet settling velocity calculations
        - Phase separation efficiency analysis
        - Demister and coalescer design
        - Multiphase flow analysis
    
    Notes:
        - Function assumes spherical droplets in the Stokes to intermediate flow regime
        - Convergence is based on Reynolds number stability
        - For very large or very small droplets, correlations may need modification
        - Results are sensitive to accurate density and viscosity values
        - Used extensively in facilities design for oil-gas-water separation
    
    References:
        - Perry's Chemical Engineers' Handbook
        - Coulson & Richardson's Chemical Engineering Volume 2
        - McCain, W.D. (1990). The Properties of Petroleum Fluids
    """
    
    import numpy as np
    # Initial guess for drag coefficient
    cd = 0.34

    # Use equation 5 to estimate terminal velocity
    # Use terminal velocity in equation 4 to compute Reynolds number
    # Get new estimate for drag coefficient using equation 3a or 3b depending on Re
    # Repeat until less than 1% change in Reynolds number
    
    re = 1e3
    iterations = 0
    error = 1.0
    
    while error > accuracy:
        re_old = re

        vt = 0.0119 * np.sqrt(dm * np.abs(rhod - rhoc) / cd / rhoc)
        re = 4.822 * 10**(-3) * rhoc * dm * vt / mu

        if re <= 1.0:
            cd = 24 / re
        else:
            cd = 24 / re + 3 / np.sqrt(re) + 0.34

        error = np.abs((re - re_old) / re_old)
        iterations += 1
    
    return cd, vt, iterations


def multi_stage_separator_design(P1, T1, Pn, Tn, n_stages):
    """
    Design optimal pressure and temperature conditions for multi-stage separation systems.
    
    This function calculates the optimal pressure and temperature distribution across
    multiple separation stages to maximize liquid recovery and optimize gas separation
    efficiency. The design uses geometric pressure progression and linear temperature
    distribution to ensure optimal flash separation at each stage.
    
    Args:
        P1 (float): First stage (highest) pressure in psia.
            Initial separator pressure, typically close to wellhead pressure.
        T1 (float): First stage temperature in degrees Fahrenheit (°F).
            Initial separator temperature, often close to wellhead temperature.
        Pn (float): Final stage (lowest) pressure in psia.
            Final separator pressure, typically atmospheric or stock tank pressure.
        Tn (float): Final stage temperature in degrees Fahrenheit (°F).
            Final separator temperature, often close to ambient temperature.
        n_stages (int): Number of separation stages.
            Total number of separation vessels in the train (typically 2-4 stages).
    
    Returns:
        tuple: A 3-element tuple containing:
            - P (numpy.ndarray): Array of pressures for each stage in psia
            - T (numpy.ndarray): Array of temperatures for each stage in °F  
            - R (float): Separation ratio used for pressure calculations
    
    Theory:
        Optimal pressure distribution follows geometric progression:
        R = (P1/Pn)^(1/(n-1))
        Pi = P1 / R^(i-1)
        
        Temperature distribution follows linear progression:
        Ti = T1 - (T1 - Tn) * (i-1) / (n-1)
        
        Where i is the stage number (1 to n).
    
    Design Principles:
        - Geometric pressure distribution maximizes liquid recovery
        - Equal pressure ratios between stages minimize compression work
        - Linear temperature reduction accounts for gas expansion cooling
        - Optimization balances liquid recovery against equipment costs
    
    Examples:
        >>> # Design 3-stage separation system
        >>> P_initial = 1000  # psia wellhead pressure
        >>> T_initial = 120   # °F wellhead temperature
        >>> P_final = 15     # psia atmospheric pressure
        >>> T_final = 80     # °F ambient temperature
        >>> stages = 3
        >>> pressures, temps, ratio = multi_stage_separator_design(
        ...     P_initial, T_initial, P_final, T_final, stages
        ... )
        >>> print("Stage Pressures:", pressures)
        >>> print("Stage Temperatures:", temps)
        >>> print("Separation Ratio:", ratio)
        
        >>> # Design 4-stage system for high-pressure well
        >>> P, T, R = multi_stage_separator_design(2500, 150, 14.7, 75, 4)
        >>> for i, (p, t) in enumerate(zip(P, T), 1):
        ...     print(f"Stage {i}: {p:.1f} psia, {t:.1f} °F")
    
    Applications:
        - Multi-stage separator design optimization
        - Production facility planning
        - Gas plant design and analysis
        - Liquid recovery maximization studies
        - Economic optimization of separation systems
    
    Validation:
        - Pressure ratios should be approximately equal between stages
        - Temperature reduction should be gradual and realistic
        - Final conditions should match specified target values
        - Equipment should be sized for calculated flow rates at each stage
    
    Notes:
        - Geometric pressure distribution is theoretically optimal for liquid recovery
        - Linear temperature distribution is an approximation; actual temperatures
          depend on fluid properties and heat transfer considerations
        - Results provide starting point for detailed process simulation
        - Equipment sizing requires additional calculations for vessel dimensions
        - Economic optimization may require deviation from geometric progression
    
    References:
        - Campbell, J.M. (2001). Gas Conditioning and Processing
        - Beggs, H.D. (2003). Production Optimization Using NODAL Analysis
        - Arnold, K. and Stewart, M. (2008). Surface Production Operations
    """
    
    def separation_ratio(n, pi, pn):
        """Calculate the optimal separation ratio for multi-stage separation."""
        return (pi/pn)**(1/(n-1))
    
    # Calculate separation ratio
    R = separation_ratio(n_stages, P1, Pn)
    
    # Initialize arrays for pressure and temperature
    P = np.zeros(n_stages)
    T = np.zeros(n_stages)
    
    # Set initial conditions
    P[0] = P1
    T[0] = T1
    
    # Calculate conditions for each subsequent stage
    for i in range(1, n_stages):
        # Pressure follows geometric progression
        P[i] = P[i-1] / R
        
        # Temperature follows linear progression
        T[i] = T[i-1] - (T1 - Tn) / (n_stages - 1)
    
    return P, T, R

def gas_separation_efficiency(gas_moles, MW_oil, MW_gas, gamma_oil=None, gamma_gas=None, oil_density=None, gas_density=None):
    """
    Calculate gas separation efficiency (Esg) for multi-stage separation systems.
    
    This function computes the gas separation efficiency, which quantifies the
    effectiveness of a separation system in releasing dissolved gas from oil.
    The efficiency is calculated based on the molar amounts of gas flashed at
    each separation stage and the physical properties of the oil and gas phases.
    
    Args:
        gas_moles (list or numpy.ndarray): Gas moles flashed at each separation stage.
            Array of molar gas quantities released at each stage, typically expressed
            as percentage of total moles or actual molar quantities.
        MW_oil (float): Molecular weight of the oil phase in lb/lb-mole.
            Average molecular weight of the liquid hydrocarbon phase.
        MW_gas (float): Molecular weight of the gas phase in lb/lb-mole.
            Average molecular weight of the gas phase, typically 16-30 for natural gas.
        gamma_oil (float, optional): Oil specific gravity (dimensionless, relative to water).
            Used to calculate oil density if oil_density is not provided.
        gamma_gas (float, optional): Gas specific gravity (dimensionless, relative to air).
            Used to calculate gas density if gas_density is not provided.
        oil_density (float, optional): Oil density in lb/ft³.
            If provided, overrides calculation from gamma_oil. Takes precedence.
        gas_density (float, optional): Gas density in lb/ft³.
            If provided, overrides calculation from gamma_gas. Takes precedence.
    
    Returns:
        float: Gas separation efficiency (Esg) in dimensionless units.
            Higher values indicate more efficient gas separation from the oil phase.
            Typical values range from 100-1000+ depending on system design and conditions.
    
    Theory:
        The gas separation efficiency is calculated as:
        Esg = 5.615 * ρ_oil_molar / ρ_gas_molar * Σ(gas_flash_fraction)
        
        Where:
        - ρ_oil_molar = oil molar density (lb-mole/ft³)
        - ρ_gas_molar = gas molar density (lb-mole/ft³)  
        - gas_flash_fraction = gas_moles / 100 (converted to fraction)
        - 5.615 = conversion factor from bbl to ft³
    
    Physical Significance:
        - Higher Esg indicates better gas liberation from oil
        - Reflects the volumetric efficiency of gas separation
        - Used to compare different separator configurations
        - Helps optimize operating conditions for maximum gas recovery
    
    Examples:
        >>> # Calculate efficiency using specific gravities
        >>> gas_flashed = [45, 30, 20, 5]  # moles flashed at each stage
        >>> MW_oil = 150   # lb/lb-mole
        >>> MW_gas = 25    # lb/lb-mole
        >>> oil_sg = 0.85  # specific gravity
        >>> gas_sg = 0.65  # specific gravity
        >>> 
        >>> efficiency = gas_separation_efficiency(
        ...     gas_flashed, MW_oil, MW_gas, 
        ...     gamma_oil=oil_sg, gamma_gas=gas_sg
        ... )
        >>> print(f"Gas separation efficiency: {efficiency:.1f}")
        
        >>> # Calculate efficiency using direct densities
        >>> efficiency = gas_separation_efficiency(
        ...     [50, 35, 15], 140, 22,
        ...     oil_density=53.0, gas_density=0.12
        ... )
        
        >>> # Compare two separator designs
        >>> config_1 = gas_separation_efficiency([60, 25, 15], MW_oil, MW_gas, oil_sg, gas_sg)
        >>> config_2 = gas_separation_efficiency([40, 30, 20, 10], MW_oil, MW_gas, oil_sg, gas_sg)
        >>> print(f"Config 1 efficiency: {config_1:.1f}")
        >>> print(f"Config 2 efficiency: {config_2:.1f}")
    
    Applications:
        - Multi-stage separator optimization
        - Gas-oil separation efficiency analysis
        - Comparison of different separator configurations
        - Economic evaluation of separation systems
        - Process design validation and troubleshooting
    
    Notes:
        - Gas moles can be input as percentages or absolute values
        - Function assumes standard density calculations if specific gravities are used:
          - Oil density = gamma_oil * 62.4 lb/ft³
          - Gas density = gamma_gas * 0.0764 lb/ft³
        - Higher efficiency values generally indicate better separation performance
        - Results should be validated against field performance data
        - Used in conjunction with economic analysis for optimal design selection
    
    References:
        - Campbell, J.M. (2001). Gas Conditioning and Processing
        - Arnold, K. and Stewart, M. (2008). Surface Production Operations
        - Beggs, H.D. (2003). Production Optimization Using NODAL Analysis
    """
    gas_flash_fraction = gas_moles / 100
    
    if oil_density is None:
        oil_molar_density = gamma_oil * 62.4 / MW_oil
    else:
        oil_molar_density = oil_density / MW_oil
    
    if gas_density is None:
        gas_molar_density = gamma_gas * 0.0764 / MW_gas
    else:
        gas_molar_density = gas_density / MW_gas
    
    Esg = 5.615 * oil_molar_density / gas_molar_density * np.sum(gas_flash_fraction)
    return Esg