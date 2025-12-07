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




def iterate_drag_coeff(continuum_density, 
                       particle_density, 
                       continuum_visc,
                       particle_diameter,
                       error=1e-12,
                       max_iterations=1000000):
    """
    Iteratively calculate the drag coefficient (Cd) and terminal velocity (vt) for a particle in a fluid.
    
    This function uses an iterative approach to solve for the drag coefficient and terminal velocity
    of a spherical particle settling in a fluid, accounting for the non-linear dependence of drag
    coefficient on Reynolds number. The method is based on empirical correlations for drag in the
    intermediate and turbulent regimes, and is commonly used in multiphase flow and separation calculations.

    Args:
        continuum_density (float):
            Density of the continuous phase (fluid) in lb/ft³.
        particle_density (float):
            Density of the particle phase in lb/ft³.
        continuum_visc (float):
            Dynamic viscosity of the continuous phase (fluid) in cP (centipoise).
        particle_diameter (float):
            Diameter of the particle in inches. Can be a scalar or array-like.
        error (float, optional):
            Tolerance for convergence of the iterative solution.
        max_iterations (int, optional):
            Maximum number of iterations to perform.

    Returns:
        tuple or list: If particle_diameter is a scalar, returns a tuple (Cd, vt).
            If particle_diameter is array-like, returns a list of (Cd, vt) tuples for each diameter.

    Raises:
        ValueError: If densities are equal (division by zero in drag calculation).

    Examples:
        >>> # Scalar input
        >>> Cd, vt = iterate_drag_coeff(
        ...     continuum_density=55.0, particle_density=62.4, continuum_visc=1.2, particle_diameter=0.01
        ... )
        >>> print(f"Drag coefficient: {Cd:.3f}")
        >>> print(f"Terminal velocity: {vt:.4f} ft/s")

        >>> # Array input
        >>> results = iterate_drag_coeff(
        ...     continuum_density=55.0, particle_density=62.4, continuum_visc=1.2, particle_diameter=[0.01, 0.02]
        ... )
        >>> for Cd, vt in results:
        ...     print(f"Cd: {Cd:.3f}, vt: {vt:.4f} ft/s")

    Notes:
        - The function uses an empirical correlation for drag coefficient:
          Cd = 24/Re + 3/sqrt(Re) + 0.34 (for Re > 1), else Cd = 24/Re
        - Iteration continues until the change in terminal velocity is less than 1e-6 ft/s
        - Useful for droplet settling, particle separation, and multiphase flow analysis
        - All units must be consistent (lb/ft³, cP, inches)
    """
    # Handle both scalar and array-like input for particle_diameter
    diam_array = particle_diameter
    is_arraylike = isinstance(diam_array, (list, tuple, np.ndarray)) and not isinstance(diam_array, (str, bytes))
    if not is_arraylike:
        diam_array = [particle_diameter]
    Cd_list = []
    vt_list = []
    for drop_size in diam_array:
        if continuum_density == particle_density:
            raise ValueError("Continuous and particle densities must not be equal.")
        initial_k = np.sqrt(0.34*continuum_density/np.abs(continuum_density-particle_density))
        error_squared = 1.0
        vt = 0.0119/initial_k*np.sqrt(drop_size)
        iteration = 0
        while error_squared > error and iteration < max_iterations:
            Re = 4.882*10**-3*continuum_density*drop_size*vt/continuum_visc
            new_Cd = 24/Re + 3/np.sqrt(Re) + 0.34 if Re > 1 else 24/Re
            new_k = np.sqrt(new_Cd*continuum_density/np.abs(continuum_density-particle_density))
            new_vt = 0.0119/new_k*np.sqrt(drop_size)
            error_squared = (new_vt - vt)**2
            vt = new_vt
            iteration += 1
        Cd_list.append(new_Cd)
        vt_list.append(vt)
    if is_arraylike:
        return Cd_list, vt_list
    else:
        return Cd_list[0], vt_list[0]


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

def calculate_compressor_stage_hp(qg,
                                  ps,
                                  Ts,
                                  ec,
                                  C,
                                  gamma_g,
                                  co2_percent,
                                  n2_percent,
                                  h2s_percent,
                                  h2o_percent,
                                  deltacp=None,
                                  pd=None,
                                  R=None,
                                  return_all_vals=None,
                                  Tpc_override=None,
                                  Ppc_override=None,
                                  component=None):
    """
    Calculate the horsepower required for a single compressor stage in gas compression.
    
    This function computes the theoretical horsepower needed to compress natural gas
    from suction to discharge conditions. It handles both z-factor calculations at
    suction and discharge conditions, with robust error handling and fallback methods.
    
    Args:
        qg (float): Gas flow rate at standard conditions in MMSCF/D.
        ps (float): Suction pressure at compressor inlet in psia.
        Ts (float): Suction temperature at compressor inlet in degrees Fahrenheit (°F).
        ec (float): Compressor mechanical efficiency as decimal (0.80-0.95).
        C (float): Clearance coefficient (0.03-0.10), typically 0.05.
        gamma_g (float): Gas specific gravity (relative to air, typically 0.55-0.75).
        co2_percent (float): Carbon dioxide mole percentage in gas.
        n2_percent (float): Nitrogen mole percentage in gas.
        h2s_percent (float): Hydrogen sulfide mole percentage in gas.
        h2o_percent (float): Water vapor mole percentage in gas.
        deltacp (float, optional): Heat capacity correction factor (required).
        pd (float, optional): Discharge pressure in psia (either pd or R required).
        R (float, optional): Compression ratio (either pd or R required).
        return_all_vals (bool, optional): If True, returns additional parameters.
        Tpc_override (float, optional): Override critical temperature in °R.
        Ppc_override (float, optional): Override critical pressure in psia.
        component (arraylike, optional): Component fractions [N2, CO2, H2S, C1-C7+].
    
    Returns:
        float or tuple: Horsepower in HP, or tuple of all calculated values if return_all_vals=True.
    
    Raises:
        ValueError: If required parameters missing or calculations fail.
    """
    
    # ===== INPUT VALIDATION AND PREPROCESSING =====
        
    if pd is None:
        if R is None:
            raise ValueError("Either pd or R must be provided.")
        pd = R * ps
    
    # Convert temperature to absolute scale
    Ts = Ts + 459.67  # Convert °F to °R
    
    # ===== CRITICAL PROPERTIES CALCULATION =====
    # Component property arrays for compositional analysis

    # Values from Natural Gas Engineering Handbook
    critical_pressures = [493, 1071, 1306, 668, 708, 616, 529, 551, 490, 489, 437, 332]
    critical_temperatures = [227, 548, 672, 343, 550, 666, 735, 765, 829, 845, 913, 1070]
    cp_values = [6.96171216, 8.77344105, 1.1765*29*.238, 8.45882846, 12.33516566, 
                 17.13558525, 22.53395584, 22.50485724, 27.64697513, 28.05152977, 
                 33.34002168, 49.3124352]

    # Determine critical properties calculation method
    if component is not None:
        # Use detailed composition if provided
        Tpc = np.sum(np.array(critical_temperatures) * np.array(component))
        Ppc = np.sum(np.array(critical_pressures) * np.array(component))
        cpst = np.sum(np.array(cp_values) * np.array(component))
        co2 = component[1]
        h2s = component[2]
    elif Tpc_override is not None and Ppc_override is not None:
        # Use manually specified critical properties
        Tpc = Tpc_override
        Ppc = Ppc_override
        print(f"Using override critical properties: Tpc = {Tpc:.1f} °R, Ppc = {Ppc:.1f} psia")
    else:
        # Calculate using gas specific gravity correlations
        Tpc = 169.2 + 349.5*gamma_g - 74*gamma_g**2
        Ppc = 756.8 - 131*gamma_g - 3.6*gamma_g**2
        co2 = co2_percent / 100
        h2s = h2s_percent / 100
        print(f"Using calculated critical properties: Tpc = {Tpc:.1f} °R, Ppc = {Ppc:.1f} psia")
    
    # Apply Wichert-Aziz correction for sour gas components
    correction_factor = 120 * ((co2 + h2s)**0.9 + (co2 + h2s)**1.6) + (h2s**0.5 + h2s**4)
    Tpc_corr = Tpc - correction_factor
    Ppc_corr = (Ppc * Tpc_corr) / (Tpc + h2s * (1-h2s) * correction_factor)
    Tpc = Tpc_corr
    Ppc = Ppc_corr
    
    # Calculate pseudo-reduced properties for diagnostics
    Tr = Ts / Tpc
    Pr = ps / Ppc
    print(f"Corrected Tr: {Tr:.3f}, Corrected Pr: {Pr:.3f}")

    # ===== Z-FACTOR CALCULATION METHODS =====
    from utpgetools.utilities_package import gas_properties_calculation
    
    def calculate_z_factor_standing_katz(pressure, temperature_R, Tpc, Ppc):
        """
        Calculate z-factor using Standing-Katz iterative method as fallback.
        
        This method provides a robust fallback when primary calculations fail,
        using the classical Standing-Katz correlation with Newton-Raphson iteration.
        """
        Tr = temperature_R / Tpc
        Pr = pressure / Ppc
        
        # Initialize iteration parameters
        rho_r = 0.1  # Initial guess for reduced density
        max_iterations = 1000
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            # Standing-Katz equation for compressibility factor
            # Dranchuk, P.M. and Abou-Kassem, J.H.: "Calculations of z-Factors for Natural Gases Using Equations of State," J. Cdn. Pet. Tech. (July-Sept. 1975) 34-36.
            Z1 = (1 + (0.3265 - 1.0700/Tr - 0.5339/Tr**3 + 0.01569/Tr**4 - 0.05165/Tr**5) * rho_r
                  + (0.5475 - 0.7361/Tr + 0.1844/Tr**2) * rho_r**2
                  - 0.1056 * (-0.7361/Tr + 0.1844/Tr**2) * rho_r**5
                  + 0.6134 * (1+0.7210*rho_r**2) * (rho_r**2/Tr**3) * np.exp(-0.7210*rho_r**2))
            
            # Equation of state relationship
            Z2 = 0.27 * Pr / rho_r / Tr
            
            # Check convergence based on z-factor precision
            if iteration > 0:
                if abs(Z1 - z_last) < 0.001:  # First 3 digits unchanged
                    break
            z_last = Z1
            
            # Newton-Raphson iteration for improved convergence
            error = Z1 - Z2
            if abs(error) < tolerance:
                break
                
            # Calculate derivative numerically
            drho = 1e-6
            rho_r_plus = rho_r + drho
            Z1_plus = (1 + (0.3265 - 1.0700/Tr - 0.5339/Tr**3 + 0.01569/Tr**4 - 0.05165/Tr**5) * rho_r_plus
                       + (0.5475 - 0.7361/Tr + 0.1844/Tr**2) * rho_r_plus**2
                       - 0.1056 * (-0.7361/Tr + 0.1844/Tr**2) * rho_r_plus**5
                       + 0.6134 * (1+0.7210*rho_r_plus**2) * (rho_r_plus**2/Tr**3) * np.exp(-0.7210*rho_r_plus**2))
            Z2_plus = 0.27 * Pr / rho_r_plus / Tr
            error_plus = Z1_plus - Z2_plus
            
            derror_drho = (error_plus - error) / drho
            
            # Update reduced density with convergence safeguards
            if abs(derror_drho) > 1e-12:
                rho_r = rho_r - error / derror_drho
            else:
                rho_r = rho_r * 1.1  # Simple adjustment if derivative is too small
                
            rho_r = max(rho_r, 0.001)  # Ensure positive values
        
        return Z1

    # ===== Z1 CALCULATION (SUCTION CONDITIONS) =====
    # Primary method: Use utilities package for enhanced accuracy
    try:
        properties = gas_properties_calculation(gravity=gamma_g,
                                                co2_percent=co2_percent,
                                                n2_percent=n2_percent,
                                                h2s_percent=h2s_percent,
                                                h2o_percent=h2o_percent,
                                                pressure_psi=ps,
                                                temperature_f=Ts - 459.67)
        z1 = properties['z_factors'][-1]
        
        # Validate result quality
        if np.isnan(z1) or np.iscomplexobj(z1):
            raise ValueError(f"Primary z1 calculation returned invalid result: {z1}")
            
        print(f"Using primary z1-factor calculation (suction): {z1:.4f}")
        
    except (TypeError, ValueError) as e:
        # Fallback method: Standing-Katz correlation
        if "complex" in str(e).lower() or "invalid result" in str(e).lower():
            print("Primary z1-factor calculation failed, using Standing-Katz fallback method...")
            z1 = calculate_z_factor_standing_katz(ps, Ts, Tpc, Ppc)
            
            # Validate fallback result
            if np.isnan(z1) or np.iscomplexobj(z1):
                raise ValueError(f"Both primary and Standing-Katz z1 calculations failed. "
                               f"Gas specific gravity ({gamma_g:.3f}) may be outside valid range. "
                               f"Typical natural gas range: 0.55-0.75")
            
            print(f"Standing-Katz z1-factor (suction): {z1:.4f}")
        else:
            raise e

    # ===== HEAT CAPACITY AND ISENTROPIC EXPONENT CALCULATION =====
    # Calculate standard heat capacity if component analysis not used
    if component is None:
        gamma_array = np.array([0.6, 0.7, 0.8, 0.9])
        cpst_array = np.array([29*0.6*(3.89e-4*Ts + 0.4872),
                              29*0.7*(4.17e-4*Ts + 0.4698),
                              29*0.8*(4.44e-4*Ts + 0.445),
                              29*0.9*(5.0e-4*Ts + 0.4218)])
        # Linear interpolation for intermediate specific gravities
        coeffs = np.polyfit(gamma_array, cpst_array, 1)
        cpst = np.polyval(coeffs, gamma_g)
    
    # Apply heat capacity correction (addition method confirmed correct)
    if deltacp is None:
        print("Please provide cp correction factor")
        return
    cp = cpst + deltacp
    print(f"cpst = {cpst:.4f} Btu/lb-R, cp corrected = {cp:.4f} Btu/lb-R")
    
    # Calculate isentropic exponent
    k = cp / (cp - 1.986)  # 1.986 = universal gas constant in Btu/lb-mole-R / molecular weight

    # ===== Z2 CALCULATION (DISCHARGE CONDITIONS) =====
    # Calculate isentropic discharge temperature
    Td = Ts * (pd/ps)**((k-1)/k)
    
    # Primary method for z2
    try:
        properties = gas_properties_calculation(gravity=gamma_g,
                                                co2_percent=co2_percent,
                                                n2_percent=n2_percent,
                                                h2s_percent=h2s_percent,
                                                h2o_percent=h2o_percent,
                                                pressure_psi=pd,
                                                temperature_f=Td - 459.67)
        z2 = properties['z_factors'][-1]
        
        # Validate result quality
        if np.isnan(z2) or np.iscomplexobj(z2):
            raise ValueError(f"Primary z2 calculation returned invalid result: {z2}")
            
        print(f"Using primary z2-factor calculation (discharge): {z2:.4f}")
        
    except (TypeError, ValueError) as e:
        # Fallback method for discharge conditions
        if "complex" in str(e).lower() or "invalid result" in str(e).lower():
            print("Primary z2-factor calculation failed, using Standing-Katz fallback method...")
            z2 = calculate_z_factor_standing_katz(pd, Td, Tpc, Ppc)
            
            # Validate fallback result
            if np.isnan(z2) or np.iscomplexobj(z2):
                raise ValueError(f"Both primary and Standing-Katz z2 calculations failed. "
                               f"Discharge conditions: P={pd:.1f} psia, T={Td-459.67:.1f}°F. "
                               f"Gas specific gravity ({gamma_g:.3f}) may be outside valid range. "
                               f"Consider using more realistic gas properties or different correlations.")
            
            print(f"Standing-Katz z2-factor (discharge): {z2:.4f}")
        else:
            raise e

    # ===== VOLUMETRIC EFFICIENCY CALCULATION =====
    # Calculate compression ratio if not already defined
    if R is None:
        R = pd / ps
    
    # Updated volumetric efficiency equation accounting for z-factor effects
    ev = 1 - 0.05 - R/100 - C * (R**(1/k) * (z2/z1) - 1)

    # ===== HORSEPOWER CALCULATION =====
    # Calculate theoretical horsepower using gas property relationships
    P = 0.08584 * (k/(k-1)) * Ts * ((pd/ps)**(z1*(k-1)/k) - 1) * qg / ec / ev

    # ===== FINAL RESULT VALIDATION =====
    # Print final reduced properties for deltacp correction factor reference
    print(f"Final reduced properties for deltacp reference: Tr = {Tr:.3f}, Pr = {Pr:.3f}")
    
    # Ensure all results are physically meaningful
    if np.isnan(P) or np.iscomplexobj(P):
        raise ValueError(f"Horsepower calculation resulted in invalid value: {P}. "
                        f"Check input parameters: gamma_g={gamma_g:.3f}, deltacp={deltacp}")
    
    if np.isnan(ev) or np.iscomplexobj(ev):
        raise ValueError(f"Volumetric efficiency calculation resulted in invalid value: {ev}. "
                        f"z1={z1:.4f}, z2={z2:.4f}, R={R:.2f}")
    
    # Return results based on user preference
    if return_all_vals is None:
        return P
    else:
        return P, ev, pd, R, k, z1


def standing_katz(gamma_g, pressure, temperature_f, component=None, co2_percent=0, h2s_percent=0, 
                  n2_percent=0, h2o_percent=0, Tpc_override=None, Ppc_override=None, deltacp=None):
    """
    Calculate z-factor using Standing-Katz iterative method with component-based critical properties.
    
    This function provides a standalone implementation of the Standing-Katz correlation
    for calculating gas compressibility factors. It includes component-based critical
    property calculations using Kay's rule or correlations based on gas specific gravity.
    
    Args:
        gamma_g (float): Gas specific gravity (air = 1.0)
        pressure (float): Pressure in psia
        temperature_f (float): Temperature in °F
        component (list, optional): Gas composition as mole fractions for 12 components:
            [C1, CO2, H2S, N2, C2, C3, iC4, nC4, iC5, nC5, C6, C7+]
        co2_percent (float): CO2 content in mol% (0-100)
        h2s_percent (float): H2S content in mol% (0-100)
        n2_percent (float): N2 content in mol% (0-100)
        h2o_percent (float): H2O content in mol% (0-100)
        Tpc_override (float, optional): Override pseudo-critical temperature in °R
        Ppc_override (float, optional): Override pseudo-critical pressure in psia
        deltacp (float, optional): Heat capacity correction factor. If None, will prompt for user input
        
    Returns:
        dict: Dictionary containing:
            - 'z_factor': Calculated compressibility factor
            - 'Tpc': Pseudo-critical temperature in °R
            - 'Ppc': Pseudo-critical pressure in psia
            - 'Tr': Reduced temperature
            - 'Pr': Reduced pressure
            
    References:
        - Standing, M.B. and Katz, D.L. (1942). Density of Natural Gases
        - Dranchuk, P.M. and Abou-Kassem, J.H. (1975). Calculations of z-Factors for Natural Gases Using Equations of State
        - Katz, D.L. and McGraw-Hill (1959). Handbook of Natural Gas Engineering
    """
    
    # Convert temperature to absolute scale
    temperature_R = temperature_f + 459.67  # Convert °F to °R
    
    # ===== CRITICAL PROPERTIES CALCULATION =====
    # Component property arrays for compositional analysis
    # Values from Natural Gas Engineering Handbook
    critical_pressures = [493, 1071, 1306, 668, 708, 616, 529, 551, 490, 489, 437, 332]
    critical_temperatures = [227, 548, 672, 343, 550, 666, 735, 765, 829, 845, 913, 1070]
    cp_values = [6.96171216, 8.77344105, 1.1765*29*.238, 8.45882846, 12.33516566, 
                 17.13558525, 22.53395584, 22.50485724, 27.64697513, 28.05152977, 
                 33.34002168, 49.3124352]

    # Determine critical properties calculation method
    if component is not None:
        # Use detailed composition if provided
        Tpc = np.sum(np.array(critical_temperatures) * np.array(component))
        Ppc = np.sum(np.array(critical_pressures) * np.array(component))
        cpst = np.sum(np.array(cp_values) * np.array(component))
        co2 = component[1]
        h2s = component[2]
    elif Tpc_override is not None and Ppc_override is not None:
        # Use manually specified critical properties
        Tpc = Tpc_override
        Ppc = Ppc_override
        co2 = co2_percent / 100
        h2s = h2s_percent / 100
    else:
        # Calculate using gas specific gravity correlations
        Tpc = 169.2 + 349.5*gamma_g - 74*gamma_g**2
        Ppc = 756.8 - 131*gamma_g - 3.6*gamma_g**2
        co2 = co2_percent / 100
        h2s = h2s_percent / 100
    
    # Apply Wichert-Aziz correction for sour gas components
    correction_factor = 120 * ((co2 + h2s)**0.9 + (co2 + h2s)**1.6) + (h2s**0.5 + h2s**4)
    Tpc_corr = Tpc - correction_factor
    Ppc_corr = (Ppc * Tpc_corr) / (Tpc + h2s * (1-h2s) * correction_factor)
    Tpc = Tpc_corr
    Ppc = Ppc_corr
    
    # Calculate pseudo-reduced properties
    Tr = temperature_R / Tpc
    Pr = pressure / Ppc
    
    # Handle deltacp input
    if deltacp is None:
        # Print reduced properties for deltacp determination and raise error
        print(f"Corrected Tr: {Tr:.3f}, Corrected Pr: {Pr:.3f}")
        raise ValueError("deltacp is required. Please determine deltacp from the reduced properties chart and re-run with deltacp parameter.")
    
    # ===== STANDING-KATZ ITERATIVE CALCULATION =====
    # Initialize iteration parameters
    rho_r = 0.1  # Initial guess for reduced density
    max_iterations = 1000
    tolerance = 1e-6
    
    for iteration in range(max_iterations):
        # Standing-Katz equation for compressibility factor
        # Dranchuk, P.M. and Abou-Kassem, J.H.: "Calculations of z-Factors for Natural Gases Using Equations of State," J. Cdn. Pet. Tech. (July-Sept. 1975) 34-36.
        Z1 = (1 + (0.3265 - 1.0700/Tr - 0.5339/Tr**3 + 0.01569/Tr**4 - 0.05165/Tr**5) * rho_r
              + (0.5475 - 0.7361/Tr + 0.1844/Tr**2) * rho_r**2
              - 0.1056 * (-0.7361/Tr + 0.1844/Tr**2) * rho_r**5
              + 0.6134 * (1+0.7210*rho_r**2) * (rho_r**2/Tr**3) * np.exp(-0.7210*rho_r**2))
        
        # Equation of state relationship
        Z2 = 0.27 * Pr / rho_r / Tr
        
        # Check convergence based on z-factor precision
        if iteration > 0:
            if abs(Z1 - z_last) < 0.001:  # First 3 digits unchanged
                break
        z_last = Z1
        
        # Newton-Raphson iteration for improved convergence
        error = Z1 - Z2
        if abs(error) < tolerance:
            break
            
        # Calculate derivative numerically
        drho = 1e-6
        rho_r_plus = rho_r + drho
        Z1_plus = (1 + (0.3265 - 1.0700/Tr - 0.5339/Tr**3 + 0.01569/Tr**4 - 0.05165/Tr**5) * rho_r_plus
                   + (0.5475 - 0.7361/Tr + 0.1844/Tr**2) * rho_r_plus**2
                   - 0.1056 * (-0.7361/Tr + 0.1844/Tr**2) * rho_r_plus**5
                   + 0.6134 * (1+0.7210*rho_r_plus**2) * (rho_r_plus**2/Tr**3) * np.exp(-0.7210*rho_r_plus**2))
        Z2_plus = 0.27 * Pr / rho_r_plus / Tr
        error_plus = Z1_plus - Z2_plus
        
        derror_drho = (error_plus - error) / drho
        
        # Update reduced density with convergence safeguards
        if abs(derror_drho) > 1e-12:
            rho_r = rho_r - error / derror_drho
        else:
            rho_r = rho_r * 1.1  # Simple adjustment if derivative is too small
            
        rho_r = max(rho_r, 0.001)  # Ensure positive values
    
    # Validate result
    if np.isnan(Z1) or np.iscomplexobj(Z1):
        raise ValueError(f"Standing-Katz calculation failed to converge for given conditions")
    
    # ===== HEAT CAPACITY AND ISENTROPIC EXPONENT CALCULATION =====
    # Calculate standard heat capacity if component analysis not used
    if component is None:
        gamma_array = np.array([0.6, 0.7, 0.8, 0.9])
        cpst_array = np.array([29*0.6*(3.89e-4*temperature_R + 0.4872),
                               29*0.7*(3.89e-4*temperature_R + 0.4872),
                               29*0.8*(3.89e-4*temperature_R + 0.4872),
                               29*0.9*(3.89e-4*temperature_R + 0.4872)])
        cpst = np.interp(gamma_g, gamma_array, cpst_array)
    
    # Apply deltacp correction
    cp = cpst + deltacp
    cv = cp - 1.987  # Universal gas constant in Btu/lbmol·°R
    k = cp / cv  # Isentropic exponent
    
    return {
        'z_factor': Z1,
        'Tpc': Tpc,
        'Ppc': Ppc,
        'Tr': Tr,
        'Pr': Pr,
        'deltacp': deltacp,
        'cp': cp,
        'cv': cv,
        'k': k,
        'iterations': iteration + 1
    }


def gas_properties(temperature_f, pressure_psi, composition_fractions, z_factor):
    """
    Calculate gas density and viscosity for natural gas mixtures.
    
    This function calculates both gas density and viscosity using industry-standard
    correlations. Gas density is calculated using the real gas equation of state
    with z-factor correction, while viscosity is determined using the Lee-Gonzalez-Eakin
    correlation which is widely accepted in petroleum engineering applications.
    
    References:
        - Lee, A.L., Gonzalez, M.H., and Eakin, B.E.: "The Viscosity of Natural Gases," 
          JPT (August 1966) 997-1000; Trans., AIME, 237.
        - McCain, W.D.: "The Properties of Petroleum Fluids" 2nd Ed. (1990), Ch. 7
        - Ahmed, T.: "Reservoir Engineering Handbook" 5th Ed. (2019), Ch. 1
        - Whitson & Brule: "Phase Behavior" (2000), Ch. 3
    
    Args:
        temperature_f (float): Temperature in °F
        pressure_psi (float): Pressure in psia
        composition_fractions (list): Mole fractions for components in order:
            [N2, CO2, H2S, C1, C2, C3, iC4, nC4, iC5, nC5, C6, C7+]
        z_factor (float): Gas compressibility factor (from PVT correlations)
        
    Returns:
        dict: Dictionary containing gas properties:
            - 'density_lb_ft3': Gas density in lb/ft³
            - 'viscosity_cp': Gas viscosity in cp
            - 'molecular_weight': Apparent molecular weight in g/mol
            - 'specific_gravity': Gas specific gravity (air = 1.0)
            
    Examples:
        >>> composition = [0.01, 0.05, 0.0, 0.85, 0.07, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        >>> z = 0.85
        >>> props = gas_properties(80, 1000, composition, z)
        >>> print(f"Density: {props['density_lb_ft3']:.3f} lb/ft³")
        >>> print(f"Viscosity: {props['viscosity_cp']:.6f} cp")
        
    Notes:
        - The Lee-Gonzalez-Eakin correlation is valid for natural gas mixtures
          with specific gravities from 0.55 to 1.5 and temperatures from 100-340°F
        - Gas density calculation uses the real gas equation: ρ = PM/(zRT)
        - Component molecular weights are standard values from petroleum engineering literature
        - Function includes detailed console output for debugging and verification
    """
    import math
    
    # Convert temperature to Rankine
    T_R = temperature_f + 459.67  # °R
    
    # Component molecular weights (well-established values from NIST/API databases)
    molecular_weights = {
        'N2': 28.014, 'CO2': 44.010, 'H2S': 34.082, 'C1': 16.043,
        'C2': 30.070, 'C3': 44.097, 'iC4': 58.124, 'nC4': 58.124,
        'iC5': 72.151, 'nC5': 72.151, 'C6': 86.178, 'C7+': 100.20
    }
    
    components = ['N2', 'CO2', 'H2S', 'C1', 'C2', 'C3', 'iC4', 'nC4', 'iC5', 'nC5', 'C6', 'C7+']
    
    # Calculate apparent molecular weight of gas mixture
    Ma = sum(xi * molecular_weights[comp] for xi, comp in zip(composition_fractions, components))
    
    # Calculate specific gravity of gas (relative to air, MW = 28.97)
    gamma_g = Ma / 28.97
    
    # Calculate gas density using real gas equation of state
    # ρ = PM/(zRT) where R = 10.73 psia·ft³/(lb-mol·°R)
    rho_g = pressure_psi * Ma / (10.73 * T_R * z_factor)  # lb/ft³
    
    # Lee-Gonzalez-Eakin correlation for viscosity
    # Select correlation constants based on gas specific gravity
    if gamma_g <= 0.681:
        # Light gas mixture correlation constants
        A = (9.379 + 0.01607 * Ma) * T_R**1.5 / (209.2 + 19.26 * Ma + T_R)
        B = 3.448 + (986.4 / T_R) + 0.01009 * Ma
        C = 2.447 - 0.2224 * B
    else:
        # Heavier gas mixture correlation constants
        A = (8.188 - 0.6579 * gamma_g) * T_R**1.5 / (107.2 + 519.4 * gamma_g + T_R)
        B = 3.49 + (1.672 / gamma_g**2) * (1944.0 / T_R)
        C = 1.2 + 0.00245 * (T_R - 459.67)
    
    # Calculate viscosity in micropoises, then convert to centipoise
    mu_micropoise = A * math.exp(B * (rho_g / 62.4)**C)
    mu_cp = mu_micropoise / 10000  # Convert μP to cp
    
    # Print detailed calculation results for verification
    print(f"Gas Properties Calculation Results:")
    print(f"  Temperature: {temperature_f:.1f} °F ({T_R:.1f} °R)")
    print(f"  Pressure: {pressure_psi:.1f} psia")
    print(f"  Z-factor: {z_factor:.4f}")
    print(f"  Molecular weight: {Ma:.2f} g/mol")
    print(f"  Specific gravity: {gamma_g:.4f}")
    print(f"  Density: {rho_g:.3f} lb/ft³")
    print(f"  Lee-Gonzalez-Eakin coefficients:")
    print(f"    A = {A:.6f}")
    print(f"    B = {B:.4f}") 
    print(f"    C = {C:.4f}")
    print(f"  Viscosity: {mu_cp:.6f} cp")
    
    return {
        'density_lb_ft3': rho_g,
        'viscosity_cp': mu_cp,
        'molecular_weight': Ma,
        'specific_gravity': gamma_g
    }


def hall_yarborough_z_factor(reduced_pressure, reduced_temperature, tolerance=1e-6, max_iterations=100):
    """
    Calculate gas compressibility factor using the Hall-Yarborough correlation.
    
    The Hall-Yarborough correlation is one of the most accurate methods for calculating
    gas compressibility factors for natural gases. It's based on the Starling-Carnahan-
    Desantis equation of state and is particularly accurate at high pressures and 
    temperatures typical in natural gas applications.
    
    References:
        - Hall, K.R. and Yarborough, L.: "A New Equation of State for Z-factor Calculations,"
          Oil & Gas Journal, June 18, 1973, pp. 82-92
        - McCain, W.D.: "The Properties of Petroleum Fluids" 2nd Ed. (1990), Ch. 3
        - Ahmed, T.: "Reservoir Engineering Handbook" 5th Ed. (2019), Ch. 1
        - Standing, M.B.: "Volumetric and Phase Behavior of Oil Field Hydrocarbon Systems" (1977)
    
    Args:
        reduced_pressure (float): Reduced pressure (P/Pc), dimensionless
        reduced_temperature (float): Reduced temperature (T/Tc), dimensionless  
        tolerance (float, optional): Convergence tolerance for iteration. Default 1e-6
        max_iterations (int, optional): Maximum number of iterations. Default 100
        
    Returns:
        dict: Dictionary containing calculation results:
            - 'z_factor': Gas compressibility factor
            - 'reduced_density': Reduced density (ρr)
            - 'iterations': Number of iterations required for convergence
            - 'converged': Boolean indicating if solution converged
            
    Raises:
        ValueError: If reduced pressure or temperature are outside valid ranges
        RuntimeError: If iteration fails to converge within max_iterations
        
    Examples:
        >>> # Calculate z-factor for natural gas at 2000 psia, 150°F
        >>> # Assuming Pc = 667 psia, Tc = 380°R (typical natural gas)
        >>> Pr = 2000 / 667  # ≈ 3.0
        >>> Tr = (150 + 459.67) / 380  # ≈ 1.6
        >>> result = hall_yarborough_z_factor(Pr, Tr)
        >>> print(f"Z-factor: {result['z_factor']:.4f}")
        
        >>> # High pressure gas
        >>> result = hall_yarborough_z_factor(8.0, 1.5)
        >>> if result['converged']:
        >>>     print(f"Z = {result['z_factor']:.4f} in {result['iterations']} iterations")
        
    Notes:
        - Valid range: 0.2 ≤ Pr ≤ 30, 1.0 ≤ Tr ≤ 3.0
        - Accuracy: ±0.5% for most natural gas compositions
        - Uses Newton-Raphson iteration to solve implicit equation
        - More accurate than Dranchuk-Abu-Kassem at very high pressures
        - Particularly suitable for gas injection and high-pressure reservoir applications
    """
    import math
    
    # Validate input ranges
    if reduced_pressure < 0.2 or reduced_pressure > 30:
        raise ValueError(f"Reduced pressure {reduced_pressure:.3f} outside valid range [0.2, 30]")
    if reduced_temperature < 1.0 or reduced_temperature > 3.0:
        raise ValueError(f"Reduced temperature {reduced_temperature:.3f} outside valid range [1.0, 3.0]")
    
    Pr = reduced_pressure
    Tr = reduced_temperature
    
    # Hall-Yarborough correlation constants
    t = 1.0 / Tr
    
    # Correlation coefficients
    A = 0.06125 * t * math.exp(-1.2 * (1 - t)**2)
    B = t * (14.76 - 9.76 * t + 4.58 * t**2)
    C = t * (90.7 - 242.2 * t + 42.4 * t**2)
    D = 2.18 + 2.82 * t
    
    # Initial guess for reduced density using ideal gas approximation
    y = 0.001  # Start with low density guess
    
    iteration = 0
    converged = False
    
    print(f"Hall-Yarborough Z-factor Calculation:")
    print(f"  Pr = {Pr:.4f}, Tr = {Tr:.4f}")
    print(f"  Coefficients: A={A:.6f}, B={B:.4f}, C={C:.4f}, D={D:.4f}")
    
    while iteration < max_iterations:
        # Calculate function F(y) and its derivative F'(y)
        # F(y) = -A*Pr + (y + y^2 + y^3 - y^4)/(1-y)^3 - B*y^2 - C*y^D
        
        if y >= 1.0:
            # Prevent division by zero and ensure physical solution
            y = 0.99
        
        denominator = (1 - y)**3
        if abs(denominator) < 1e-15:
            y = 0.99
            denominator = (1 - y)**3
            
        # Function value
        F = (-A * Pr + 
             (y + y**2 + y**3 - y**4) / denominator - 
             B * y**2 - 
             C * y**D)
        
        # Derivative calculation
        numerator_deriv = (1 + 2*y + 3*y**2 - 4*y**3) * (1-y)**3 + 3*(1-y)**2 * (y + y**2 + y**3 - y**4)
        dF_dy = (numerator_deriv / ((1-y)**6) - 
                 2 * B * y - 
                 C * D * y**(D-1))
        
        # Newton-Raphson update
        if abs(dF_dy) < 1e-15:
            raise RuntimeError("Derivative became zero - cannot continue iteration")
            
        y_new = y - F / dF_dy
        
        # Ensure physical bounds
        y_new = max(0.001, min(0.99, y_new))
        
        # Check convergence
        if abs(y_new - y) < tolerance:
            converged = True
            y = y_new
            break
            
        y = y_new
        iteration += 1
        
        if iteration <= 5 or iteration % 10 == 0:
            print(f"  Iteration {iteration}: y = {y:.6f}, F = {F:.8f}")
    
    if not converged:
        raise RuntimeError(f"Hall-Yarborough iteration failed to converge after {max_iterations} iterations")
    
    # Calculate final z-factor
    z_factor = A * Pr / y
    
    print(f"  Converged in {iteration} iterations")
    print(f"  Final reduced density: {y:.6f}")
    print(f"  Z-factor: {z_factor:.6f}")
    
    return {
        'z_factor': z_factor,
        'reduced_density': y,
        'iterations': iteration,
        'converged': converged
    }


def hall_yarborough(gamma_g, pressure, temperature, composition_fractions, 
                    co2_percent=0.0, h2s_percent=0.0, deltacp=None):
    """
    Calculate gas compressibility factor using Hall-Yarborough correlation with component-by-component method.
    
    This function uses the same interface and methodology as the standing_katz function but implements
    the Hall-Yarborough correlation instead of Dranchuk-Abou-Kassem. It includes component-by-component
    critical property calculations and Wichert-Aziz corrections for acid gas components.
    
    References:
        - Hall, K.R. and Yarborough, L.: "A New Equation of State for Z-factor Calculations,"
          Oil & Gas Journal, June 18, 1973, pp. 82-92
        - Wichert, E. and Aziz, K.: "Calculate Z's for Sour Gases," Hydrocarbon Processing (1972)
        - Standing, M.B.: "A Pressure-Volume-Temperature Correlation for Mixtures of California Oils and Gases" (1947)
    
    Args:
        gamma_g (float): Gas specific gravity (relative to air = 1.0)
        pressure (float): Pressure in psia
        temperature (float): Temperature in °F
        composition_fractions (list): Component mole fractions in order:
            [N2, CO2, H2S, C1, C2, C3, iC4, nC4, iC5, nC5, C6, C7+]
        co2_percent (float, optional): CO2 mole percentage for acid gas correction. Default 0.0
        h2s_percent (float, optional): H2S mole percentage for acid gas correction. Default 0.0  
        deltacp (float, optional): Correction factor for cp calculations. If None, function will
            print reduced properties and request deltacp input.
            
    Returns:
        dict: Dictionary containing calculation results:
            - 'z_factor': Gas compressibility factor
            - 'Tpc': Pseudo-critical temperature in °R
            - 'Ppc': Pseudo-critical pressure in psia
            - 'Tr': Reduced temperature
            - 'Pr': Reduced pressure
            - 'deltacp': Applied deltacp correction factor
            - 'cp': Heat capacity at constant pressure in Btu/(lb·mol·°R)
            - 'cv': Heat capacity at constant volume in Btu/(lb·mol·°R)
            - 'k': Isentropic exponent (cp/cv)
            - 'iterations': Number of iterations required
            - 'converged': Boolean indicating convergence success
            
    Raises:
        ValueError: If deltacp is not provided (prints Tr, Pr for user reference)
        RuntimeError: If Hall-Yarborough iteration fails to converge
        
    Examples:
        >>> # Natural gas composition (mole fractions)
        >>> comp = [0.01, 0.05, 0.0, 0.85, 0.07, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        >>> result = hall_yarborough(0.65, 2000, 150, comp, deltacp=0.6)
        >>> print(f"Z-factor: {result['z_factor']:.4f}")
        
        >>> # High pressure sour gas
        >>> comp = [0.005, 0.15, 0.05, 0.75, 0.045, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  
        >>> result = hall_yarborough(0.75, 5000, 200, comp, co2_percent=15, h2s_percent=5, deltacp=0.8)
        
    Notes:
        - Uses component-by-component critical properties for enhanced accuracy
        - Includes Wichert-Aziz correction for CO2 and H2S content
        - More accurate than Dranchuk-Abou-Kassem at very high pressures (Pr > 15)
        - Maintains same interface as standing_katz for easy substitution
    """
    import math
    
    # Convert temperature to Rankine
    temperature_R = temperature + 459.67
    
    # Component critical properties: [Tc(°R), Pc(psia), MW(lb/lbmol)]
    critical_properties = {
        'N2':   [227.16, 493.1, 28.014],   'CO2':  [547.56, 1071.0, 44.010],
        'H2S':  [672.35, 1300.0, 34.082],  'C1':   [343.33, 667.8, 16.043],
        'C2':   [549.78, 707.8, 30.070],   'C3':   [665.73, 616.3, 44.097],
        'iC4':  [734.98, 529.1, 58.124],   'nC4':  [765.29, 550.7, 58.124],
        'iC5':  [828.77, 490.4, 72.151],   'nC5':  [845.47, 488.6, 72.151],
        'C6':   [913.27, 436.9, 86.178],   'C7+':  [972.0, 396.0, 100.20]
    }
    
    components = ['N2', 'CO2', 'H2S', 'C1', 'C2', 'C3', 'iC4', 'nC4', 'iC5', 'nC5', 'C6', 'C7+']
    
    # Calculate pseudo-critical properties using component-by-component method
    if len(composition_fractions) == len(components):
        # Use detailed component composition
        Tpc = sum(xi * critical_properties[comp][0] for xi, comp in zip(composition_fractions, components))
        Ppc = sum(xi * critical_properties[comp][1] for xi, comp in zip(composition_fractions, components))
        
        print(f"Component-by-component pseudo-critical properties:")
        print(f"  Tpc = {Tpc:.2f} °R, Ppc = {Ppc:.1f} psia")
        
        # Extract acid gas fractions from composition if not provided
        if co2_percent == 0.0 and composition_fractions[1] > 0:
            co2_percent = composition_fractions[1] * 100
        if h2s_percent == 0.0 and composition_fractions[2] > 0:
            h2s_percent = composition_fractions[2] * 100
            
        co2 = co2_percent / 100
        h2s = h2s_percent / 100
    else:
        # Fallback to gas specific gravity correlations
        Tpc = 169.2 + 349.5*gamma_g - 74*gamma_g**2
        Ppc = 756.8 - 131*gamma_g - 3.6*gamma_g**2
        co2 = co2_percent / 100
        h2s = h2s_percent / 100
    
    # Apply Wichert-Aziz correction for sour gas components
    if (co2 + h2s) > 0.001:  # Apply correction if significant acid gas content
        correction_factor = 120 * ((co2 + h2s)**0.9 + (co2 + h2s)**1.6) + (h2s**0.5 + h2s**4)
        Tpc_corr = Tpc - correction_factor
        Ppc_corr = (Ppc * Tpc_corr) / (Tpc + h2s * (1-h2s) * correction_factor)
        Tpc = Tpc_corr
        Ppc = Ppc_corr
        print(f"Applied Wichert-Aziz correction for {co2_percent:.1f}% CO2, {h2s_percent:.1f}% H2S")
        print(f"  Corrected Tpc = {Tpc:.2f} °R, Ppc = {Ppc:.1f} psia")
    
    # Calculate pseudo-reduced properties
    Tr = temperature_R / Tpc
    Pr = pressure / Ppc
    
    # Handle deltacp input
    if deltacp is None:
        # Print reduced properties for deltacp determination and raise error
        print(f"Corrected Tr: {Tr:.3f}, Corrected Pr: {Pr:.3f}")
        raise ValueError("deltacp is required. Please determine deltacp from the reduced properties chart and re-run with deltacp parameter.")
    
    # ===== HALL-YARBOROUGH CALCULATION =====
    print(f"Hall-Yarborough Z-factor Calculation:")
    print(f"  Pr = {Pr:.4f}, Tr = {Tr:.4f}")
    
    # Hall-Yarborough correlation constants
    t = 1.0 / Tr
    
    # Correlation coefficients  
    A = 0.06125 * t * math.exp(-1.2 * (1 - t)**2)
    B = t * (14.76 - 9.76 * t + 4.58 * t**2)
    C = t * (90.7 - 242.2 * t + 42.4 * t**2)
    D = 2.18 + 2.82 * t
    
    print(f"  Coefficients: A={A:.6f}, B={B:.4f}, C={C:.4f}, D={D:.4f}")
    
    # Initial guess for reduced density
    y = 0.001  # Start with low density guess
    
    max_iterations = 100
    tolerance = 1e-6
    iteration = 0
    converged = False
    
    while iteration < max_iterations:
        # Hall-Yarborough equation: F(y) = -A*Pr + (y + y^2 + y^3 - y^4)/(1-y)^3 - B*y^2 - C*y^D
        
        if y >= 1.0:
            y = 0.99  # Prevent division by zero
        
        denominator = (1 - y)**3
        if abs(denominator) < 1e-15:
            y = 0.99
            denominator = (1 - y)**3
            
        # Function value
        F = (-A * Pr + 
             (y + y**2 + y**3 - y**4) / denominator - 
             B * y**2 - 
             C * y**D)
        
        # Derivative calculation for Newton-Raphson
        numerator_deriv = (1 + 2*y + 3*y**2 - 4*y**3) * (1-y)**3 + 3*(1-y)**2 * (y + y**2 + y**3 - y**4)
        dF_dy = (numerator_deriv / ((1-y)**6) - 
                 2 * B * y - 
                 C * D * y**(D-1))
        
        # Newton-Raphson update
        if abs(dF_dy) < 1e-15:
            raise RuntimeError("Hall-Yarborough derivative became zero - cannot continue iteration")
            
        y_new = y - F / dF_dy
        
        # Ensure physical bounds
        y_new = max(0.001, min(0.99, y_new))
        
        # Check convergence
        if abs(y_new - y) < tolerance:
            converged = True
            y = y_new
            break
            
        y = y_new
        iteration += 1
        
        if iteration <= 5 or iteration % 10 == 0:
            print(f"  Iteration {iteration}: y = {y:.6f}, F = {F:.8f}")
    
    if not converged:
        raise RuntimeError(f"Hall-Yarborough iteration failed to converge after {max_iterations} iterations")
    
    # Calculate final z-factor
    Z1 = A * Pr / y
    
    print(f"  Converged in {iteration} iterations")
    print(f"  Final reduced density: {y:.6f}")
    print(f"  Z-factor: {Z1:.6f}")
    
    # Calculate heat capacity properties (same as standing_katz)
    cp = 0.25 * (28 + 0.0054 * temperature) + deltacp
    cv = cp - 1.987  # Universal gas constant in Btu/(lb·mol·°R)
    k = cp / cv  # Isentropic exponent
    
    return {
        'z_factor': Z1,
        'Tpc': Tpc,
        'Ppc': Ppc,
        'Tr': Tr,
        'Pr': Pr,
        'deltacp': deltacp,
        'cp': cp,
        'cv': cv,
        'k': k,
        'iterations': iteration + 1,
        'converged': converged
    }


def multistage_compressor(qg,
                         ps,
                         Ts,
                         ec,
                         C,
                         gamma_g,
                         co2_percent,
                         n2_percent,
                         h2s_percent,
                         h2o_percent,
                         final_pd,
                         num_stages,
                         deltacp_list=None,
                         Tpc_override=None,
                         Ppc_override=None,
                         component=None):
    """
    Calculate the total horsepower required for multistage gas compression.
    
    This function computes the horsepower for each compression stage, where each
    stage uses the discharge conditions of the previous stage as its suction conditions.
    The function automatically calculates the optimal pressure ratio distribution
    across stages and prompts for deltacp correction factors for each stage.
    
    Args:
        qg (float): Gas flow rate at standard conditions in MMSCF/D.
        ps (float): Initial suction pressure at first stage inlet in psia.
        Ts (float): Initial suction temperature at first stage inlet in °F.
        ec (float): Compressor mechanical efficiency as decimal (0.80-0.95).
        C (float): Clearance coefficient (0.03-0.10), typically 0.05.
        gamma_g (float): Gas specific gravity (relative to air, typically 0.55-0.75).
        co2_percent (float): Carbon dioxide mole percentage in gas.
        n2_percent (float): Nitrogen mole percentage in gas.
        h2s_percent (float): Hydrogen sulfide mole percentage in gas.
        h2o_percent (float): Water vapor mole percentage in gas.
        final_pd (float): Final discharge pressure after all stages in psia.
        num_stages (int): Number of compression stages.
        deltacp_list (list, optional): List of deltacp correction factors for each stage.
            If None, will raise error with reduced properties for manual determination.
        Tpc_override (float, optional): Override critical temperature in °R.
        Ppc_override (float, optional): Override critical pressure in psia.
        component (arraylike, optional): Component fractions [N2, CO2, H2S, C1-C7+].
    
    Returns:
        dict: Dictionary containing:
            - 'total_hp': Total horsepower for all stages in HP
            - 'stage_results': List of dictionaries with results for each stage
            - 'stage_pressures': List of pressures at each stage
            - 'stage_temperatures': List of temperatures at each stage
            - 'overall_compression_ratio': Total compression ratio
            - 'stage_compression_ratio': Individual stage compression ratio
    
    Raises:
        ValueError: If required parameters missing or calculations fail.
    """
    
    print(f"\n{'='*60}")
    print(f"MULTISTAGE COMPRESSOR ANALYSIS")
    print(f"{'='*60}")
    print(f"Number of stages: {num_stages}")
    print(f"Initial suction pressure: {ps:.1f} psia")
    print(f"Final discharge pressure: {final_pd:.1f} psia")
    
    # Calculate overall compression ratio and individual stage ratio
    overall_ratio = final_pd / ps
    stage_ratio = overall_ratio ** (1/num_stages)
    
    print(f"Overall compression ratio: {overall_ratio:.2f}")
    print(f"Stage compression ratio: {stage_ratio:.2f}")
    print(f"{'='*60}")
    
    # Initialize storage for results
    stage_results = []
    stage_pressures = [ps]  # Start with initial suction pressure
    stage_temperatures = [Ts]  # Start with initial suction temperature
    total_hp = 0
    
    # Process each compression stage
    current_ps = ps
    current_Ts = Ts
    
    for stage in range(num_stages):
        print(f"\nSTAGE {stage + 1} ANALYSIS:")
        print(f"{'='*40}")
        
        # Calculate discharge pressure for this stage
        current_pd = current_ps * stage_ratio
        stage_pressures.append(current_pd)
        
        print(f"Stage {stage + 1} suction pressure: {current_ps:.1f} psia")
        print(f"Stage {stage + 1} suction temperature: {current_Ts:.1f} °F")
        print(f"Stage {stage + 1} discharge pressure: {current_pd:.1f} psia")
        print(f"Stage {stage + 1} compression ratio: {stage_ratio:.2f}")
        
        # Calculate critical properties and reduced properties for this stage
        print(f"\nCritical Properties Calculation for Stage {stage + 1}:")
        
        # Component property arrays
        critical_pressures = [493, 1071, 1306, 668, 708, 616, 529, 551, 490, 489, 437, 332]
        critical_temperatures = [227, 548, 672, 343, 550, 666, 735, 765, 829, 845, 913, 1070]
        
        # Determine critical properties
        if component is not None:
            Tpc = np.sum(np.array(critical_temperatures) * np.array(component))
            Ppc = np.sum(np.array(critical_pressures) * np.array(component))
            co2 = component[1]
            h2s = component[2]
        elif Tpc_override is not None and Ppc_override is not None:
            Tpc = Tpc_override
            Ppc = Ppc_override
            co2 = co2_percent / 100
            h2s = h2s_percent / 100
        else:
            Tpc = 169.2 + 349.5*gamma_g - 74*gamma_g**2
            Ppc = 756.8 - 131*gamma_g - 3.6*gamma_g**2
            co2 = co2_percent / 100
            h2s = h2s_percent / 100
        
        # Apply Wichert-Aziz correction
        correction_factor = 120 * ((co2 + h2s)**0.9 + (co2 + h2s)**1.6) + (h2s**0.5 + h2s**4)
        Tpc_corr = Tpc - correction_factor
        Ppc_corr = (Ppc * Tpc_corr) / (Tpc + h2s * (1-h2s) * correction_factor)
        
        # Calculate reduced properties for suction conditions
        Ts_R = current_Ts + 459.67
        Tr = Ts_R / Tpc_corr
        Pr = current_ps / Ppc_corr
        
        print(f"Pseudo-critical temperature (Tpc): {Tpc_corr:.1f} °R")
        print(f"Pseudo-critical pressure (Ppc): {Ppc_corr:.1f} psia")
        print(f"Reduced temperature (Tr): {Tr:.3f}")
        print(f"Reduced pressure (Pr): {Pr:.3f}")
        
        # Check if deltacp list is provided - stop execution if not
        if deltacp_list is None:
            print(f"\nFor Stage {stage + 1}, reduced properties:")
            print(f"Tr = {Tr:.3f}, Pr = {Pr:.3f}")
            raise ValueError(f"deltacp_list is required for multistage compression. "
                           f"Please determine deltacp from reduced properties charts and "
                           f"call the function with deltacp_list=[stage1_deltacp, stage2_deltacp, ...]")
        
        # Validate deltacp list length
        if len(deltacp_list) != num_stages:
            raise ValueError(f"deltacp_list must contain {num_stages} values, got {len(deltacp_list)}")
        
        # Get deltacp for this stage
        deltacp = deltacp_list[stage]
        
        # Calculate horsepower for this stage
        try:
            stage_hp = calculate_compressor_stage_hp(
                qg=qg,
                ps=current_ps,
                Ts=current_Ts,
                ec=ec,
                C=C,
                gamma_g=gamma_g,
                co2_percent=co2_percent,
                n2_percent=n2_percent,
                h2s_percent=h2s_percent,
                h2o_percent=h2o_percent,
                deltacp=deltacp,
                pd=current_pd,
                Tpc_override=Tpc_override,
                Ppc_override=Ppc_override,
                component=component,
                return_all_vals=True
            )
            
            # Extract results from stage calculation
            hp, ev, pd_calc, R_calc, k, z1 = stage_hp
            
            # Store stage results
            stage_result = {
                'stage_number': stage + 1,
                'suction_pressure': current_ps,
                'discharge_pressure': current_pd,
                'suction_temperature': current_Ts,
                'compression_ratio': stage_ratio,
                'horsepower': hp,
                'volumetric_efficiency': ev,
                'isentropic_exponent': k,
                'suction_z_factor': z1,
                'reduced_temperature': Tr,
                'reduced_pressure': Pr,
                'deltacp': deltacp
            }
            stage_results.append(stage_result)
            
            # Add to total horsepower
            total_hp += hp
            
            print(f"\nStage {stage + 1} Results:")
            print(f"Horsepower: {hp:.2f} HP")
            print(f"Volumetric efficiency: {ev:.4f}")
            print(f"Isentropic exponent (k): {k:.4f}")
            
            # Calculate discharge temperature for next stage suction
            # Using isentropic relation: T2 = T1 * (P2/P1)^((k-1)/k)
            discharge_temp_R = Ts_R * (current_pd / current_ps)**((k-1)/k)
            discharge_temp_F = discharge_temp_R - 459.67
            stage_temperatures.append(discharge_temp_F)
            
            print(f"Discharge temperature: {discharge_temp_F:.1f} °F")
            
            # Update conditions for next stage
            current_ps = current_pd
            current_Ts = discharge_temp_F
            
        except Exception as e:
            print(f"Error calculating Stage {stage + 1}: {str(e)}")
            raise
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"MULTISTAGE COMPRESSOR SUMMARY")
    print(f"{'='*60}")
    print(f"Total horsepower required: {total_hp:.2f} HP")
    print(f"Number of stages: {num_stages}")
    print(f"Overall compression ratio: {overall_ratio:.2f}")
    print(f"Individual stage ratio: {stage_ratio:.2f}")
    
    print(f"\nStage-by-Stage Summary:")
    print(f"{'Stage':<6} {'Ps (psia)':<10} {'Pd (psia)':<10} {'Ts (°F)':<8} {'HP':<8} {'ηv':<6}")
    print(f"{'-'*60}")
    
    for i, result in enumerate(stage_results):
        stage_num = result['stage_number']
        ps_stage = result['suction_pressure']
        pd_stage = result['discharge_pressure']
        ts_stage = result['suction_temperature']
        hp_stage = result['horsepower']
        ev_stage = result['volumetric_efficiency']
        
        print(f"{stage_num:<6} {ps_stage:<10.1f} {pd_stage:<10.1f} {ts_stage:<8.1f} {hp_stage:<8.1f} {ev_stage:<6.3f}")
    
    # Return comprehensive results
    return {
        'total_hp': total_hp,
        'stage_results': stage_results,
        'stage_pressures': stage_pressures,
        'stage_temperatures': stage_temperatures,
        'overall_compression_ratio': overall_ratio,
        'stage_compression_ratio': stage_ratio,
        'num_stages': num_stages,
        'initial_conditions': {'pressure': ps, 'temperature': Ts},
        'final_conditions': {'pressure': final_pd, 'temperature': stage_temperatures[-1]}
    }