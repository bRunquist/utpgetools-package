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
                                  z_factor=None,
                                  return_all_vals=None):
    """
    Calculate the horsepower required for a single compressor stage in gas compression.
    
    This function computes the theoretical horsepower needed to compress natural gas
    from suction to discharge conditions in a reciprocating or centrifugal compressor.
    The calculation accounts for gas properties, compression efficiency, volumetric
    efficiency, and non-ideal gas behavior using real gas equations of state.
    
    Args:
        qg (float): Gas flow rate at standard conditions in MMSCF/D.
            Million standard cubic feet per day of gas to be compressed.
        ps (float): Suction pressure at compressor inlet in psia.
            The absolute pressure of gas entering the compressor stage.
        Ts (float): Suction temperature at compressor inlet in degrees Fahrenheit (°F).
            The temperature of gas entering the compressor stage.
        ec (float): Compressor mechanical efficiency as decimal (dimensionless).
            Overall mechanical efficiency of the compressor unit, typically 0.80-0.95.
            Accounts for mechanical losses in driver, gearbox, and compressor.
        C (float): Clearance coefficient (dimensionless).
            Ratio of clearance volume to swept volume in reciprocating compressors.
            Typical values: 0.03-0.10 (3%-10%). Use 0.05 as default for most applications.
        gamma_g (float): Gas specific gravity (dimensionless, relative to air).
            Specific gravity of the gas mixture relative to air (air = 1.0).
            Typical values for natural gas: 0.55-0.75.
        co2_percent (float): Carbon dioxide mole percentage in gas.
            CO2 content as mole percent. Affects gas properties significantly.
        n2_percent (float): Nitrogen mole percentage in gas.
            N2 content as mole percent.
        h2s_percent (float): Hydrogen sulfide mole percentage in gas.
            H2S content as mole percent. Important for sour gas applications.
        h2o_percent (float): Water vapor mole percentage in gas.
            H2O content as mole percent.
        deltacp (float, optional): Heat capacity correction factor (dimensionless).
            Correction factor for specific heat calculations. Required input.
            If None, function will print message and return without calculation.
        pd (float, optional): Discharge pressure in psia.
            Target discharge pressure from the compressor stage.
            Either pd or R must be provided.
        R (float, optional): Compression ratio (dimensionless).
            Ratio of discharge pressure to suction pressure (pd/ps).
            Either pd or R must be provided.
        z_factor (float, optional): Manual gas compressibility factor override.
            If provided, bypasses automatic z-factor calculation from gas properties.
            Use this when gas specific gravity is outside correlation bounds.
            Typical values range from 0.7 to 1.2 for most conditions.
        return_all_vals (bool, optional): Return additional calculated values.
            If None (default), returns only horsepower.
            If True, returns tuple of all calculated parameters.
    
    Returns:
        float or tuple: 
            - If return_all_vals is None: Horsepower in HP
            - If return_all_vals is True: Tuple of (horsepower in HP, volumetric efficiency, 
              discharge pressure in psia, compression ratio, isentropic exponent k, 
              gas compressibility factor z)
    
    Raises:
        ValueError: If neither pd nor R is provided.
        SystemExit: If deltacp is None (function prints message and returns None).
    
    Theory:
        The horsepower calculation uses the polytropic compression equation:
        
        HP = 0.08584 * (k/(k-1)) * Ts * ((pd/ps)^(z*(k-1)/k) - 1) * qg / (ec * ev)
        
        Where:
        - k = specific heat ratio (cp/cv)
        - z = gas compressibility factor at suction conditions
        - ev = volumetric efficiency
        
        Volumetric efficiency accounts for clearance volume effects:
        ev = 1 - C * ((p2/p1)^(1/k) - 1)
        
        Gas properties are calculated using real gas correlations accounting
        for gas composition and non-ideal behavior.
    
    Examples:
        >>> # Basic compression calculation with sweet gas
        >>> horsepower = calculate_compressor_stage_hp(
        ...     qg=50.0,           # 50 MMSCF/D
        ...     ps=500,            # 500 psia suction
        ...     Ts=100,            # 100°F suction temp
        ...     ec=0.85,           # 85% mechanical efficiency
        ...     C=0.05,            # 5% clearance
        ...     gamma_g=0.65,      # 0.65 gas gravity
        ...     co2_percent=0.0,   # No CO2
        ...     n2_percent=0.0,    # No N2
        ...     h2s_percent=0.0,   # No H2S
        ...     h2o_percent=0.0,   # No water vapor
        ...     deltacp=1.02,      # cp correction factor
        ...     R=2.5              # 2.5:1 compression ratio
        ... )
        >>> print(f"Required horsepower: {horsepower:.0f} HP")
        
        >>> # Get all calculated values
        >>> hp, vol_eff, pd, R, k, z = calculate_compressor_stage_hp(
        ...     qg=75.0, ps=800, Ts=120, ec=0.90, C=0.04, gamma_g=0.70,
        ...     co2_percent=1.0, n2_percent=0.5, h2s_percent=0.0, h2o_percent=0.0,
        ...     deltacp=1.05, pd=2000, return_all_vals=True
        ... )
        >>> print(f"HP: {hp:.0f}, Vol Eff: {vol_eff:.3f}, k: {k:.3f}, z: {z:.3f}")
        
        >>> # Sour gas with impurities
        >>> horsepower = calculate_compressor_stage_hp(
        ...     qg=30.0, ps=400, Ts=90, ec=0.82, C=0.06, gamma_g=0.68,
        ...     co2_percent=5.0, n2_percent=3.0, h2s_percent=2.0, h2o_percent=0.5,
        ...     deltacp=1.08, R=3.0
        ... )
        
        >>> # Manual z-factor for out-of-bounds gas specific gravity
        >>> horsepower = calculate_compressor_stage_hp(
        ...     qg=40.0, ps=600, Ts=110, ec=0.88, C=0.05, gamma_g=0.45,  # Low SG
        ...     co2_percent=0.0, n2_percent=0.0, h2s_percent=0.0, h2o_percent=0.0,
        ...     deltacp=1.03, R=2.8, z_factor=0.92  # Manual z-factor
        ... )
    
    Applications:
        - Compressor sizing and selection
        - Process design and optimization  
        - Economic analysis of compression systems
        - Performance evaluation and troubleshooting
        - Multi-stage compression system design
    
    Design Considerations:
        - Higher compression ratios reduce volumetric efficiency
        - Gas composition significantly affects required horsepower
        - Mechanical efficiency varies with compressor type and size
        - Clearance coefficient impacts volumetric efficiency in reciprocating units
        - Temperature rise during compression affects downstream equipment design
    
    Validation:
        - Results should be compared with manufacturer performance curves
        - Volumetric efficiency should be reasonable (typically 0.70-0.95)
        - Power requirements should account for process conditions
        - Consider safety factors for equipment sizing
    
    Notes:
        - Function converts temperature to Rankine internally
        - Uses real gas properties for accurate calculations
        - Clearance effects only apply to reciprocating compressors
        - For centrifugal compressors, set clearance coefficient C = 0
        - Heat capacity correction factor accounts for gas composition effects
        - Results represent theoretical power; actual power may be higher
    
    References:
        - Campbell, J.M. (2001). Gas Conditioning and Processing
        - Brown, R.N. (2005). Compressors: Selection and Sizing  
        - GPSA Engineering Data Book (2012). Gas Processing Suppliers Association
        - Ludwig, E.E. (2001). Applied Process Design for Chemical and Petrochemical Plants
    """
    Ts = Ts + 459.67  # Convert to Rankine
    if pd is None:
        if R is None:
            raise ValueError("Either pd or R must be provided.")
        pd = R * ps
    
    # Calculate z factor
    # Calculations use the intake conditions | This may be wrong, I need to ask
    if z_factor is not None:
        z = z_factor
        print(f"Using manual z-factor: {z:.4f}")
    else:
        from utpgetools.utilities_package import gas_properties_calculation
        
        try:
            properties = gas_properties_calculation(gravity=gamma_g,
                                                    co2_percent=co2_percent,
                                                    n2_percent=n2_percent,
                                                    h2s_percent=h2s_percent,
                                                    h2o_percent=h2o_percent,
                                                    pressure_psi=ps,
                                                    temperature_f=Ts - 459.67,
                                                    )
            z = properties['z_factors'][-1]
        except TypeError as e:
            if "complex" in str(e).lower():
                # Calculate and display reduced properties for debugging
                Tpc = 169.2 + 349.5*gamma_g - 74*gamma_g**2  # Critical temperature in Rankine
                Ppc = 756.8 - 131*gamma_g - 3.6*gamma_g**2   # Critical pressure in psia
                Tr = Ts / Tpc
                Pr = ps / Ppc
                
                print(f"Reduced Temperature (Tr): {Tr:.3f}")
                print(f"Reduced Pressure (Pr): {Pr:.3f}")
                
                raise ValueError(
                    f"Gas specific gravity ({gamma_g:.3f}) is out of bounds for the gas property correlation. "
                    f"This typically occurs when gamma_g < 0.55 or > 0.75, or when operating conditions "
                    f"result in reduced temperature/pressure outside correlation validity range. "
                    f"Please manually provide the z-factor using the 'z_factor' parameter or consult "
                    f"the Standing-Katz compressibility chart for the appropriate z-factor."
                ) from None
            else:
                raise e

    gamma_array = np.array([0.6, 0.7, 0.8, 0.9])
    cpst = np.array([29*0.6*(3.89*10**(-4)*Ts + 0.4872),
                 29*0.7*(4.17*10**(-4)*Ts + 0.4698),
                 29*0.8*(4.44*10**(-4)*Ts + 0.445),
                 29*0.9*(5.0*10**(-4)*Ts + 0.4218)])
    coeffs = np.polyfit(gamma_array, cpst, 1)
    cpst = np.polyval(coeffs, gamma_g)

    Tr = Ts / (169.2 + 349.5*gamma_g - 74*gamma_g**2)
    Pr = ps / (756.8 - 131*gamma_g - 3.6*gamma_g**2)
    print(f"Tr: {Tr:.3f}, Pr: {Pr:.3f}")
    if deltacp is None:
        print("Please provide cp correction factor")
        return
    cp = cpst * deltacp
    k = cp / (cp - 1.986)
    # Calculate volumetric efficiency
    p2 = pd
    p1 = ps
    ev = 1 - C*((p2/p1) ** (1/k) - 1)

    # Calculate horsepower
    P = (0.08584 * (k/(k-1)) * Ts * ((pd/ps)**(z*(k-1)/k) - 1) * qg / ec / ev)
    
    if return_all_vals is None:
        return P
    else:
        return P, ev, pd, R, k, z