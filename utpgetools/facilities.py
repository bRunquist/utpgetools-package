import numpy as np

def liquid_area(ql,C,rho=None,gamma=None,deltap=None,p_sep=None):
    '''
    Calculates the required liquid escape area for a separator pressure valve based on flow rate, sizing coefficient, and fluid properties.

    Parameters:
        ql (float): Liquid flow in bbl/d.
        C (float):
        rho (float, optional): Liquid density in lb/ft³. If not provided, it will be calculated from gamma.
        gamma (float, optional): Specific gravity of the liquid (relative to water). Used to calculate density if rho is not provided.
        deltap (float, optional): Pressure drop across the separator in psia. If not provided, it is calculated as (p_sep - 14.7).
        p_sep (float, optional): Separator pressure in psia. Used to calculate deltap if deltap is not provided.

    Returns:
        float: Required liquid area (typically in ft²).

    Notes:
        - If `deltap` is not provided, it is calculated as the difference between separator pressure and atmospheric pressure (14.7 psia).
        - If `rho` is not provided, it is calculated as `gamma * 62.4` (density of water in lb/ft³).
        - The formula uses a constant (8081.7) and assumes consistent units for all parameters.
    '''
    
    if deltap is None:
        deltap = p_sep - 14.7 # psia
    if rho is None:
        rho = gamma * 62.4 # lb/ft^3

    return (np.pi / 4 * ql / (8081.7 * C) * np.sqrt(rho / deltap))

def reduced_properties(gamma_g,P,T):
    '''
    Calculate the reduced properties of a gas given its specific gravity, pressure, and temperature.
    
    Parameters:
        gamma_g (float): Specific gravity of the gas (dimensionless).
        P (float): Pressure in psia.
        T (float): Temperature in °F.
        
    Returns:
        Pr (float): Reduced pressure (dimensionless).
        Tr (float): Reduced temperature (dimensionless).
    '''
    Pr = P / (756.8 - 131*gamma_g - 3.6*gamma_g**2)
    Tr = (T + 460) / (169.2 + 349.5*gamma_g - 74*gamma_g**2)

    return Pr, Tr


def iterate_drag_coeff(rhoc, rhod, dm, mu, accuracy=0.01):
    """
    Iterate to find the drag coefficient and terminal velocity of a droplet suspended in gas.
    Parameters:
        rhoc (float): Continuum (gas) density in lb/ft³.
        rhod (float): Droplet density in lb/ft³.
        dm (float): Droplet diameter in microns.
        mu (float): viscosity of the continuum in cp
    Returns:
        cd (float): Converged drag coefficient.
        vt (float): Terminal velocity of the droplet (units depend on calculation).
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
    Calculate pressure and temperature conditions for each stage of a multi-stage train.
    
    Parameters:
        P1 (float): Stage 1 (initial) pressure in psia.
        T1 (float): Stage 1 (initial) temperature in °F.
        Pn (float): Final stage pressure in psia.
        Tn (float): Final stage temperature in °F.
        n_stages (int): Number of separation stages.
        
    Returns:
        tuple: A tuple containing:
            - P (numpy.ndarray): Array of pressures for each stage in psia.
            - T (numpy.ndarray): Array of temperatures for each stage in °F.
            - R (float): Separation ratio used for pressure calculations.
            
    Notes:
        - Pressures are calculated using a geometric progression based on the separation ratio.
        - Temperatures are calculated using linear interpolation between initial and final values.
        - The separation ratio R = (P1/Pn)^(1/(n-1)) ensures optimal pressure distribution.
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
    '''Calculate the gas separation efficiency (Esg).
    
    Parameters:
        gas_moles (list or numpy.ndarray): List or array of gas moles flashed at each stage of separation
        MW_oil (float): Molecular weight of the oil in lb/lb-mole.
        MW_gas (float): Molecular weight of the gas in lb/lb-mole.
        gamma_oil (float, optional): Specific gravity of the oil (dimensionless).
        gamma_gas (float, optional): Specific gravity of the gas (dimensionless).
        oil_density (float, optional): Density of the oil in lb/ft³. If provided, it overrides gamma_oil.
        gas_density (float, optional): Density of the gas in lb/ft³. If provided, it overrides gamma_gas.

    Returns:
        float: Gas separation efficiency (Esg).
        '''
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