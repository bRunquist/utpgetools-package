import numpy as np
from rich.console import Console
from rich.text import Text

console = Console()

def VLP(
    diameter_in,
    total_length_ft,
    length_increment_ft,
    angle_deg,
    roughness,
    GLR,
    WOR,
    API,
    gas_gravity,
    water_gravity,
    separator_temp_F,
    separator_pressure_psi,
    outlet_temp_F,
    inlet_temp_F,
    wellhead_pressure_psi,
    flowrate_range_stb_per_day,
):
    """
    Calculates the Vertical Lift Performance (VLP) curve for a well using multiphase flow correlations.
    
    Parameters:
    -----------
    flowrate_range_stb_per_day : array-like or scalar
        If array-like: Returns arrays of flowrate (STB/D) and corresponding bottomhole pressure (psia).
        If scalar: Returns arrays of depth (ft) and corresponding pressure (psia) along the wellbore.
    
    Returns:
    --------
    If flowrate_range_stb_per_day is array-like:
        tuple: (flowrates, bottomhole_pressures)
    If flowrate_range_stb_per_day is scalar:
        tuple: (depths, pressures_along_wellbore)
    """
    diameter_ft = diameter_in / 12.0
    n_steps = int(total_length_ft / length_increment_ft)
    angle_rad = np.radians(angle_deg)
    e = roughness
    Tgrad = (inlet_temp_F - outlet_temp_F) / total_length_ft
    
    # Check if input is a single value or array
    is_single_flowrate = np.isscalar(flowrate_range_stb_per_day)

    def vlp_single(rateSTB, return_pressure_array=False):
        # Preallocate arrays
        p = np.zeros(n_steps + 1)
        T = np.zeros(n_steps + 1)
        olddeltap = 30.0
        p[0] = wellhead_pressure_psi
        T[0] = outlet_temp_F
        for i in range(1, n_steps + 1):
            error = 1.0
            iter_count = 0
            while error >= 0.001 and iter_count < 200:
                T_i = T[i-1] + (Tgrad * length_increment_ft / 2)
                p_i = p[i-1] + (olddeltap / 2)
                Oilsg = 141.5 / (API + 131.5)
                GOR = GLR * (WOR + 1)
                GasSg = gas_gravity * (1 + 5.912e-5 * API * separator_temp_F * (np.log((separator_pressure_psi + 14.7) / 114.7) / np.log(10)))
                A = API / (T_i + 460)
                if API <= 30:
                    Pb = (27.64 * GOR / (GasSg * 10 ** (11.172 * A))) ** (1 / 1.0937)
                else:
                    Pb = (56.06 * GOR / (GasSg * 10 ** (10.393 * A))) ** (1 / 1.187)
                if API <= 30:
                    Rs = GasSg * (p_i ** 1.0937) * (10 ** (11.172 * A)) / 27.64
                else:
                    Rs = GasSg * (p_i ** 1.187) * (10 ** (10.393 * A)) / 56.06
                Rs = min(max(Rs, 0), GOR)
                F = (T_i - 60) * (API / GasSg)
                if p_i < Pb:
                    if API <= 30:
                        Bo = 1 + 4.677e-4 * Rs + 1.751e-5 * F - 1.8106e-8 * Rs * F
                    else:
                        Bo = 1 + 4.677e-4 * Rs + 1.1e-5 * F - 1.337e-9 * Rs * F
                else:
                    if API <= 30:
                        Bob = 1 + 4.677e-4 * GOR + 1.751e-5 * F - 1.8106e-8 * GOR * F
                    else:
                        Bob = 1 + 4.677e-4 * GOR + 1.1e-5 * F - 1.337e-9 * GOR * F
                    co = (-1.433 + 5 * Rs + 17.2 * T_i - 1.18 * GasSg + 12.61 * API) / (p_i * 1e5)
                    Bo = Bob * np.exp(co * (Pb - p_i))
                Zv = 3.0324 - 0.02023 * API
                y = 10 ** Zv
                x = y * (T_i ** -1.163)
                Muod = (10 ** x) - 1
                aa = 10.715 * ((Rs + 100) ** -0.515)
                bb = 5.44 * ((Rs + 150) ** -0.338)
                if p_i <= Pb:
                    Muo = aa * (Muod ** bb)
                else:
                    Muob = aa * (Muod ** bb)
                    m = 2.6 * (p_i ** 1.187) * np.exp(-11.513 - 8.98e-5 * p_i)
                    Muo = Muob * ((p_i / Pb) ** m)
                Tc = gas_gravity * 314.8148 + 168.5185
                Pc = gas_gravity * (-47.619) + 700.4762
                Tr = (T_i + 460) / Tc
                Pr = p_i / Pc
                Dz = 10 ** (0.3106 - 0.49 * Tr + 0.1824 * (Tr ** 2))
                Cz = 0.132 - 0.32 * (np.log(Tr) / np.log(10))
                Bz = (0.62 - 0.23 * Tr) * Pr + ((0.066 / (Tr - 0.86)) - 0.037) * (Pr ** 2) + 0.32 * (Pr ** 6) / (10 ** (9 * (Tr - 1)))
                Az = 1.39 * ((Tr - 0.92) ** 0.5) - 0.36 * Tr - 0.101
                Z = Az + (1 - Az) * np.exp(-Bz) + Cz * (Pr ** Dz)
                densG = 2.7 * gas_gravity * p_i / ((T_i + 460) * Z)
                Bg = 0.0283 * Z * (T_i + 460) / p_i
                Xvisc = 3.5 + (986 / (T_i + 460)) + 0.01 * 29 * gas_gravity
                Lamda = 2.4 - 0.2 * Xvisc
                Kvisc = ((9.4 + 0.02 * 29 * gas_gravity) * ((T_i + 460) ** 1.5)) / (209 + 19 * 29 * gas_gravity + T_i + 460)
                MuG = Kvisc * 1e-4 * np.exp(Xvisc * ((0.01602 * densG) ** Lamda))
                densAir = 28.97 * p_i / (10.73159 * Z * (T_i + 460))
                dissgasgr = densG / densAir
                densO = (62.4 * Oilsg / Bo) + (0.0764 * dissgasgr * Rs / (Bo * 5.615))
                densW = 62.4 * water_gravity
                densL = (WOR * densW + Bo * densO) / (WOR + Bo)
                MuW = np.exp(1.003 - 1.479e-2 * T_i + 1.982e-5 * (T_i ** 2))
                MuL = ((WOR * densW) / (WOR * densW + Bo * densO)) * MuW + ((Bo * densO) / (WOR * densW + Bo * densO)) * Muo
                sigmaO = 30
                sigmaW = 74
                sigma = ((WOR * densW) / (WOR * densW + Bo * densO)) * sigmaW + ((Bo * densO) / (WOR * densW + Bo * densO)) * sigmaO
                Area = np.pi * (diameter_ft ** 2) / 4
                ql = (WOR + Bo) * rateSTB * 5.615
                qg = Bg * (GOR - Rs) * rateSTB
                qt = qg + ql
                Fg = qg / qt
                Fl = 1 - Fg
                usg = qg / (Area * 86400)
                usl = ql / (Area * 86400)
                um = usg + usl
                LB = 1.071 - (0.2218 * (um ** 2) / diameter_ft)
                LB = max(LB, 0.13)
                if Fg < LB:
                    us = 0.8
                    yl = 1 - 0.5 * (1 + (um / us) - ((1 + (um / us)) ** 2 - 4 * usg / us) ** 0.5)
                    yl = min(max(yl, Fl), 1)
                    mdotl = Area * usl * densL * 86400
                    Nre = 0.022 * mdotl / (diameter_ft * MuL)
                    try:
                        ff = (1 / (-4 * (np.log((e / 3.7065) - (5.0452 / Nre) * (np.log(((e ** 1.1098) / 2.8257) + ((7.149 / Nre) ** 0.8981)) / np.log(10))) / np.log(10)))) ** 2
                    except:
                        ff = 0.02
                    DensAvg = (1 - yl) * densG + yl * densL
                    dpdz = (1 / 144) * ((np.sin(angle_rad) * DensAvg) + ((ff * (mdotl ** 2)) / (7.413e10 * (diameter_ft ** 5) * densL * (yl ** 2))))
                    DeltaP = dpdz * length_increment_ft
                else:
                    Nvl = 1.938 * usl * ((densL / sigma) ** 0.25)
                    Nvg = 1.938 * usg * ((densL / sigma) ** 0.25)
                    Nd = 120.872 * diameter_ft * ((densL / sigma) ** 0.5)
                    Nl = 0.15726 * MuL * ((1 / (densL * (sigma ** 3))) ** 0.25)
                    CNl = 7.9595 * (Nl ** 6) - 13.144 * (Nl ** 5) + 8.3825 * (Nl ** 4) - 2.4629 * (Nl ** 3) + 0.2213 * (Nl ** 2) + 0.0473 * Nl + 0.0018
                    group1 = Nvl * (p_i ** 0.1) * CNl / ((Nvg ** 0.575) * (14.7 ** 0.1) * Nd)
                    ylpsy = -3.44985871528755e15 * (group1 ** 6) + 56858620047687.2 * (group1 ** 5) - 368100995579.95 * (group1 ** 4) + 1189881753.18 * (group1 ** 3) - 2037716.09 * (group1 ** 2) + 1868.71 * group1 + 0.1
                    group2 = Nvg * (Nl ** 0.38) / (Nd ** 2.14)
                    psy = 116159 * (group2 ** 4) - 22251 * (group2 ** 3) + 1232.1 * (group2 ** 2) - 4.8183 * group2 + 0.9116
                    psy = max(psy, 1)
                    yl = ylpsy * psy
                    yl = min(max(yl, Fl), 1)
                    DensAvg = (1 - yl) * densG + yl * densL
                    mdot = Area * (usl * densL + usg * densG) * 86400
                    try:
                        Nre = 0.022 * mdot / (diameter_ft * (MuL ** yl) * (MuG ** (1 - yl)))
                        ff = (1 / (-4 * (np.log((e / 3.7065) - (5.0452 / Nre) * (np.log(((e ** 1.1098) / 2.8257) + ((7.149 / Nre) ** 0.8981)) / np.log(10))) / np.log(10)))) ** 2
                    except:
                        ff = 0.02
                    dpdz = (1 / 144) * ((np.sin(angle_rad) * DensAvg) + ((ff * (mdot ** 2)) / (7.413e10 * (diameter_ft ** 5) * DensAvg)))
                    DeltaP = dpdz * length_increment_ft
                error = abs(DeltaP - olddeltap) / (abs(olddeltap) + 1e-6)
                olddeltap = DeltaP
                iter_count += 1
            p[i] = p[i-1] + DeltaP
            T[i] = T[i-1] + length_increment_ft * Tgrad
        
        if return_pressure_array:
            # Create depth array from wellhead (0) to total depth
            depths = np.arange(0, total_length_ft + length_increment_ft, length_increment_ft)
            return depths[:len(p)], p
        else:
            return p[-1]

    # Handle single flowrate vs flowrate range
    if is_single_flowrate:
        # Return pressure vs depth curve for single flowrate
        depths, pressures = vlp_single(flowrate_range_stb_per_day, return_pressure_array=True)
        return depths, pressures
    else:
        # Run for each flowrate sequentially to get VLP curve
        bhp_list = [vlp_single(rateSTB) for rateSTB in flowrate_range_stb_per_day]
        return np.array(flowrate_range_stb_per_day), np.array(bhp_list)
def IPR(q_test=None, p_test=None, p_res=None, pwf=None, J=None, p_b=None, show_plot=False, plot_mode='overlay', constant_J=True):
    """
    Calculates the Inflow Performance Relationship (IPR) for a well, estimating the production rate based on reservoir and well parameters.
    This function adapts to both below- and above-bubble point conditions, providing a versatile tool for artificial lift and reservoir engineering analysis.
    
    Parameters:
    -----------
    q_test : array-like or scalar
        Well test flowrate data (STB/D). If array, must have same length as p_test and p_res.
    p_test : array-like or scalar
        Well test pressure data (psi). If array, must have same length as q_test and p_res.
    p_res : array-like or scalar
        Reservoir pressure(s) (psi). If array, must have same length as q_test and p_test.
        Each reservoir pressure is associated with the corresponding well test data.
    pwf : array-like or scalar
        Wellbore flowing pressure(s) for IPR calculation (psi).
    J : scalar, optional
        Productivity index. If provided, overrides calculation from test data.
    p_b : scalar
        Bubble point pressure (psi).
    show_plot : bool, default False
        Whether to display IPR plots.
    plot_mode : str, default 'overlay'
        Plot mode: 'overlay' or 'separate'.
    constant_J : bool, default True
        Whether to use constant productivity index across different reservoir pressures.
        If True, uses J from first test point for all reservoir pressures.
        If False, calculates J individually for each reservoir pressure using its corresponding test data.
    
    Returns:
    --------
    array or list of arrays
        Flow rates corresponding to input pwf values.
        
    Raises:
    -------
    ValueError
        If the number of well tests doesn't match the number of reservoir pressures,
        or if q_test and p_test have different shapes.
    
    Examples:
    ---------
    # Single test point and reservoir pressure
    IPR(q_test=100, p_test=2000, p_res=3000, pwf=np.linspace(0, 3000, 100), p_b=1800)
    
    # Multiple test points, each associated with a reservoir pressure
    q_tests = [80, 120, 150]  # STB/D
    p_tests = [2200, 1800, 1500]  # psi  
    p_res = [2800, 3000, 3200]  # psi - each corresponds to a test point
    IPR(q_test=q_tests, p_test=p_tests, p_res=p_res, pwf=np.linspace(0, 3000, 100), p_b=1800)
    """
    import numpy as np

    # Convert inputs to arrays
    q_test_arr = np.asarray(q_test)
    p_test_arr = np.asarray(p_test)
    pwf_arr = np.asarray(pwf)
    
    # Validate test data inputs
    if q_test_arr.shape != p_test_arr.shape:
        raise ValueError("q_test and p_test must have the same shape when both are arrays")
    
    # Ensure p_res is iterable and convert to array
    try:
        p_res_arr = np.asarray(p_res)
        p_res_list = list(p_res_arr)
    except TypeError:
        p_res_arr = np.asarray([p_res])
        p_res_list = [p_res]
    
    # Validate that well test data matches reservoir pressure data
    if q_test_arr.ndim > 0 and p_res_arr.ndim > 0:
        if len(q_test_arr) != len(p_res_arr):
            raise ValueError(f"Number of well tests ({len(q_test_arr)}) must match number of reservoir pressures ({len(p_res_arr)})")
    elif q_test_arr.ndim > 0 or p_res_arr.ndim > 0:
        # One is array, one is scalar - check if they're compatible
        if q_test_arr.ndim > 0 and len(q_test_arr) != len(p_res_list):
            raise ValueError(f"Number of well tests ({len(q_test_arr)}) must match number of reservoir pressures ({len(p_res_list)})")
        elif p_res_arr.ndim > 0 and len(p_res_list) > 1:
            # Multiple reservoir pressures but single test point
            raise ValueError(f"Single well test provided but multiple reservoir pressures ({len(p_res_list)}) given. Provide one test per reservoir pressure.")

    q_curves = []
    
    # Calculate J once if constant_J is True and J is None
    J_fixed = None
    if constant_J and J is None:
        # Use the first test point for constant J calculation
        if q_test_arr.ndim == 0:  # scalar
            J_fixed = q_test_arr / (p_res_list[0] - p_test_arr)
        else:  # array - use first test point
            J_fixed = q_test_arr[0] / (p_res_list[0] - p_test_arr[0])
            
    for i, pres in enumerate(p_res_list):
        if J is not None:
            J_val = J
        elif constant_J:
            J_val = J_fixed
        else:
            # Use corresponding test data for this reservoir pressure
            if q_test_arr.ndim == 0:  # scalar test data
                q_test_val = q_test_arr
                p_test_val = p_test_arr
            else:  # array test data - use corresponding index
                q_test_val = q_test_arr[i]
                p_test_val = p_test_arr[i]
            J_val = q_test_val / (pres - p_test_val)
        
        # Check if reservoir pressure is below bubble point
        if pres >= p_b:
            # Original behavior when reservoir pressure is above bubble point
            qb = J_val * (pres - p_b)
            qmax = qb / (1-0.2 * (p_b / pres) - 0.8 * (p_b / pres) ** 2)
            console.print(f"\n[bright_cyan]qb[/bright_cyan] for [bright_cyan]p_res[/bright_cyan]={pres} psi: [bright_red]{qb:.2f}[/bright_red] STB/D")
            console.print(f"[bright_cyan]qmax[/bright_cyan] for [bright_cyan]p_res[/bright_cyan]={pres} psi: [bright_red]{qmax:.2f}[/bright_red] STB/D")
            q = np.where(
                pwf_arr > p_b,
                # qb + (J_val * p_b / 1.8) * (1 - 0.2 * pwf_arr / p_b - 0.8 * (pwf_arr / p_b) ** 2), # previous version
                # qb / (1-0.2 * pwf_arr / p_b - 0.8 * (pwf_arr / p_b) ** 2), # testing new version
                J_val * (pres - pwf_arr),
                (1 - 0.2 * (pwf_arr / pres) - 0.8 * (pwf_arr / pres) ** 2) * qmax # revised version
            )
        else:
            # Use full two-phase Vogel IPR when reservoir pressure is below bubble point
            if q_test_arr.ndim == 0:  # scalar test data
                q_test_val = q_test_arr
                p_test_val = p_test_arr
            else:  # array test data - use corresponding index
                q_test_val = q_test_arr[i]
                p_test_val = p_test_arr[i]
            
            qmax = q_test_val / (1-0.2 * (p_test_val / pres) - 0.8 * (p_test_val / pres) ** 2)
            console.print(f"Reservoir pressure below bubble point - using two-phase Vogel IPR")
            console.print(f"[bright_cyan]qmax[/bright_cyan] for [bright_cyan]p_res[/bright_cyan]={pres} psi: [bright_red]{qmax:.2f}[/bright_red] STB/D")
            q = (1 - 0.2 * (pwf_arr / pres) - 0.8 * (pwf_arr / pres) ** 2) * qmax
                
            
        # If input was scalar, return scalar
        if np.isscalar(pwf):
            q = float(q)
        q_curves.append(q)

    if show_plot:
        import matplotlib.pyplot as plt
        if plot_mode == 'overlay':
            for i, q in enumerate(q_curves):
                mask = pwf_arr <= p_res_list[i]
                plt.plot(np.array(q)[mask], pwf_arr[mask], label=f'p_res={p_res_list[i]}')
            plt.xlabel('q (STB/D)')
            plt.ylabel('pwf (psi)')
            plt.title('Reservoir IPR')
            plt.grid()
            plt.xlim(0)
            plt.ylim(0)
            plt.legend()
            plt.show()
        elif plot_mode == 'separate':
            for i, q in enumerate(q_curves):
                mask = pwf_arr <= p_res_list[i]
                plt.figure()
                plt.plot(np.array(q)[mask], pwf_arr[mask])
                plt.xlabel('q (STB/D)')
                plt.ylabel('pwf (psi)')
                plt.title(f'Reservoir IPR (p_res={p_res_list[i]})')
                plt.xlim(0)
                plt.ylim(0)
                plt.grid()
                plt.show()

    return q_curves if len(q_curves) > 1 else q_curves[0]

def newton_laplace_vg(z,T,gamma_g,k=1.25):
    T = T + 460 # Convert to Rankine
    return 41.44*np.sqrt(k * z * T / gamma_g)

def fluid_level_from_shot(deltat,vg):
    return deltat*vg/2

def calculate_flow_divided_by_area(deltap,h,deltat):
    return 0.68 * (deltap * h / deltat)

def mccoy_correlation(q_over_a):
    return 4.6572 * (q_over_a)**(-0.319)

def fluid_fractions(fl,WOR):
    fo = fl / (1+WOR)
    fw = fl-fo
    fg = 1-fl
    return fo,fw,fg

def bhp_from_fluid_level(gamma_f,TD,H,psa=50):
    return psa * (1 + H/40000) + 0.433 * gamma_f * (TD - H)

def echometer_fl_bhp(PBU_time, travel_time, deltap, API, gamma_g, gamma_w, temperature_f, psa, WOR, TVD):
    """
    Calculates fluid level and bottomhole pressure (BHP) from an echometer shot using acoustic analysis.
    
    This function analyzes echometer (acoustic) well test data to determine the fluid level in the wellbore
    and calculate the corresponding bottomhole pressure. The calculation process involves determining the 
    speed of sound in gas, calculating fluid level from travel time, estimating flow rates and fluid 
    fractions, computing mixture density, and finally calculating BHP using hydrostatic pressure principles.
    
    The function prints intermediate calculations and equations at each step for transparency and validation.
    
    Inputs:
    -------
    PBU_time : float
        Pressure buildup time (seconds) - time the well is shut in for the PBU test
    travel_time : float
        Acoustic travel time (seconds) - time for sound to travel down to fluid level and back up
    deltap : float
        Pressure buildup during test (psi) - pressure increase observed during the acoustic shot
    API : float
        Oil API gravity (degrees API) - used to calculate oil specific gravity
    gamma_g : float
        Gas specific gravity (dimensionless, relative to air) - used for gas property calculations
    gamma_w : float
        Water specific gravity (dimensionless, relative to water) - typically around 1.0-1.1
    temperature_f : float
        Temperature (degrees Fahrenheit) - wellbore temperature for gas property calculations
    psa : float
        Shut-in annulus pressure / surface separator pressure (psia) - reference pressure at surface
    WOR : float
        Water-to-oil ratio (dimensionless) - volumetric ratio of water to oil production
    TVD : float
        Well true vertical depth (feet) - total vertical depth of the well
    
    Returns:
    --------
    tuple : (float, float)
        - Fluid level (feet) : Distance from surface to fluid interface
        - BHP (psia) : Calculated bottomhole pressure
    
    Process Overview:
    -----------------
    1. Calculate gas compressibility factor (z) using gas properties
    2. Determine sonic velocity in gas using Newton-Laplace equation
    3. Calculate fluid level from acoustic travel time
    4. Estimate flow rate per unit area from pressure buildup
    5. Calculate liquid fraction using McCoy correlation
    6. Determine individual fluid fractions (oil, water, gas)
    7. Compute mixture density accounting for gas expansion
    8. Calculate BHP using hydrostatic pressure with elevation correction
    
    Notes:
    ------
    - All intermediate calculations and equations are printed to console
    - Uses McCoy correlation for liquid holdup estimation
    - Accounts for gas expansion effects in mixture density calculation
    - Includes elevation correction for annulus pressure
    - Requires utpgetools.utilities_package for gas property calculations
    
    Example:
    --------
    >>> fluid_level, bhp = echometer_fl_bhp(
    ...     deltat=2.5,      # 2.5 seconds travel time
    ...     deltap=15.0,     # 15 psi pressure buildup  
    ...     API=35.0,        # 35 API oil
    ...     gamma_g=0.65,    # 0.65 gas specific gravity
    ...     gamma_w=1.05,    # 1.05 water specific gravity
    ...     temperature_f=150, # 150°F temperature
    ...     psa=50.0,        # 50 psia surface pressure
    ...     WOR=2.0,         # 2:1 water-oil ratio
    ...     TVD=5000.0       # 5000 ft well depth
    ... )
    """
    # Steps for calculation

    # Speed of sound in gas (vg) from Newton-Laplace equation
        # vg = 41.44 * sqrt(k * z * T / gamma_g)
        # Use z at shut in annulus pressure or average from surface to SI ann
        # SG at standard conditions (given in problem)
    from utpgetools.utilities_package import gas_properties_calculation
    
    properties = gas_properties_calculation(gamma_g,
                                            pressure_psi=psa,
                                            temperature_f=temperature_f,
                                            )
    z = properties['z_factors'][-1] # Use z at SI ann
    console.print(f"\nUsing average pressure in annulus, [bright_cyan]z[/bright_cyan] = [bright_red]{z:.3f}[/bright_red]")
    T = temperature_f + 460  # Convert to Rankine
    vg = 41.44 * np.sqrt(1.25 * z * T / gamma_g)
    console.print(f"\nSonic velocity in gas [bright_cyan]vg[/bright_cyan] = [bright_red]{vg:.3f}[/bright_red] ft/s")
    console.print(f"\n[bright_cyan]vg[/bright_cyan] = 41.44 * sqrt(1.25 * z * T / gamma_g)")
    # Calculate L
        # L = deltat * vg / 2
    L = travel_time * vg / 2
    console.print(f"\nFluid level [bright_cyan]L[/bright_cyan] = [bright_red]{L:.3f}[/bright_red] ft")
    console.print(f"\n[bright_cyan]L[/bright_cyan] = deltat * vg / 2")

    # Calculate q/A
        # q/A = 0.68 * (deltap * h / deltat)
        # deltap is pressure buildup during test
    q_over_a = 0.68 * (deltap * L / PBU_time)
    console.print(f"\nFlow rate divided by area [bright_cyan]q/A[/bright_cyan] = [bright_red]{q_over_a:.3f}[/bright_red] ft/s")
    console.print(f"\n[bright_cyan]q/A[/bright_cyan] = 0.68 * (deltap * h / deltat)")
    # Calculate fl (liquid fraction)
        # fl = 4.6572 * (q/A)^-0.319
    fl = 4.6572 * (q_over_a ** -0.319)
    console.print(f"\nLiquid fraction using McCoy: [bright_cyan]fl[/bright_cyan] = [bright_red]{fl:.3f}[/bright_red]")
    console.print(f"\n[bright_cyan]fl[/bright_cyan] = 4.6572 * (q/A)^-0.319")
    # Calculate fo, fw, fg
        # fo = fl / (1 + WOR)
        # fw = fl - fo
        # fg = 1 - fl
    fo = fl / (1 + WOR)
    fw = fl - fo
    fg = 1 - fl
    console.print(f"\nOil fraction [bright_cyan]fo[/bright_cyan] = [bright_red]{fo:.3f}[/bright_red], water fraction [bright_cyan]fw[/bright_cyan] = [bright_red]{fw:.3f}[/bright_red], gas fraction [bright_cyan]fg[/bright_cyan] = [bright_red]{fg:.3f}[/bright_red]")
    console.print(f"\n[bright_cyan]fo[/bright_cyan] = fl / (1 + WOR), [bright_cyan]fw[/bright_cyan] = fl - fo, [bright_cyan]fg[/bright_cyan] = 1 - fl")
    # Calculate the mixture density rhof
        # rhof = 62.4 * (gamma_o * fo + gamma_w * fw) / (1 - 0.0187 * (TVD - L) * gamma_g / (z * T ) * fg)
        # Temp is in rankine. z is at SI ann
    gamma_o = 141.5 / (API + 131.5)
    z = properties['z_factors'][-1]  # Use z at SI ann
    rhof = 62.4 * (gamma_o * fo + gamma_w * fw) / (1 - 0.0187 * (TVD - L) * gamma_g / (z * T) * fg)
    console.print(f"\nMixture density [bright_cyan]rhof[/bright_cyan] = [bright_red]{rhof:.3f}[/bright_red] lb/ft3")
    console.print(f"\n[bright_cyan]rhof[/bright_cyan] = 62.4 * (gamma_o * fo + gamma_w * fw) / (1 - 0.0187 * (TVD - L) * gamma_g / (z * T ) * fg)")
    # Calculate BHP
        # BHP = PSA * (1 + H / 40000) + 0.433 * rhof / 62.4 * (TD - L)
    bhp = psa * (1 + L / 40000) + 0.433 * rhof/62.4 * (TVD - L)
    console.print(f"\nBottomhole pressure [bright_cyan]BHP[/bright_cyan] = [bright_red]{bhp:.3f}[/bright_red] psia")
    console.print(f"\n[bright_cyan]BHP[/bright_cyan] = PSA * (1 + H / 40000) + 0.433 * SGf * (TD - L)")
    return L,bhp

def gas_valve_depths(Pinj, pwh, Gk, Gg, Pdt, Gdt, packer_depth, Kickoff=None):

    """
    Calculate the Required depths for gas lift valves in a well.

    Inputs:
    -------
    Pinj : float
        Injection pressure at the surface (psi).
    pwh : float
        Wellhead pressure (psi).
    Gk : float
        Kill fluid gradient (psi/ft). 0.433 for water
    Gg : float
        Gas gradient in the annulus (psi/ft).
    Pdt : float
        Design discharge pressure (psi).
    Gdt : float
        Pressure gradient for unloading
    packer_depth : float
        Depth of the packer (ft).
    Kickoff : float, optional
        Kickoff pressure (psi). If None, uses Pinj for initial pressure.

    Returns:
    --------
    valve_depths : list of float
        List of calculated valve depths (ft).

    """
    valve_depths = []
    if Kickoff is None:
        p1 = Pinj
    else:
        p1 = Kickoff
    H1 = (p1-pwh) / (Gk-Gg)
    valve_depths.append(H1)
    Hn = H1

    while True:
        H_old = Hn

        Hn = (Pinj - Pdt + (Gg-Gdt) * H_old) / (Gk-Gg) + H_old
    
        if Hn > packer_depth:
            Hn = packer_depth
            valve_depths.append(Hn)
            break
        
        valve_depths.append(Hn)

    valve_depths = [round(d) for d in valve_depths]

    return valve_depths

def plunger_rate_calculation(TD, t_blowdown, WOR, tubing_ID, slug_height, loss_fraction, pt, Wp, oil_API, water_gravity):
    """
    Calculate plunger cycle time, slug volume, production rate, and Required compressor pressure.

    Inputs:
    -------
    TD : float
        Total depth of the well (ft).
    t_blowdown : float
        Blowdown time (minutes).
    WOR : float
        Water-oil ratio (dimensionless).
    tubing_ID : float
        Inner diameter of the tubing (inches).
    slug_height : float
        Height of the fluid slug (ft).
    loss_fraction : float
        Percentage loss per thousand ft as a decimal.
    pt : float
        Tubing pressure at the surface (psia).
    Wp : float
        Weight of the plunger (lbs).
    oil_API : float
        Oil API gravity (degrees API).
    water_gravity : float
        Water specific gravity (dimensionless).
    """

    t_rise = TD / 750 # minutes to rise
    console.print(f"\nPlunger rise time: [bright_red]{t_rise:.0f}[/bright_red] minutes")
    console.print(f"\n[bright_cyan]t_rise[/bright_cyan] = TD / 750")
    
    t_fall = TD / 250 # minutes to fall
    console.print(f"\nPlunger fall time: [bright_red]{t_fall:.0f}[/bright_red] minutes")
    console.print(f"\n[bright_cyan]t_fall[/bright_cyan] = TD / 250")
    
    t_cycle = t_rise + t_fall + t_blowdown # total cycle time in minutes
    console.print(f"\nPlunger cycle time: [bright_red]{t_cycle:.0f}[/bright_red] minutes")

    cycles_per_day = 1440 / t_cycle
    console.print(f"\nPlunger cycles per day: [bright_red]{cycles_per_day:.0f}[/bright_red]")

    tubing_capacity = np.pi * (tubing_ID / 2 / 12) ** 2 # in cubic feet per foot of tubing
    console.print(f"\nTubing capacity: [bright_red]{tubing_capacity:.3f}[/bright_red] ft3/ft")
    
    slug_volume = tubing_capacity * slug_height # ft3
    console.print(f"\nSlug volume: [bright_red]{slug_volume:.3f}[/bright_red] ft3")
    
    slug_volume_stb = slug_volume / 5.615 # convert to STB
    console.print(f"\nSlug volume: [bright_red]{slug_volume_stb:.3f}[/bright_red] STB")
    
    loss_fraction_perthousand = loss_fraction / 1000 # convert to loss per thousand feet
    cycle_volume = slug_volume_stb * (1 - loss_fraction_perthousand * TD) # STB per cycle
    console.print(f"\nCycle volume (after losses): [bright_red]{cycle_volume:.3f}[/bright_red] STB")
    console.print(f"\n[bright_cyan]cycle_volume[/bright_cyan] = slug_volume_stb * (1 - loss_fraction_perthousand * TD)")
    
    production_rate = cycle_volume * cycles_per_day # STB per day
    console.print(f"\nProduction rate: [bright_red]{production_rate:.0f}[/bright_red] BLPD")
    console.print(f"[bright_cyan]Production_rate[/bright_cyan] = [bright_red]{(production_rate / (1+WOR)):.0f}[/bright_red] BOPD")

    # Calculate Required compressor pressure
    
    At = np.pi * (tubing_ID / 2) ** 2 # in square inches
    # calculate liquid specific gravity, which is the weighted average of oil and water at surface conditions
    water_cut = WOR / (1 + WOR)
    oil_sg = 141.5 / (131.5 + oil_API) # assuming 40 API oil
    gamma_l = oil_sg * (1 - water_cut) + water_gravity * water_cut
    console.print(f"\nLiquid specific gravity [bright_cyan]gamma_l[/bright_cyan] = [bright_red]{gamma_l:.3f}[/bright_red]")
    console.print(f"\n[bright_cyan]gamma_l[/bright_cyan] = oil_sg * (1 - water_cut) + water_gravity * water_cut")
    Ws = slug_volume_stb * 350 * gamma_l
    pg = 1.5 * ((Ws + Wp) / At) + pt
    console.print(f"\nRequired compressor pressure: [bright_red]{pg:.0f}[/bright_red] psia")
    console.print(f"\n[bright_cyan]pg[/bright_cyan] = 1.5 * ((Ws + Wp) / At) + pt")
    return

def pcp_design(API, 
               gas_gravity, 
               water_gravity, 
               GLR, 
               WOR, 
               pwf, 
               BHT, 
               tubing_ID, 
               rod_diameter,
               Wr, 
               pump_depth_ft, 
               oil_rate, 
               liquid_rate,
               pump_capacity,
               rotor_diameter,
               separator_pressure=100, 
               separator_temperature=100,
               pump_efficiency=None,
               t_surface=None,
               wellhead_pressure=None,
               bearing_load_rating=50500,
               lifetime_revolutions=90*10**6,
               ):
    """
    Designs and analyzes a Progressive Cavity Pump (PCP) artificial lift system for oil wells.
    
    This function performs comprehensive design calculations for PCP systems, including pump performance
    analysis, power requirements, mechanical stress analysis, and bearing life estimation. The function
    calculates fluid properties at pump conditions, determines discharge pressure using multiphase flow
    correlations, computes pump operating parameters, and evaluates mechanical design criteria.
    
    All intermediate calculations and equations are printed to console with color formatting for 
    clear visualization of the design process and results.
    
    Inputs:
    -----------
    API : float
        Oil API gravity (degrees API) - used to calculate oil specific gravity and properties
    gas_gravity : float
        Gas specific gravity (dimensionless, relative to air) - for gas property calculations
    water_gravity : float
        Water specific gravity (dimensionless, relative to water) - typically 1.0-1.1
    GLR : float
        Gas-to-liquid ratio (scf/stb) - total gas production per stock tank barrel of liquid
    WOR : float
        Water-to-oil ratio (dimensionless) - volumetric ratio of water to oil production
    pwf : float
        Wellbore flowing pressure at pump intake (psia) - bottomhole flowing pressure
    BHT : float
        Bottomhole temperature (degrees Fahrenheit) - temperature at pump depth
    tubing_ID : float
        Tubing internal diameter (inches) - production tubing inner diameter
    rod_diameter : float
        Rod diameter (inches) - PCP drive rod diameter
    Wr : float
        Rod weight per unit length (lbf/ft) - weight of rod string per foot
    pump_depth_ft : float
        Pump setting depth (feet) - vertical depth where pump is installed
    oil_rate : float
        Oil production rate (STB/D) - stock tank barrels of oil per day
    liquid_rate : float
        Total liquid production rate (STB/D) - oil plus water production rate
    pump_capacity : float
        Pump capacity (bbl/d/rpm) - volumetric displacement per revolution per rpm
    rotor_diameter : float
        Pump rotor diameter (inches) - diameter of the PCP rotor
    separator_pressure : float, default 100
        Separator pressure (psia) - surface separation pressure for property calculations
    separator_temperature : float, default 100
        Separator temperature (degrees Fahrenheit) - surface separation temperature
    pump_efficiency : float, optional
        Pump volumetric efficiency (decimal) - if None, function prompts for input after discharge pressure calculation
    t_surface : float, optional
        Surface temperature (degrees Fahrenheit) - if None, uses separator_temperature
    wellhead_pressure : float, optional
        Wellhead pressure (psia) - if None, uses separator_pressure
    bearing_load_rating : float, default 50500
        Bearing load rating (lbf) - manufacturer's bearing capacity rating
    lifetime_revolutions : float, default 90e6
        Expected bearing lifetime (revolutions) - total revolutions for bearing life calculation
    
    Returns:
    --------
    None
        Function prints all results to console and does not return values
    
    Calculations Performed:
    -----------------------
    1. Fluid property analysis at pump intake conditions (Rs, z-factor)
    2. Multiphase flow pressure drop calculation from surface to pump depth
    3. Pump operating speed determination based on production requirements
    4. Pump torque calculation for drive system sizing
    5. Pump power requirements (BHP input and HHP output)
    6. Pump efficiency evaluation
    7. Mechanical stress analysis including:
       - Buoyed rod load calculation
       - Pump thrust force analysis
       - Von Mises stress determination
    8. Bearing life estimation based on load and operating speed
    
    Design Considerations:
    ----------------------
    - Accounts for dissolved gas effects on fluid properties
    - Includes buoyancy effects on rod loading
    - Considers pump thrust forces in bearing analysis
    - Uses multiphase flow correlations for accurate pressure calculations
    - Provides mechanical design validation through stress analysis
    
    Notes:
    ------
    - Requires utpgetools.utilities_package for property calculations
    - Uses rich console formatting for enhanced output readability
    - Function will prompt for pump efficiency if not provided initially
    - All intermediate equations are displayed for transparency
    - Bearing life calculation assumes L10 life criteria
    
    Example:
    --------
    >>> pcp_design(
    ...     API=30.0,                    # 30 API oil
    ...     gas_gravity=0.65,            # 0.65 gas specific gravity  
    ...     water_gravity=1.05,          # 1.05 water specific gravity
    ...     GLR=500,                     # 500 scf/stb GLR
    ...     WOR=2.0,                     # 2:1 water-oil ratio
    ...     pwf=1500,                    # 1500 psia intake pressure
    ...     BHT=180,                     # 180°F bottomhole temperature
    ...     tubing_ID=2.992,             # 2.992" tubing ID
    ...     rod_diameter=1.25,           # 1.25" rod diameter
    ...     Wr=2.5,                      # 2.5 lbf/ft rod weight
    ...     pump_depth_ft=5000,          # 5000 ft pump depth
    ...     oil_rate=100,                # 100 STB/D oil rate
    ...     liquid_rate=300,             # 300 STB/D liquid rate
    ...     pump_capacity=0.5,           # 0.5 bbl/d/rpm capacity
    ...     rotor_diameter=3.0,          # 3.0" rotor diameter
    ...     pump_efficiency=0.85         # 85% pump efficiency
    ... )
    """
    from utpgetools.utilities_package import oil_properties_calculation, gas_properties_calculation, two_phase_flow

    oil_properties = oil_properties_calculation(API,
                                                gas_gravity,
                                                water_gravity,
                                                GLR,
                                                WOR,
                                                pwf,
                                                BHT,
                                                pressure_increment_psi=100,
                                                separator_temperature_f=separator_temperature,
                                                separator_pressure_psi=separator_pressure
                                                )

    gas_properties = gas_properties_calculation(gravity=gas_gravity,
                                                pressure_psi=pwf,
                                                temperature_f=BHT
                                                )


    Rs = oil_properties['rs_scf_bbl'][-1]
    console.print(f"\nDissolved gas-oil ratio [bright_cyan]Rs[/bright_cyan] at pump intake: [bright_red]{Rs:.2f}[/bright_red] scf/bbl")
    z = gas_properties['z_factors'][-1]
    console.print(f"\nGas compressibility factor [bright_cyan]z[/bright_cyan] at pump intake: [bright_red]{z:.3f}[/bright_red]")
    dissolved_GLR = oil_rate * Rs / liquid_rate # this gets used for the two phase flow pressure calculation
    console.print(f"\nDissolved [bright_cyan]GLR[/bright_cyan] at pump intake: [bright_red]{dissolved_GLR:.2f}[/bright_red] scf/stb")
    if t_surface is None:
        t_surface = separator_temperature
    if wellhead_pressure is None:
        wellhead_pressure = separator_pressure
    depths, pressures = two_phase_flow(diameter_in=tubing_ID-rod_diameter,
                                       total_length_ft=pump_depth_ft,
                                       gas_liquid_ratio_scf_stb=dissolved_GLR,
                                       water_oil_ratio_stb_stb=WOR,
                                       oil_gravity_api=API,
                                       gas_gravity=gas_gravity,
                                       water_gravity=water_gravity,
                                       separator_temperature_f=separator_temperature,
                                       separator_pressure_psi=separator_pressure,
                                       oil_flowrate_stb_d=oil_rate,
                                       surface_temperature_f=t_surface,
                                       bottom_temperature_f=BHT,
                                       wellhead_pressure_psi=wellhead_pressure,
                                       length_increment_ft=500
                                       )
    discharge_pressure = pressures[-1]
    console.print(f"\nCalculated discharge pressure at pump depth: [bright_red]{discharge_pressure:.2f}[/bright_red] psi")

    # pump capacity should be bbl/d/rpm
    # pump efficiency comes from the curve on the slides of efficiency vs percent of max pressure
    if pump_efficiency is None:
        console.print(f"\nUse the discharge pressure and provide pump efficiency to continue")
        return
    pump_speed = liquid_rate * 0.4 / pump_efficiency / pump_capacity
    console.print(f"\nRequired pump speed: [bright_red]{pump_speed:.0f}[/bright_red] rpm")
    console.print(f"\n[bright_cyan]pump_speed[/bright_cyan] = liquid_rate * 0.4 / pump_efficiency / pump_capacity")

    T = 0.0894 * ((liquid_rate / pump_efficiency) * (discharge_pressure - pwf)) / (pump_speed * 0.8)
    console.print(f"\nRequired pump torque: [bright_red]{T:.2f}[/bright_red] ft-lb")
    console.print(f"\n[bright_cyan]T[/bright_cyan] = 0.0894 * ((liquid_rate / pump_efficiency) * (discharge_pressure - pwf)) / (pump_speed * 0.8)")
    
    BHPin = ((liquid_rate / pump_efficiency) * (discharge_pressure - pwf)) / (0.8 * 58771)
    console.print(f"\nRequired pump BHP: [bright_red]{BHPin:.2f}[/bright_red] hp")
    console.print(f"\n[bright_cyan]BHPin[/bright_cyan] = ((liquid_rate / pump_efficiency) * (discharge_pressure - pwf)) / (0.8 * 58771)")

    HHPout = 1.7 * 10**(-5) * 250 * (discharge_pressure - pwf)
    console.print(f"\nPump HHP: [bright_red]{HHPout:.2f}[/bright_red] hp")
    console.print(f"\n[bright_cyan]HHPout[/bright_cyan] = 1.7 * 10^(-5) * 250 * (discharge_pressure - pwf)")

    epcp = HHPout / BHPin * 100
    console.print(f"\nPump efficiency: [bright_red]{epcp:.2f}[/bright_red] %")

    # gamma_l is the weighted average of oil and water specific gravities
    oil_sg = 141.5 / (131.5 + API)
    gamma_l = oil_sg * (1 - (WOR / (1 + WOR))) + water_gravity * (WOR / (1 + WOR))
    console.print(f"\nLiquid specific gravity [bright_cyan]gamma_l[/bright_cyan]: [bright_red]{gamma_l:.3f}[/bright_red]")

    Fr = pump_depth_ft * Wr * (1 - 0.127 * gamma_l)
    console.print(f"\nBuoyed rod load [bright_cyan]Fr[/bright_cyan]: [bright_red]{Fr:.2f}[/bright_red] lbf")


    Fb = 9/16 * np.pi * rotor_diameter**2 * (discharge_pressure - pwf)
    console.print(f"\nPump thrust [bright_cyan]Fb[/bright_cyan]: [bright_red]{Fb:.2f}[/bright_red] lbf")
    console.print(f"\n[bright_cyan]Fb[/bright_cyan] = 9/16 * pi * d^2 * (discharge_pressure - pwf)")

    von_mises_stress = 4 / np.pi / rod_diameter**3 * np.sqrt((Fr + Fb)**2 * rod_diameter**2 + 48 * 144 * T**2)
    console.print(f"\nVon Mises stress : [bright_red]{von_mises_stress:.2f}[/bright_red] psi")
    console.print(f"\n[bright_cyan]Von Mises stress[/bright_cyan] = 4 / pi / d^3 * sqrt((Fr + Fb)^2 * d^2 + 48 * 144 * T^2)")

    bearing_life = (bearing_load_rating / (Fr + Fb))**(10/3) * (lifetime_revolutions / pump_speed) / 1440 / 365
    console.print(f"\nEstimated bearing life: [bright_red]{bearing_life:.2f}[/bright_red] years")
    console.print(f"\n[bright_cyan]bearing_life[/bright_cyan] = (bearing_load_rating / (Fr + Fb))**(10/3) * (lifetime_revolutions / pump_speed) / 1440 / 365")
    return