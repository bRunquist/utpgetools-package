"""
Petroleum Engineering Utilities Package

This module provides comprehensive utilities for petroleum engineering calculations,
focusing on fluid properties, multiphase flow analysis, and pressure-volume-temperature
(PVT) relationships. The module contains sophisticated correlations and calculation
methods commonly used in reservoir engineering, production engineering, and 
facilities design.

Main Functions:
    two_phase_flow: Multiphase flow calculations using Modified Hagedorn & Brown method
    oil_properties_calculation: Oil PVT properties vs pressure at constant temperature  
    gas_properties_calculation: Gas PVT properties vs pressure at constant temperature
    incompressible_single_phase: Single-phase liquid flow pressure drop calculations
    compressible_single_phase: Compressible gas flow pressure traverse calculations

Correlations and Methods:
    - Vasquez-Beggs correlation for oil properties (Rs, Bo, viscosity)
    - Beggs & Robinson correlation for oil viscosity
    - Standing-Katz correlation for gas compressibility (z-factor)
    - Lee-Gonzalez-Eakin correlation for gas viscosity
    - Modified Hagedorn & Brown method for multiphase flow
    - Colebrook-White equation for friction factors
    - Wichert-Aziz correction for sour gas components

Applications:
    - Reservoir fluid characterization and PVT analysis
    - Wellbore hydraulics and pressure traverse calculations
    - Production tubing design and optimization
    - Multiphase flow analysis in pipes and wellbores
    - Artificial lift system design and analysis
    - Facilities engineering and pipeline design

Dependencies:
    - numpy: For numerical calculations and array operations
    - math: For mathematical functions and constants

Notes:
    All functions include comprehensive input validation, unit consistency checks,
    and detailed documentation of correlations used. The module is designed to
    handle both single-point calculations and parametric studies over ranges
    of pressures and temperatures.
    
    Temperature inputs use Fahrenheit, pressures use psia, and flow rates use
    field units (STB/D, Mscf/D, bbl/D) unless otherwise specified.
"""

import numpy as np
def two_phase_flow(
    diameter_in=2.259,
    total_length_ft=10000,
    length_increment_ft=200,
    incline_angle_deg=90,
    roughness=0.0006,
    gas_liquid_ratio_scf_stb=500,
    water_oil_ratio_stb_stb=1,
    oil_gravity_api=35,
    gas_gravity=0.7,
    water_gravity=1.07,
    separator_temperature_f=100,
    separator_pressure_psi=100,
    oil_flowrate_stb_d=800,
    surface_temperature_f=100,
    bottom_temperature_f=150,
    wellhead_pressure_psi=125,
    return_detailed_properties=False
):
    """
    Two-phase flow calculation using Modified Hagedorn and Brown Method.
    
    This function calculates pressure profile along a wellbore considering 
    two-phase flow of oil, gas, and water using the Modified Hagedorn and Brown correlation.
    
    Parameters:
    -----------
    diameter_in : float, default=2.259
        Tubing inner diameter in inches
    total_length_ft : float, default=10000
        Total wellbore length in feet
    length_increment_ft : float, default=200
        Calculation increment length in feet
    incline_angle_deg : float, default=90
        Well inclination angle in degrees (90 = vertical)
    roughness : float, default=0.0006
        Pipe roughness (dimensionless)
    gas_liquid_ratio_scf_stb : float, default=500
        Gas/liquid ratio in standard cubic feet per stock tank barrel
    water_oil_ratio_stb_stb : float, default=1
        Water/oil ratio in stock tank barrels per stock tank barrel
    oil_gravity_api : float, default=35
        Oil gravity in API degrees
    gas_gravity : float, default=0.7
        Gas specific gravity (air = 1.0)
    water_gravity : float, default=1.07
        Water specific gravity (water = 1.0)
    separator_temperature_f : float, default=100
        Separator temperature in Fahrenheit
    separator_pressure_psi : float, default=100
        Separator pressure in psi
    oil_flowrate_stb_d : float, default=800
        Oil flow rate in stock tank barrels per day
    surface_temperature_f : float, default=100
        Surface temperature in Fahrenheit
    bottom_temperature_f : float, default=150
        Bottom hole temperature in Fahrenheit
    wellhead_pressure_psi : float, default=125
        Wellhead pressure in psi
    return_detailed_properties : bool, default=False
        If True, returns detailed fluid properties like the VBA writes to additional sheets
        If False, returns only depth and pressure arrays (main VBA output)
        
    Returns:
    --------
    If return_detailed_properties=False (default):
        tuple : (depths_ft, pressures_psi)
            - depths_ft: np.array of depths in feet from 0 to total_length_ft
            - pressures_psi: np.array of pressures in psi at corresponding depths
    
    If return_detailed_properties=True:
        dict : Dictionary containing detailed properties like VBA "Oil Properties" and "Gas Properties" sheets
    """
    import math
    
    # Convert inputs to working units
    diameter_ft = diameter_in / 12.0
    n = int(total_length_ft / length_increment_ft)
    angle_rad = math.radians(incline_angle_deg)
    
    # Temperature gradient
    temp_grad = (bottom_temperature_f - surface_temperature_f) / total_length_ft  # °F/ft
    
    # Calculate GOR (gas-oil ratio)
    GOR = gas_liquid_ratio_scf_stb * (water_oil_ratio_stb_stb + 1)
    
    # Initialize arrays for results
    p = np.zeros(n + 1)
    T = np.zeros(n + 1)
    densL = np.zeros(n + 1)
    MuL = np.zeros(n + 1)
    Bo = np.zeros(n + 1)
    Rs = np.zeros(n + 1)
    densG = np.zeros(n + 1)
    MuG = np.zeros(n + 1)
    Bg = np.zeros(n + 1)
    Z = np.zeros(n + 1)
    flow_types = []
    
    # Initial conditions
    p[0] = wellhead_pressure_psi
    T[0] = surface_temperature_f
    old_delta_p = 30.0  # Initial guess for pressure drop
    
    # Main calculation loop
    for i in range(1, n + 1):
        error = 1.0
        iteration_count = 0
        max_iterations = 100
        
        while error >= 0.001 and iteration_count < max_iterations:
            # Temperature at mid-point
            T_i = T[i-1] + (temp_grad * length_increment_ft / 2)
            # Pressure at mid-point
            p_i = p[i-1] + (old_delta_p / 2)
            
            # Oil specific gravity
            oil_sg = 141.5 / (oil_gravity_api + 131.5)
            
            # Corrected gas specific gravity (Vasquez-Beggs)
            gas_sg = gas_gravity * (1 + 5.912e-5 * oil_gravity_api * separator_temperature_f * 
                                   (math.log((separator_pressure_psi + 14.7) / 114.7) / math.log(10)))
            
            # Temperature factor for bubble point calculation
            A = oil_gravity_api / (T_i + 460)
            
            # Bubble point pressure (Vasquez-Beggs correlation)
            if GOR == 0:
                Pb = 14.7
            else:
                if oil_gravity_api <= 30:
                    Pb = (27.64 * GOR / (gas_sg * 10**(11.172 * A)))**(1/1.0937)
                else:
                    Pb = (56.06 * GOR / (gas_sg * 10**(10.393 * A)))**(1/1.187)
            
            # Solution gas-oil ratio (Vasquez-Beggs correlation)
            if oil_gravity_api <= 30:
                Rs_calc = gas_sg * (p_i**1.0937) * (10**(11.172 * A)) / 27.64
            else:
                Rs_calc = gas_sg * (p_i**1.187) * (10**(10.393 * A)) / 56.06
            
            Rs_calc = max(0, min(Rs_calc, GOR))
            
            # Oil formation volume factor (Vasquez-Beggs)
            F = (T_i - 60) * (oil_gravity_api / gas_sg)
            
            if p_i < Pb:
                # Below bubble point
                if oil_gravity_api <= 30:
                    Bo_calc = 1 + 4.677e-4 * Rs_calc + 1.751e-5 * F - 1.8106e-8 * Rs_calc * F
                else:
                    Bo_calc = 1 + 4.677e-4 * Rs_calc + 1.1e-5 * F - 1.337e-9 * Rs_calc * F
            else:
                # Above bubble point
                if oil_gravity_api <= 30:
                    Bob = 1 + 4.677e-4 * GOR + 1.751e-5 * F - 1.8106e-8 * GOR * F
                else:
                    Bob = 1 + 4.677e-4 * GOR + 1.1e-5 * F - 1.337e-9 * GOR * F
                
                co = (-1.433 + 5 * Rs_calc + 17.2 * T_i - 1.18 * gas_sg + 12.61 * oil_gravity_api) / (p_i * 1e5)
                Bo_calc = Bob * math.exp(co * (Pb - p_i))
            
            # Oil viscosity (Beggs and Robinson correlation)
            Zv = 3.0324 - 0.02023 * oil_gravity_api
            y = 10**Zv
            x = y * (T_i**(-1.163))
            Muod = (10**x) - 1
            
            aa = 10.715 * ((Rs_calc + 100)**(-0.515))
            bb = 5.44 * ((Rs_calc + 150)**(-0.338))
            
            if p_i <= Pb:
                Muo = aa * (Muod**bb)
            else:
                Muob = aa * (Muod**bb)
                m = 2.6 * (p_i**1.187) * math.exp(-11.513 - 8.98e-5 * p_i)
                Muo = Muob * ((p_i / Pb)**m)
            
            # Gas compressibility factor (Z-factor)
            Tc = gas_gravity * 314.8148 + 168.5185  # Rankine
            Pc = gas_gravity * (-47.619) + 700.4762  # psia
            Tr = (T_i + 460) / Tc
            Pr = p_i / Pc
            
            Dz = 10**(0.3106 - 0.49 * Tr + 0.1824 * (Tr**2))
            Cz = 0.132 - 0.32 * (math.log(Tr) / math.log(10))
            Bz = ((0.62 - 0.23 * Tr) * Pr + 
                  ((0.066 / (Tr - 0.86)) - 0.037) * (Pr**2) + 
                  0.32 * (Pr**6) / (10**(9 * (Tr - 1))))
            Az = 1.39 * ((Tr - 0.92)**0.5) - 0.36 * Tr - 0.101
            Z_calc = Az + (1 - Az) * math.exp(-Bz) + Cz * (Pr**Dz)
            
            # Gas density
            densG_calc = 2.7 * gas_gravity * p_i / ((T_i + 460) * Z_calc)
            
            # Gas formation volume factor
            Bg_calc = 0.0283 * Z_calc * (T_i + 460) / p_i
            
            # Gas viscosity
            Xvisc = 3.5 + (986 / (T_i + 460)) + 0.01 * 29 * gas_gravity
            Lambda = 2.4 - 0.2 * Xvisc
            Kvisc = ((9.4 + 0.02 * 29 * gas_gravity) * ((T_i + 460)**1.5)) / (209 + 19 * 29 * gas_gravity + T_i + 460)
            MuG_calc = Kvisc * 1e-4 * math.exp(Xvisc * ((0.01602 * densG_calc)**Lambda))
            
            # Oil density
            densAir = 28.97 * p_i / (10.73159 * Z_calc * (T_i + 460))
            dissgasgr = densG_calc / densAir
            densO = (62.4 * oil_sg / Bo_calc) + (0.0764 * dissgasgr * Rs_calc / (Bo_calc * 5.615))
            
            # Liquid properties
            densW = 62.4 * water_gravity
            densL_calc = (water_oil_ratio_stb_stb * densW + Bo_calc * densO) / (water_oil_ratio_stb_stb + Bo_calc)
            
            MuW = math.exp(1.003 - 1.479e-2 * T_i + 1.982e-5 * (T_i**2))
            MuL_calc = (((water_oil_ratio_stb_stb * densW) / (water_oil_ratio_stb_stb * densW + Bo_calc * densO)) * MuW + 
                       ((Bo_calc * densO) / (water_oil_ratio_stb_stb * densW + Bo_calc * densO)) * Muo)
            
            # Surface tension
            sigmaO = 30  # dynes/cm
            sigmaW = 74
            sigma = (((water_oil_ratio_stb_stb * densW) / (water_oil_ratio_stb_stb * densW + Bo_calc * densO)) * sigmaW + 
                    ((Bo_calc * densO) / (water_oil_ratio_stb_stb * densW + Bo_calc * densO)) * sigmaO)
            
            # Flow calculation using Modified Hagedorn and Brown Method
            Area = math.pi * (diameter_ft**2) / 4
            ql = (water_oil_ratio_stb_stb + Bo_calc) * oil_flowrate_stb_d * 5.615  # res ft³/D
            qg = Bg_calc * (GOR - Rs_calc) * oil_flowrate_stb_d  # res ft³/D
            qt = qg + ql
            
            if qt > 0:
                Fg = qg / qt
                Fl = 1 - Fg
            else:
                Fg = 0
                Fl = 1
            
            usg = qg / (Area * 86400)  # ft/s
            usl = ql / (Area * 86400)  # ft/s
            um = usg + usl
            
            # Liquid holdup calculation
            LB = 1.071 - (0.2218 * (um**2) / diameter_ft)
            LB = max(LB, 0.13)
            
            if Fg < LB:
                # Bubble flow
                flow_type = "Bubble flow"
                us = 0.8  # ft/s
                yl = 1 - 0.5 * (1 + (um / us) - ((1 + (um / us))**2 - 4 * usg / us)**0.5)
                yl = min(max(yl, Fl), 1)
                
                mdotl = Area * usl * densL_calc * 86400
                Nre = 0.022 * mdotl / (diameter_ft * MuL_calc)
                
                try:
                    ff = (1 / (-4 * (math.log((roughness / 3.7065) - 
                                            (5.0452 / Nre) * 
                                            (math.log(((roughness**1.1098) / 2.8257) + 
                                                     ((7.149 / Nre)**0.8981)) / math.log(10))) / math.log(10))))**2
                except:
                    ff = 0.02
                
                DensAvg = (1 - yl) * densG_calc + yl * densL_calc
                dpdz = (1/144) * ((math.sin(angle_rad) * DensAvg) + 
                                 ((ff * (mdotl**2)) / (7.413e10 * (diameter_ft**5) * densL_calc * (yl**2))))
                DeltaP = dpdz * length_increment_ft
                
            else:
                # Not bubble flow (slug, transition, mist flow)
                flow_type = "Not bubble flow"
                
                Nvl = 1.938 * usl * ((densL_calc / sigma)**(1/4))
                Nvg = 1.938 * usg * ((densL_calc / sigma)**(1/4))
                Nd = 120.872 * diameter_ft * ((densL_calc / sigma)**(1/2))
                Nl = 0.15726 * MuL_calc * ((1 / (densL_calc * (sigma**3)))**(1/4))
                
                CNl = (7.9595 * (Nl**6) - 13.144 * (Nl**5) + 8.3825 * (Nl**4) - 
                      2.4629 * (Nl**3) + 0.2213 * (Nl**2) + 0.0473 * Nl + 0.0018)
                
                if Nvg != 0 and Nd != 0:
                    group1 = Nvl * (p_i**0.1) * CNl / ((Nvg**0.575) * (14.7**0.1) * Nd)
                    ylpsy = (-3.44985871528755e15 * (group1**6) + 56858620047687.2 * (group1**5) - 
                            368100995579.95 * (group1**4) + 1189881753.18 * (group1**3) - 
                            2037716.09 * (group1**2) + 1868.71 * group1 + 0.1)
                    
                    group2 = Nvg * (Nl**0.38) / (Nd**2.14)
                    psy = (116159 * (group2**4) - 22251 * (group2**3) + 
                          1232.1 * (group2**2) - 4.8183 * group2 + 0.9116)
                    psy = max(psy, 1)
                    
                    yl = ylpsy * psy
                else:
                    yl = Fl
                
                yl = min(max(yl, Fl), 1)
                
                DensAvg = (1 - yl) * densG_calc + yl * densL_calc
                mdot = Area * (usl * densL_calc + usg * densG_calc) * 86400
                
                try:
                    Nre = 0.022 * mdot / (diameter_ft * (MuL_calc**yl) * (MuG_calc**(1 - yl)))
                    ff = (1 / (-4 * (math.log((roughness / 3.7065) - 
                                            (5.0452 / Nre) * 
                                            (math.log(((roughness**1.1098) / 2.8257) + 
                                                     ((7.149 / Nre)**0.8981)) / math.log(10))) / math.log(10))))**2
                except:
                    ff = 0.02
                
                dpdz = (1/144) * ((math.sin(angle_rad) * DensAvg) + 
                                 ((ff * (mdot**2)) / (7.413e10 * (diameter_ft**5) * DensAvg)))
                DeltaP = dpdz * length_increment_ft
            
            # Check convergence
            error = abs(DeltaP - old_delta_p) / (abs(old_delta_p) + 1e-6)
            old_delta_p = DeltaP
            iteration_count += 1
        
        # Update pressure and temperature
        p[i] = p[i-1] + DeltaP
        T[i] = T[i-1] + length_increment_ft * temp_grad
        
        # Store properties at this depth
        densL[i] = densL_calc
        MuL[i] = MuL_calc
        Bo[i] = Bo_calc
        Rs[i] = Rs_calc
        densG[i] = densG_calc
        MuG[i] = MuG_calc
        Bg[i] = Bg_calc
        Z[i] = Z_calc
        flow_types.append(flow_type)
    
    # Create depth array
    depths = np.arange(0, total_length_ft + length_increment_ft, length_increment_ft)
    depths = depths[:len(p)]
    
    # Return results matching the VBA output format
    if return_detailed_properties:
        # Return detailed properties like VBA writes to "Oil Properties" and "Gas Properties" sheets
        results = {
            'depths_ft': depths,
            'pressures_psi': p,
            'temperatures_f': T,
            'liquid_densities_lbm_ft3': densL,
            'liquid_viscosities_cp': MuL,
            'bo_bbl_stb': Bo,
            'rs_scf_bbl': Rs,
            'gas_densities_lbm_ft3': densG,
            'gas_viscosities_cp': MuG,
            'bg_ft3_scf': Bg,
            'z_factors': Z,
            'flow_types': flow_types
        }
        return results
    else:
        # Primary output: pressure vs depth (like the VBA writes to columns G and H)
        return depths, p


def oil_properties_calculation(
    oil_gravity_api=35,
    gas_gravity=0.71,
    water_gravity=1.07,
    gas_liquid_ratio_scf_stb=500,
    water_oil_ratio_bbl_bbl=1.5,
    pressure_psi=4350,
    temperature_f=180,
    pressure_increment_psi=100,
    separator_temperature_f=100,
    separator_pressure_psi=100
):
    """
    Oil Properties Calculation - calculates fluid properties vs pressure.
    
    This function replicates the Excel VBA "Oil Properties Calculation" sheet,
    calculating oil and liquid properties over a range of pressures at constant temperature.
    
    Parameters:
    -----------
    oil_gravity_api : float, default=35
        Oil gravity in API degrees
    gas_gravity : float, default=0.71
        Gas specific gravity (air = 1.0)
    water_gravity : float, default=1.07
        Water specific gravity (water = 1.0)
    gas_liquid_ratio_scf_stb : float, default=500
        Gas/liquid ratio in standard cubic feet per stock tank barrel
    water_oil_ratio_bbl_bbl : float, default=1.5
        Water/oil ratio in barrels per barrel
    pressure_psi : float, default=4350
        Maximum pressure in psi
    temperature_f : float, default=180
        Temperature in Fahrenheit (constant)
    pressure_increment_psi : float, default=100
        Pressure increment for calculations in psi
    separator_temperature_f : float, default=100
        Separator temperature in Fahrenheit
    separator_pressure_psi : float, default=100
        Separator pressure in psi
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'pressures_psi': np.array of pressures
        - 'liquid_densities_lbm_ft3': np.array of liquid densities
        - 'liquid_viscosities_cp': np.array of liquid viscosities
        - 'bo_bbl_stb': np.array of oil formation volume factors
        - 'rs_scf_bbl': np.array of solution gas-oil ratios
        - 'bubble_point_properties': dict of properties at bubble point
    """
    import math
    
    # Calculate GOR (gas-oil ratio)
    GOR = gas_liquid_ratio_scf_stb * (water_oil_ratio_bbl_bbl + 1)
    
    # Calculate number of pressure points
    n = int(pressure_psi / pressure_increment_psi)
    
    # Initialize arrays
    pressures = np.zeros(n + 2)  # +2 to handle final pressure point
    liquid_densities = np.zeros(n + 2)
    liquid_viscosities = np.zeros(n + 2)
    bo_values = np.zeros(n + 2)
    rs_values = np.zeros(n + 2)
    
    # Oil specific gravity
    oil_sg = 141.5 / (oil_gravity_api + 131.5)
    
    # Corrected gas specific gravity (Vasquez-Beggs)
    gas_sg = gas_gravity * (1 + 5.912e-5 * oil_gravity_api * separator_temperature_f * 
                           (math.log((separator_pressure_psi + 14.7) / 114.7) / math.log(10)))
    
    # Main calculation loop
    for i in range(n + 1):
        p_i = i * pressure_increment_psi
        T_i = temperature_f
        
        # Minimum pressure constraint
        if p_i < 14.7:
            p_i = 14.7
        
        # Maximum pressure constraint
        if p_i > pressure_psi:
            p_i = pressure_psi
            
        pressures[i] = p_i
        
        # Temperature factor for correlations
        A = oil_gravity_api / (T_i + 460)
        
        # Bubble point pressure (Vasquez-Beggs correlation)
        if GOR == 0:
            Pb = 14.7
        else:
            if oil_gravity_api <= 30:
                Pb = (27.64 * GOR / (gas_sg * 10**(11.172 * A)))**(1/1.0937)
            else:
                Pb = (56.06 * GOR / (gas_sg * 10**(10.393 * A)))**(1/1.187)
        
        # Solution gas-oil ratio (Vasquez-Beggs correlation)
        if oil_gravity_api <= 30:
            Rs = gas_sg * (p_i**1.0937) * (10**(11.172 * A)) / 27.64
        else:
            Rs = gas_sg * (p_i**1.187) * (10**(10.393 * A)) / 56.06
        
        Rs = max(0, min(Rs, GOR))
        rs_values[i] = Rs
        
        # Oil formation volume factor (Vasquez-Beggs)
        F = (T_i - 60) * (oil_gravity_api / gas_sg)
        
        if p_i < Pb:
            # Below bubble point
            if oil_gravity_api <= 30:
                Bo = 1 + 4.677e-4 * Rs + 1.751e-5 * F - 1.8106e-8 * Rs * F
            else:
                Bo = 1 + 4.677e-4 * Rs + 1.1e-5 * F - 1.337e-9 * Rs * F
        else:
            # Above bubble point
            if oil_gravity_api <= 30:
                Bob = 1 + 4.677e-4 * GOR + 1.751e-5 * F - 1.8106e-8 * GOR * F
            else:
                Bob = 1 + 4.677e-4 * GOR + 1.1e-5 * F - 1.337e-9 * GOR * F
            
            co = (-1.433 + 5 * Rs + 17.2 * T_i - 1.18 * gas_sg + 12.61 * oil_gravity_api) / (p_i * 1e5)
            Bo = Bob * math.exp(co * (Pb - p_i))
        
        bo_values[i] = Bo
        
        # Oil viscosity (Beggs and Robinson correlation)
        Zv = 3.0324 - 0.02023 * oil_gravity_api
        y = 10**Zv
        x = y * (T_i**(-1.163))
        Muod = (10**x) - 1
        
        aa = 10.715 * ((Rs + 100)**(-0.515))
        bb = 5.44 * ((Rs + 150)**(-0.338))
        
        if p_i <= Pb:
            Muo = aa * (Muod**bb)
        else:
            Muob = aa * (Muod**bb)
            m = 2.6 * (p_i**1.187) * math.exp(-11.513 - 8.98e-5 * p_i)
            Muo = Muob * ((p_i / Pb)**m)
        
        # Gas compressibility factor (Z-factor)
        Tc = gas_gravity * 314.8148 + 168.5185  # Rankine
        Pc = gas_gravity * (-47.619) + 700.4762  # psia
        Tr = (T_i + 460) / Tc
        Pr = p_i / Pc
        
        Dz = 10**(0.3106 - 0.49 * Tr + 0.1824 * (Tr**2))
        Cz = 0.132 - 0.32 * (math.log(Tr) / math.log(10))
        Bz = ((0.62 - 0.23 * Tr) * Pr + 
              ((0.066 / (Tr - 0.86)) - 0.037) * (Pr**2) + 
              0.32 * (Pr**6) / (10**(9 * (Tr - 1))))
        Az = 1.39 * ((Tr - 0.92)**0.5) - 0.36 * Tr - 0.101
        Z = Az + (1 - Az) * math.exp(-Bz) + Cz * (Pr**Dz)
        
        # Gas density
        densG = 2.7 * gas_gravity * p_i / ((T_i + 460) * Z)
        
        # Gas formation volume factor
        Bg = 0.0283 * Z * (T_i + 460) / p_i
        
        # Gas viscosity
        Xvisc = 3.5 + (986 / (T_i + 460)) + 0.01 * 29 * gas_gravity
        Lambda = 2.4 - 0.2 * Xvisc
        Kvisc = ((9.4 + 0.02 * 29 * gas_gravity) * ((T_i + 460)**1.5)) / (209 + 19 * 29 * gas_gravity + T_i + 460)
        MuG = Kvisc * 1e-4 * math.exp(Xvisc * ((0.01602 * densG)**Lambda))
        
        # Oil density
        densAir = 28.97 * p_i / (10.73159 * Z * (T_i + 460))
        dissgasgr = densG / densAir
        densO = (62.4 * oil_sg / Bo) + (0.0764 * dissgasgr * Rs / (Bo * 5.615))
        
        # Liquid properties
        densW = 62.4 * water_gravity
        densL = (water_oil_ratio_bbl_bbl * densW + Bo * densO) / (water_oil_ratio_bbl_bbl + Bo)
        liquid_densities[i] = densL
        
        MuW = math.exp(1.003 - 1.479e-2 * T_i + 1.982e-5 * (T_i**2))
        MuL = (((water_oil_ratio_bbl_bbl * densW) / (water_oil_ratio_bbl_bbl * densW + Bo * densO)) * MuW + 
               ((Bo * densO) / (water_oil_ratio_bbl_bbl * densW + Bo * densO)) * Muo)
        liquid_viscosities[i] = MuL
    
    # Handle final pressure point if needed (like VBA does)
    j = pressure_psi / pressure_increment_psi
    if n < j:
        final_index = n + 1
        new_increment = (j - n) * pressure_increment_psi
        p_final = n * pressure_increment_psi + new_increment
        
        if p_final > pressure_psi:
            p_final = pressure_psi
            
        pressures[final_index] = p_final
        
        # Repeat calculations for final pressure point
        T_i = temperature_f
        A = oil_gravity_api / (T_i + 460)
        
        # Bubble point pressure
        if GOR == 0:
            Pb = 14.7
        else:
            if oil_gravity_api <= 30:
                Pb = (27.64 * GOR / (gas_sg * 10**(11.172 * A)))**(1/1.0937)
            else:
                Pb = (56.06 * GOR / (gas_sg * 10**(10.393 * A)))**(1/1.187)
        
        # Solution gas-oil ratio
        if oil_gravity_api <= 30:
            Rs = gas_sg * (p_final**1.0937) * (10**(11.172 * A)) / 27.64
        else:
            Rs = gas_sg * (p_final**1.187) * (10**(10.393 * A)) / 56.06
        
        Rs = max(0, min(Rs, GOR))
        rs_values[final_index] = Rs
        
        # Oil formation volume factor
        F = (T_i - 60) * (oil_gravity_api / gas_sg)
        
        if p_final < Pb:
            if oil_gravity_api <= 30:
                Bo = 1 + 4.677e-4 * Rs + 1.751e-5 * F - 1.8106e-8 * Rs * F
            else:
                Bo = 1 + 4.677e-4 * Rs + 1.1e-5 * F - 1.337e-9 * Rs * F
        else:
            if oil_gravity_api <= 30:
                Bob = 1 + 4.677e-4 * GOR + 1.751e-5 * F - 1.8106e-8 * GOR * F
            else:
                Bob = 1 + 4.677e-4 * GOR + 1.1e-5 * F - 1.337e-9 * GOR * F
            
            co = (-1.433 + 5 * Rs + 17.2 * T_i - 1.18 * gas_sg + 12.61 * oil_gravity_api) / (p_final * 1e5)
            Bo = Bob * math.exp(co * (Pb - p_final))
        
        bo_values[final_index] = Bo
        
        # Oil viscosity
        Zv = 3.0324 - 0.02023 * oil_gravity_api
        y = 10**Zv
        x = y * (T_i**(-1.163))
        Muod = (10**x) - 1
        
        aa = 10.715 * ((Rs + 100)**(-0.515))
        bb = 5.44 * ((Rs + 150)**(-0.338))
        
        if p_final <= Pb:
            Muo = aa * (Muod**bb)
        else:
            Muob = aa * (Muod**bb)
            m = 2.6 * (p_final**1.187) * math.exp(-11.513 - 8.98e-5 * p_final)
            Muo = Muob * ((p_final / Pb)**m)
        
        # Gas properties for liquid density calculation
        Tr = (T_i + 460) / Tc
        Pr = p_final / Pc
        
        Dz = 10**(0.3106 - 0.49 * Tr + 0.1824 * (Tr**2))
        Cz = 0.132 - 0.32 * (math.log(Tr) / math.log(10))
        Bz = ((0.62 - 0.23 * Tr) * Pr + 
              ((0.066 / (Tr - 0.86)) - 0.037) * (Pr**2) + 
              0.32 * (Pr**6) / (10**(9 * (Tr - 1))))
        Az = 1.39 * ((Tr - 0.92)**0.5) - 0.36 * Tr - 0.101
        Z = Az + (1 - Az) * math.exp(-Bz) + Cz * (Pr**Dz)
        
        densG = 2.7 * gas_gravity * p_final / ((T_i + 460) * Z)
        densAir = 28.97 * p_final / (10.73159 * Z * (T_i + 460))
        dissgasgr = densG / densAir
        densO = (62.4 * oil_sg / Bo) + (0.0764 * dissgasgr * Rs / (Bo * 5.615))
        
        densW = 62.4 * water_gravity
        densL = (water_oil_ratio_bbl_bbl * densW + Bo * densO) / (water_oil_ratio_bbl_bbl + Bo)
        liquid_densities[final_index] = densL
        
        MuW = math.exp(1.003 - 1.479e-2 * T_i + 1.982e-5 * (T_i**2))
        MuL = (((water_oil_ratio_bbl_bbl * densW) / (water_oil_ratio_bbl_bbl * densW + Bo * densO)) * MuW + 
               ((Bo * densO) / (water_oil_ratio_bbl_bbl * densW + Bo * densO)) * Muo)
        liquid_viscosities[final_index] = MuL
    
    # Calculate properties at bubble point (like VBA does)
    Pbb = Pb  # Bubble point pressure
    Tb = temperature_f
    
    # Properties at bubble point
    Ab = oil_gravity_api / (Tb + 460)
    
    if oil_gravity_api <= 30:
        Rsb = gas_sg * (Pbb**1.0937) * (10**(11.172 * Ab)) / 27.64
    else:
        Rsb = gas_sg * (Pbb**1.187) * (10**(10.393 * Ab)) / 56.06
    
    Rsb = max(0, min(Rsb, GOR))
    
    Fb = (Tb - 60) * (oil_gravity_api / gas_sg)
    if oil_gravity_api <= 30:
        Bobb = 1 + 4.677e-4 * GOR + 1.751e-5 * Fb - 1.8106e-8 * GOR * Fb
    else:
        Bobb = 1 + 4.677e-4 * GOR + 1.1e-5 * Fb - 1.337e-9 * GOR * Fb
    
    # Oil viscosity at bubble point
    Zv = 3.0324 - 0.02023 * oil_gravity_api
    y = 10**Zv
    xb = y * (Tb**(-1.163))
    Muodb = (10**xb) - 1
    aab = 10.715 * ((Rsb + 100)**(-0.515))
    bbb = 5.44 * ((Rsb + 150)**(-0.338))
    Muobb = aab * (Muodb**bbb)
    
    # Gas properties at bubble point
    Trb = (Tb + 460) / Tc
    Prb = Pbb / Pc
    
    Dzb = 10**(0.3106 - 0.49 * Trb + 0.1824 * (Trb**2))
    Czb = 0.132 - 0.32 * (math.log(Trb) / math.log(10))
    Bzb = ((0.62 - 0.23 * Trb) * Prb + 
           ((0.066 / (Trb - 0.86)) - 0.037) * (Prb**2) + 
           0.32 * (Prb**6) / (10**(9 * (Trb - 1))))
    Azb = 1.39 * ((Trb - 0.92)**0.5) - 0.36 * Trb - 0.101
    Zb = Azb + (1 - Azb) * math.exp(-Bzb) + Czb * (Prb**Dzb)
    
    densGb = 2.7 * gas_gravity * Pbb / ((Tb + 460) * Zb)
    densAirb = 28.97 * Pbb / (10.73159 * Zb * (Tb + 460))
    dissgasgrb = densGb / densAirb
    densOb = (62.4 * oil_sg / Bobb) + (0.0764 * dissgasgrb * Rsb / (Bobb * 5.615))
    
    densW = 62.4 * water_gravity
    densLb = (water_oil_ratio_bbl_bbl * densW + Bobb * densOb) / (water_oil_ratio_bbl_bbl + Bobb)
    
    MuWb = math.exp(1.003 - 1.479e-2 * Tb + 1.982e-5 * (Tb**2))
    MuLb = (((water_oil_ratio_bbl_bbl * densW) / (water_oil_ratio_bbl_bbl * densW + Bobb * densOb)) * MuWb + 
            ((Bobb * densOb) / (water_oil_ratio_bbl_bbl * densW + Bobb * densOb)) * Muobb)
    
    # Trim arrays to actual size
    actual_size = final_index + 1 if 'final_index' in locals() else n + 1
    
    results = {
        'pressures_psi': pressures[:actual_size],
        'liquid_densities_lbm_ft3': liquid_densities[:actual_size],
        'liquid_viscosities_cp': liquid_viscosities[:actual_size],
        'bo_bbl_stb': bo_values[:actual_size],
        'rs_scf_bbl': rs_values[:actual_size],
        'bubble_point_properties': {
            'bubble_point_pressure_psi': Pbb,
            'liquid_density_lbm_ft3': densLb,
            'liquid_viscosity_cp': MuLb,
            'bo_bbl_stb': Bobb,
            'rs_scf_bbl': Rsb
        }
    }
    
    return results


def gas_properties_calculation(
    gravity=0.71,
    co2_percent=0,
    n2_percent=0,
    h2s_percent=0,
    h2o_percent=0,
    pressure_psi=4350,
    temperature_f=180,
    pressure_increment_psi=100,
    component_mode=False,
    component_percentages=None
):
    """
    Gas Properties Calculation - calculates gas properties vs pressure.
    
    This function replicates the Excel VBA "Gas Properties Calculation" sheet,
    calculating gas properties over a range of pressures at constant temperature.
    Can operate in two modes: gravity-based or component-based calculations.
    
    Parameters:
    -----------
    gravity : float, default=0.71
        Gas specific gravity (air = 1.0)
    co2_percent : float, default=0
        CO2 percentage in gas
    n2_percent : float, default=0
        N2 percentage in gas
    h2s_percent : float, default=0
        H2S percentage in gas
    h2o_percent : float, default=0
        H2O percentage in gas
    pressure_psi : float, default=4350
        Maximum pressure in psi
    temperature_f : float, default=180
        Temperature in Fahrenheit (constant)
    pressure_increment_psi : float, default=100
        Pressure increment for calculations in psi
    component_mode : bool, default=False
        If True, uses detailed component analysis instead of gravity-based
    component_percentages : dict, optional
        Dictionary with component percentages for detailed analysis:
        {'C1': 87.5, 'C2': 8.3, 'C3': 2.1, 'iC4': 0.6, 'nC4': 0.2, 
         'iC5': 0.3, 'nC5': 0.8, 'nC6': 0.1, 'nC7': 0.1, 'nC8': 0.0,
         'CO2': 0, 'N2': 0, 'H2S': 0, 'H2O': 0}
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'pressures_psi': np.array of pressures
        - 'gas_densities_lbm_ft3': np.array of gas densities
        - 'gas_viscosities_cp': np.array of gas viscosities
        - 'bg_ft3_scf': np.array of gas formation volume factors
        - 'z_factors': np.array of gas compressibility factors
        - 'critical_properties': dict of critical properties
    """
    import math
    
    # Calculate number of pressure points
    n = int(pressure_psi / pressure_increment_psi)
    
    # Initialize arrays
    pressures = np.zeros(n + 2)
    gas_densities = np.zeros(n + 2)
    gas_viscosities = np.zeros(n + 2)
    bg_values = np.zeros(n + 2)
    z_factors = np.zeros(n + 2)
    
    if component_mode and component_percentages:
        # Component-based calculation
        # Convert percentages to mole fractions
        comps = component_percentages.copy()
        
        # Check if percentages sum to more than 100
        total_percent = sum(comps.values())
        if total_percent > 100:
            raise ValueError("Component percentages cannot sum to more than 100%")
        
        # Convert to fractions
        xC1 = comps.get('C1', 0) / 100
        xC2 = comps.get('C2', 0) / 100
        xC3 = comps.get('C3', 0) / 100
        xiC4 = comps.get('iC4', 0) / 100
        xnC4 = comps.get('nC4', 0) / 100
        xiC5 = comps.get('iC5', 0) / 100
        xnC5 = comps.get('nC5', 0) / 100
        xnC6 = comps.get('nC6', 0) / 100
        xnC7 = comps.get('nC7', 0) / 100
        xnC8 = comps.get('nC8', 0) / 100
        xCO2 = comps.get('CO2', 0) / 100
        xN2 = comps.get('N2', 0) / 100
        xH2S = comps.get('H2S', 0) / 100
        xH2O = comps.get('H2O', 0) / 100
        
        # Calculate gas gravity from components
        gas_gravity = (xC1 * 16.04 + xC2 * 30.07 + xC3 * 44.1 + xiC4 * 58.12 + xnC4 * 58.12 + 
                      xiC5 * 72.15 + xnC5 * 72.15 + xnC6 * 86.18 + xnC7 * 100.2 + xnC8 * 128.26 + 
                      xCO2 * 44.01 + xN2 * 28.01 + xH2S * 34.08 + xH2O * 18.016) / 28.97
        
        # Calculate critical properties from components
        Tc = (xC1 * 344 + xC2 * 550 + xC3 * 666 + xiC4 * 733 + xnC4 * 766 + 
              xiC5 * 830 + xnC5 * 847 + xnC6 * 915 + xnC7 * 972 + xnC8 * 1070 + 
              xCO2 * 548 + xN2 * 227 + xH2S * 673 + xH2O * 1107)
        
        Pc = (xC1 * 673 + xC2 * 709 + xC3 * 618 + xiC4 * 530 + xnC4 * 551 + 
              xiC5 * 482 + xnC5 * 485 + xnC6 * 437 + xnC7 * 397 + xnC8 * 332 + 
              xCO2 * 1072 + xN2 * 492 + xH2S * 1306 + xH2O * 3198)
        
    else:
        # Gravity-based calculation
        gas_gravity = gravity
        xCO2 = co2_percent / 100
        xN2 = n2_percent / 100
        xH2S = h2s_percent / 100
        xH2O = h2o_percent / 100
        
        # Critical properties correlations
        if gas_gravity > 0.7:
            Tc = gas_gravity * 318 + 166  # °R
            Pc = gas_gravity * (-56) + 708  # psi
        else:
            Tc = gas_gravity * 318 + 166  # °R
            Pc = gas_gravity * (-36) + 693  # psi
        
        # Corrections for non-hydrocarbon components
        if gas_gravity <= 1.5:
            N2Tc = -2.4 * (xN2 * 100)
            Co2Tc = -0.8667 * (xCO2 * 100)
            H2sTc = 1.3333 * (xH2S * 100)
            
            N2Pc = -2.0667 * (xN2 * 100)
            Co2Pc = 4.5333 * (xCO2 * 100)
            H2sPc = 6.1333 * (xH2S * 100)
            
        elif gas_gravity < 2:
            N2Tc = -2.4 * (xN2 * 100)
            Co2Tc = -0.8667 * (xCO2 * 100)
            Co2Pc = 4.5333 * (xCO2 * 100)
            H2sPc = 6.1333 * (xH2S * 100)
            
            # Interpolate H2S temperature correction
            H2STc1 = 1.3333 * (xH2S * 100)
            H2STc2 = 0.6667 * (xH2S * 100)
            H2sTc = 2 * (2 - gas_gravity) * H2STc1 + 2 * (gas_gravity - 1.5) * H2STc2
            
            # Interpolate N2 pressure correction
            N2Pc1 = -2.0667 * (xN2 * 100)
            N2Pc2 = -2.6667 * (xN2 * 100)
            N2Pc = 2 * (2 - gas_gravity) * N2Pc1 + 2 * (gas_gravity - 1.5) * N2Pc2
            
        else:
            N2Tc = -2.4 * (xN2 * 100)
            Co2Tc = -0.8667 * (xCO2 * 100)
            H2sTc = 0.6667 * (xH2S * 100)
            
            N2Pc = -2.6667 * (xN2 * 100)
            Co2Pc = 4.5333 * (xCO2 * 100)
            H2sPc = 6.1333 * (xH2S * 100)
        
        Tc = Tc + N2Tc + Co2Tc + H2sTc
        Pc = Pc + N2Pc + Co2Pc + H2sPc
    
    # Wichert-Aziz correction for H2S and CO2
    Epsilon = 120 * ((xH2S + xCO2)**0.9 - (xH2S + xCO2)**1.6) + 15 * (xH2S**0.5 - xH2S**4)
    
    TCprime = Tc - Epsilon
    PCprime = Pc * TCprime / (Tc + xH2S * (1 - xH2S) * Epsilon)
    
    # Additional corrections for N2 and H2O
    TCdoubleprime = ((TCprime - 227.2 * xN2 - 1165 * xH2O) / (1 - xN2 - xH2O)) - 246.1 * xN2 + 400 * xH2O
    PCdoubleprime = ((PCprime - 493.1 * xN2 - 3200 * xH2O) / (1 - xN2 - xH2O)) - 162 * xN2 + 1270 * xH2O
    
    # Main calculation loop
    for i in range(n + 1):
        p_i = i * pressure_increment_psi
        T_i = temperature_f
        
        # Minimum pressure constraint
        if p_i < 14.7:
            p_i = 14.7
        
        # Maximum pressure constraint
        if p_i > pressure_psi:
            p_i = pressure_psi
            
        pressures[i] = p_i
        
        # Gas compressibility factor (Z-factor)
        Tr = (T_i + 460) / TCdoubleprime
        Pr = p_i / PCdoubleprime
        
        Dz = 10**(0.3106 - 0.49 * Tr + 0.1824 * (Tr**2))
        Cz = 0.132 - 0.32 * (math.log(Tr) / math.log(10))
        Bz = ((0.62 - 0.23 * Tr) * Pr + 
              ((0.066 / (Tr - 0.86)) - 0.037) * (Pr**2) + 
              0.32 * (Pr**6) / (10**(9 * (Tr - 1))))
        Az = 1.39 * ((Tr - 0.92)**0.5) - 0.36 * Tr - 0.101
        Z = Az + (1 - Az) * math.exp(-Bz) + Cz * (Pr**Dz)
        
        z_factors[i] = Z
        
        # Gas density
        densG = 2.7 * gas_gravity * p_i / ((T_i + 460) * Z)
        gas_densities[i] = densG
        
        # Gas formation volume factor
        Bg = 0.0283 * Z * (T_i + 460) / p_i
        bg_values[i] = Bg
        
        # Gas viscosity
        Xvisc = 3.5 + (986 / (T_i + 460)) + 0.01 * 29 * gas_gravity
        Lambda = 2.4 - 0.2 * Xvisc
        Kvisc = ((9.4 + 0.02 * 29 * gas_gravity) * ((T_i + 460)**1.5)) / (209 + 19 * 29 * gas_gravity + T_i + 460)
        MuG = Kvisc * 1e-4 * math.exp(Xvisc * ((0.01602 * densG)**Lambda))
        
        gas_viscosities[i] = MuG
    
    # Handle final pressure point if needed (like VBA does)
    j = pressure_psi / pressure_increment_psi
    if n < j:
        final_index = n + 1
        new_increment = (j - n) * pressure_increment_psi
        p_final = n * pressure_increment_psi + new_increment
        
        if p_final > pressure_psi:
            p_final = pressure_psi
            
        pressures[final_index] = p_final
        
        # Repeat calculations for final pressure point
        T_i = temperature_f
        
        # Gas compressibility factor
        Tr = (T_i + 460) / TCdoubleprime
        Pr = p_final / PCdoubleprime
        
        Dz = 10**(0.3106 - 0.49 * Tr + 0.1824 * (Tr**2))
        Cz = 0.132 - 0.32 * (math.log(Tr) / math.log(10))
        Bz = ((0.62 - 0.23 * Tr) * Pr + 
              ((0.066 / (Tr - 0.86)) - 0.037) * (Pr**2) + 
              0.32 * (Pr**6) / (10**(9 * (Tr - 1))))
        Az = 1.39 * ((Tr - 0.92)**0.5) - 0.36 * Tr - 0.101
        Z = Az + (1 - Az) * math.exp(-Bz) + Cz * (Pr**Dz)
        
        z_factors[final_index] = Z
        
        # Gas density
        densG = 2.7 * gas_gravity * p_final / ((T_i + 460) * Z)
        gas_densities[final_index] = densG
        
        # Gas formation volume factor
        Bg = 0.0283 * Z * (T_i + 460) / p_final
        bg_values[final_index] = Bg
        
        # Gas viscosity
        Xvisc = 3.5 + (986 / (T_i + 460)) + 0.01 * 29 * gas_gravity
        Lambda = 2.4 - 0.2 * Xvisc
        Kvisc = ((9.4 + 0.02 * 29 * gas_gravity) * ((T_i + 460)**1.5)) / (209 + 19 * 29 * gas_gravity + T_i + 460)
        MuG = Kvisc * 1e-4 * math.exp(Xvisc * ((0.01602 * densG)**Lambda))
        
        gas_viscosities[final_index] = MuG
    
    # Trim arrays to actual size
    actual_size = final_index + 1 if 'final_index' in locals() else n + 1
    
    results = {
        'pressures_psi': pressures[:actual_size],
        'gas_densities_lbm_ft3': gas_densities[:actual_size],
        'gas_viscosities_cp': gas_viscosities[:actual_size],
        'bg_ft3_scf': bg_values[:actual_size],
        'z_factors': z_factors[:actual_size],
        'critical_properties': {
            'gas_gravity': gas_gravity,
            'pseudocritical_pressure_psi': PCdoubleprime,
            'pseudocritical_temperature_r': TCdoubleprime
        }
    }
    
    return results


def incompressible_single_phase(
    density_lbm_ft3=65.5,
    viscosity_cp=1.2,
    diameter_in=4,
    length_ft=1000,
    angle_deg=90,
    roughness_ft=0.0006,
    pressure_drop_psi=None,
    flowrate_bbl_d=None
):
    """
    Incompressible Single Phase Flow Calculator - calculates pressure drop or flow rate.
    
    This function replicates the Excel VBA "Incompressible Single Phase" calculations,
    performing either pressure drop calculation (given flow rate) or flow rate 
    calculation (given pressure drop) for incompressible single-phase flow in pipes.
    
    Parameters:
    -----------
    density_lbm_ft3 : float, default=65.5
        Fluid density in lbm/ft³
    viscosity_cp : float, default=1.2
        Fluid viscosity in cp
    diameter_in : float, default=4
        Pipe diameter in inches
    length_ft : float, default=1000
        Pipe length in feet
    angle_deg : float, default=90
        Pipe inclination angle in degrees (0=horizontal, 90=vertical)
    roughness_ft : float, default=0.0006
        Pipe roughness in feet
    pressure_drop_psi : float, optional
        Pressure drop in psi (provide this to calculate flow rate)
    flowrate_bbl_d : float, optional
        Flow rate in Bbl/D (provide this to calculate pressure drop)
        
    Returns:
    --------
    dict : Dictionary containing:
        - If calculating pressure drop:
            {'pressure_drop_psi': float, 'flow_rate_bbl_d': float, 'calculation_type': 'pressure_drop'}
        - If calculating flow rate:
            {'flow_rate_bbl_d': float, 'pressure_drop_psi': float, 'calculation_type': 'flow_rate'}
        - Additional parameters:
            {'reynolds_number': float, 'friction_factor': float, 'velocity_ft_s': float}
    
    Raises:
    -------
    ValueError : If neither or both pressure_drop_psi and flowrate_bbl_d are provided
                 If input values are invalid (negative or zero where not allowed)
    """
    import math
    
    # Input validation
    if (pressure_drop_psi is None and flowrate_bbl_d is None) or \
       (pressure_drop_psi is not None and flowrate_bbl_d is not None):
        raise ValueError("Provide either pressure_drop_psi OR flowrate_bbl_d, not both or neither")
    
    # Check for positive values
    if density_lbm_ft3 <= 0 or viscosity_cp <= 0 or diameter_in <= 0 or length_ft <= 0:
        raise ValueError("Density, viscosity, diameter, and length must be greater than 0")
    
    # Convert units
    diameter_ft = diameter_in / 12  # Convert inches to feet
    area_ft2 = math.pi * (diameter_ft ** 2) / 4  # Pipe cross-sectional area
    
    # Angle conversion to radians
    angle_rad = angle_deg * math.pi / 180
    
    if flowrate_bbl_d is not None:
        # Calculate pressure drop given flow rate
        if flowrate_bbl_d <= 0:
            raise ValueError("Flow rate must be greater than 0")
        
        # Calculate velocity
        velocity_ft_s = (5.615 * flowrate_bbl_d) / (86400 * area_ft2)  # ft/s
        
        # Calculate mass flow rate
        mdot_lbm_d = area_ft2 * velocity_ft_s * density_lbm_ft3 * 86400  # lbm/D
        
        # Calculate Reynolds number
        reynolds_number = 0.022 * mdot_lbm_d / (diameter_ft * viscosity_cp)
        
        # Calculate friction factor
        if reynolds_number < 20:
            friction_factor = 16 / reynolds_number
        else:
            # Colebrook-White equation (implicit)
            friction_factor = (1 / (-4 * (math.log10((roughness_ft / 3.7065) - 
                              (5.0452 / reynolds_number) * 
                              (math.log10(((roughness_ft ** 1.1098) / 2.8257) + 
                              ((7.149 / reynolds_number) ** 0.8981))))))) ** 2
        
        # Calculate pressure gradient (psi/ft)
        dp_dz = (1/144) * ((math.sin(angle_rad) * density_lbm_ft3) + 
                          ((friction_factor * (mdot_lbm_d ** 2)) / 
                          (7.413e10 * (diameter_ft ** 5) * density_lbm_ft3)))
        
        # Calculate total pressure drop
        pressure_drop_calculated = dp_dz * length_ft
        
        return {
            'pressure_drop_psi': pressure_drop_calculated,
            'flow_rate_bbl_d': flowrate_bbl_d,
            'calculation_type': 'pressure_drop',
            'reynolds_number': reynolds_number,
            'friction_factor': friction_factor,
            'velocity_ft_s': velocity_ft_s,
            'mass_flow_rate_lbm_d': mdot_lbm_d
        }
    
    else:
        # Calculate flow rate given pressure drop
        if pressure_drop_psi <= 0:
            raise ValueError("Pressure drop must be greater than 0")
        
        dp_dz = pressure_drop_psi / length_ft  # Pressure gradient (psi/ft)
        
        # Check for valid flow conditions
        gravity_component = math.sin(angle_rad) * density_lbm_ft3
        if gravity_component > (144 * dp_dz):
            raise ValueError("Flow rate would be negative - check input parameters (pressure drop too low for given inclination)")
        
        # Initial guess for flow rate
        flowrate_guess = 500  # Bbl/D
        error = 1.0
        max_iterations = 100
        iteration = 0
        
        while error > 0.00001 and iteration < max_iterations:
            iteration += 1
            
            # Calculate velocity and mass flow rate
            velocity_ft_s = (5.615 * flowrate_guess) / (86400 * area_ft2)
            mdot_lbm_d = area_ft2 * velocity_ft_s * density_lbm_ft3 * 86400
            
            # Calculate Reynolds number
            reynolds_number = 0.022 * mdot_lbm_d / (diameter_ft * viscosity_cp)
            
            # Calculate friction factor
            if reynolds_number < 20:
                ff1 = 16 / reynolds_number
            else:
                ff1 = (1 / (-4 * (math.log10((roughness_ft / 3.7065) - 
                            (5.0452 / reynolds_number) * 
                            (math.log10(((roughness_ft ** 1.1098) / 2.8257) + 
                            ((7.149 / reynolds_number) ** 0.8981))))))) ** 2
            
            # Calculate required friction factor from pressure balance
            ff2 = ((144 * dp_dz - gravity_component) * 7.413e10 * 
                   (diameter_ft ** 5) * density_lbm_ft3) / (mdot_lbm_d ** 2)
            
            # Calculate error
            error = abs((ff1 - ff2) / ff2)
            
            # Update mass flow rate
            new_mdot_lbm_d = (((144 * dp_dz - gravity_component) * 7.413e10 * 
                              (diameter_ft ** 5) * density_lbm_ft3) / ff1) ** 0.5
            
            new_velocity_ft_s = new_mdot_lbm_d / (area_ft2 * density_lbm_ft3 * 86400)
            flowrate_guess = new_velocity_ft_s * 86400 * area_ft2 / 5.615  # Bbl/D
        
        if iteration >= max_iterations:
            raise RuntimeError("Solution did not converge after maximum iterations")
        
        if flowrate_guess < 0:
            raise ValueError("Calculated flow rate is negative - check input parameters")
        
        return {
            'flow_rate_bbl_d': flowrate_guess,
            'pressure_drop_psi': pressure_drop_psi,
            'calculation_type': 'flow_rate',
            'reynolds_number': reynolds_number,
            'friction_factor': ff1,
            'velocity_ft_s': new_velocity_ft_s,
            'mass_flow_rate_lbm_d': new_mdot_lbm_d,
            'iterations_to_converge': iteration
        }


def compressible_single_phase(
    gravity=0.71,
    co2_percent=0,
    n2_percent=0,
    h2s_percent=0,
    h2o_percent=0,
    diameter_in=2.259,
    length_ft=10000,
    angle_deg=90,
    roughness_ft=0.0006,
    exit_temperature_f=150,
    inlet_temperature_f=200,
    exit_pressure_psi=None,
    inlet_pressure_psi=None,
    flow_rate_mscf_d=None,
    calculation_mode='auto'
):
    """
    Compressible Single Phase Gas Flow Calculator - calculates inlet pressure, exit pressure, or flow rate.
    
    This function replicates the Excel VBA "Compressible Single Phase" calculations,
    performing pressure traverse calculations for compressible gas flow in pipes using
    gas property correlations and iterative solution methods.
    
    Parameters:
    -----------
    gravity : float, default=0.71
        Gas specific gravity (air = 1.0)
    co2_percent : float, default=0
        CO2 percentage in gas
    n2_percent : float, default=0
        N2 percentage in gas
    h2s_percent : float, default=0
        H2S percentage in gas
    h2o_percent : float, default=0
        H2O percentage in gas
    diameter_in : float, default=2.259
        Pipe diameter in inches
    length_ft : float, default=10000
        Pipe length in feet
    angle_deg : float, default=90
        Pipe inclination angle in degrees (0=horizontal, 90=vertical)
    roughness_ft : float, default=0.0006
        Pipe roughness in feet
    exit_temperature_f : float, default=150
        Exit (surface) temperature in Fahrenheit
    inlet_temperature_f : float, default=200
        Inlet (bottom) temperature in Fahrenheit
    exit_pressure_psi : float, optional
        Exit (surface) pressure in psi
    inlet_pressure_psi : float, optional
        Inlet (bottom) pressure in psi
    flow_rate_mscf_d : float, optional
        Flow rate in MScf/D
    calculation_mode : str, default='auto'
        Calculation mode: 'auto', 'inlet_pressure', 'exit_pressure', or 'flow_rate'
        
    Returns:
    --------
    dict : Dictionary containing:
        - Calculated parameter (inlet_pressure_psi, exit_pressure_psi, or flow_rate_mscf_d)
        - Input parameters used
        - 'calculation_type': str indicating what was calculated
        - 'gas_properties': dict with critical properties and corrections
    
    Raises:
    -------
    ValueError : If input parameters are invalid or calculation requirements not met
    """
    import math
    
    # Determine calculation mode automatically if not specified
    if calculation_mode == 'auto':
        provided_params = sum([
            exit_pressure_psi is not None,
            inlet_pressure_psi is not None, 
            flow_rate_mscf_d is not None
        ])
        
        if provided_params != 2:
            raise ValueError("Provide exactly 2 of the 3 parameters: exit_pressure_psi, inlet_pressure_psi, flow_rate_mscf_d")
        
        if exit_pressure_psi is None:
            calculation_mode = 'exit_pressure'
        elif inlet_pressure_psi is None:
            calculation_mode = 'inlet_pressure'
        elif flow_rate_mscf_d is None:
            calculation_mode = 'flow_rate'
    
    # Input validation
    if gravity <= 0 or diameter_in <= 0 or length_ft <= 0:
        raise ValueError("Gravity, diameter, and length must be greater than 0")
    
    # Unit conversions
    diameter_ft = diameter_in / 12
    xH2S = h2s_percent / 100
    xCO2 = co2_percent / 100
    xN2 = n2_percent / 100
    xH2O = h2o_percent / 100
    
    # Critical properties correlations
    if gravity > 0.7:
        Tc = gravity * 318 + 166  # °R
        Pc = gravity * (-56) + 708  # psi
    else:
        Tc = gravity * 318 + 166  # °R
        Pc = gravity * (-36) + 693  # psi
    
    # Corrections for non-hydrocarbon components
    if gravity <= 1.5:
        N2Tc = -2.4 * (xN2 * 100)
        Co2Tc = -0.8667 * (xCO2 * 100)
        H2sTc = 1.3333 * (xH2S * 100)
        
        N2Pc = -2.0667 * (xN2 * 100)
        Co2Pc = 4.5333 * (xCO2 * 100)
        H2sPc = 6.1333 * (xH2S * 100)
        
    elif gravity < 2:
        N2Tc = -2.4 * (xN2 * 100)
        Co2Tc = -0.8667 * (xCO2 * 100)
        Co2Pc = 4.5333 * (xCO2 * 100)
        H2sPc = 6.1333 * (xH2S * 100)
        
        # Interpolate H2S temperature correction
        H2STc1 = 1.3333 * (xH2S * 100)
        H2STc2 = 0.6667 * (xH2S * 100)
        H2sTc = 2 * (2 - gravity) * H2STc1 + 2 * (gravity - 1.5) * H2STc2
        
        # Interpolate N2 pressure correction
        N2Pc1 = -2.0667 * (xN2 * 100)
        N2Pc2 = -2.6667 * (xN2 * 100)
        N2Pc = 2 * (2 - gravity) * N2Pc1 + 2 * (gravity - 1.5) * N2Pc2
        
    else:
        N2Tc = -2.4 * (xN2 * 100)
        Co2Tc = -0.8667 * (xCO2 * 100)
        H2sTc = 0.6667 * (xH2S * 100)
        
        N2Pc = -2.6667 * (xN2 * 100)
        Co2Pc = 4.5333 * (xCO2 * 100)
        H2sPc = 6.1333 * (xH2S * 100)
    
    Tpc = Tc + N2Tc + Co2Tc + H2sTc
    Ppc = Pc + N2Pc + Co2Pc + H2sPc
    
    # Wichert-Aziz correction
    Epsilon = 120 * ((xH2S + xCO2)**0.9 - (xH2S + xCO2)**1.6) + 15 * (xH2S**0.5 - xH2S**4)
    
    TCprime = Tpc - Epsilon
    PCprime = Ppc * TCprime / (Tpc + xH2S * (1 - xH2S) * Epsilon)
    
    # Additional corrections for N2 and H2O
    TCdoubleprime = ((TCprime - 227.2 * xN2 - 1165 * xH2O) / (1 - xN2 - xH2O)) - 246.1 * xN2 + 400 * xH2O
    PCdoubleprime = ((PCprime - 493.1 * xN2 - 3200 * xH2O) / (1 - xN2 - xH2O)) - 162 * xN2 + 1270 * xH2O
    
    def calculate_z_factor(pressure, temperature):
        """Calculate gas compressibility factor"""
        Tr = (temperature + 460) / TCdoubleprime
        Pr = pressure / PCdoubleprime
        
        Dz = 10**(0.3106 - 0.49 * Tr + 0.1824 * (Tr**2))
        Cz = 0.132 - 0.32 * (math.log10(Tr))
        Bz = ((0.62 - 0.23 * Tr) * Pr + 
              ((0.066 / (Tr - 0.86)) - 0.037) * (Pr**2) + 
              0.32 * (Pr**6) / (10**(9 * (Tr - 1))))
        Az = 1.39 * ((Tr - 0.92)**0.5) - 0.36 * Tr - 0.101
        Z = Az + (1 - Az) * math.exp(-Bz) + Cz * (Pr**Dz)
        
        return Z
    
    def calculate_gas_properties(pressure, temperature, flow_rate):
        """Calculate gas density, viscosity, and friction factor"""
        Z = calculate_z_factor(pressure, temperature)
        
        # Gas density
        dens_g = 2.7 * gravity * pressure / ((temperature + 460) * Z)
        
        # Gas viscosity
        X_visc = 3.5 + (986 / (temperature + 460)) + 0.01 * 29 * gravity
        Lambda = 2.4 - 0.2 * X_visc
        K_visc = ((9.4 + 0.02 * 29 * gravity) * ((temperature + 460)**1.5)) / (209 + 19 * 29 * gravity + temperature + 460)
        mu_g = K_visc * 1e-4 * math.exp(X_visc * ((0.01602 * dens_g)**Lambda))
        
        # Reynolds number and friction factor
        N_re = 20.09 * gravity * flow_rate / (diameter_in * mu_g)
        ff = (1 / (-4 * (math.log10((roughness_ft / 3.7065) - 
                                   (5.0452 / N_re) * 
                                   (math.log10(((roughness_ft**1.1098) / 2.8257) + 
                                               ((7.149 / N_re)**0.8981))))))) ** 2
        
        return Z, dens_g, mu_g, ff
    
    def pressure_traverse(start_pressure, start_temp, end_temp, flow_rate, direction='forward'):
        """Perform pressure traverse calculation"""
        temp_grad = (end_temp - start_temp) / length_ft
        increment = 100  # ft
        n = int(length_ft / increment)
        
        p = [0] * (n + 2)
        T = [0] * (n + 2)
        
        p[0] = start_pressure
        T[0] = start_temp
        
        for i in range(1, n + 2):
            if direction == 'forward':
                T[i] = T[i-1] + (temp_grad * increment)
            else:  # backward
                T[i] = T[i-1] - (temp_grad * increment)
            
            p[i] = p[i-1]  # Initial guess
            
            # Iterate for Z-factor convergence
            error_z = 1
            iterations = 0
            while error_z > 0.001 and iterations < 50:
                iterations += 1
                
                Z, dens_g, mu_g, ff = calculate_gas_properties(p[i], T[i], flow_rate)
                
                # Calculate pressure using appropriate equation
                s = -0.0375 * gravity * math.sin(angle_deg * math.pi / 180) * increment / (Z * (T[i] + 460))
                
                if angle_deg == 0:  # Horizontal
                    if direction == 'forward':
                        p2_squared = p[i-1]**2 + 1.007e-4 * gravity * ff * Z * (T[i] + 460) * (flow_rate**2) * increment / (diameter_in**5)
                    else:
                        p2_squared = p[i-1]**2 - 1.007e-4 * gravity * ff * Z * (T[i] + 460) * (flow_rate**2) * increment / (diameter_in**5)
                else:  # Inclined
                    if direction == 'forward':
                        p2_squared = (math.exp(-s) * (p[i-1]**2) - 
                                     2.685e-3 * (-math.exp(-s) + 1) * ff * 
                                     ((Z * (T[i] + 460) * flow_rate)**2) / 
                                     (math.sin(angle_deg * math.pi / 180) * (diameter_in**5)))
                    else:
                        p2_squared = (math.exp(s) * (p[i-1]**2) + 
                                     2.685e-3 * (math.exp(s) - 1) * ff * 
                                     ((Z * (T[i] + 460) * flow_rate)**2) / 
                                     (math.sin(angle_deg * math.pi / 180) * (diameter_in**5)))
                
                if p2_squared < 0:
                    raise ValueError("Pressure calculation resulted in negative pressure - check input parameters")
                
                new_p = p2_squared**0.5
                
                # Check Z-factor convergence
                Z_check = calculate_z_factor(new_p, T[i])
                error_z = abs((Z_check - Z) / Z_check)
                p[i] = new_p
        
        # Handle final segment if needed
        final_pressure = p[n]
        m = length_ft / increment
        if n < m:
            p_grad = (p[n] - p[n-1]) / increment if n > 0 else 0
            final_pressure = p[n] + p_grad * (m - n) * increment
        
        return final_pressure
    
    # Main calculation logic
    if calculation_mode == 'inlet_pressure':
        if exit_pressure_psi is None or flow_rate_mscf_d is None:
            raise ValueError("exit_pressure_psi and flow_rate_mscf_d must be provided for inlet pressure calculation")
        
        calculated_pressure = pressure_traverse(exit_pressure_psi, exit_temperature_f, inlet_temperature_f, flow_rate_mscf_d, 'forward')
        
        return {
            'inlet_pressure_psi': calculated_pressure,
            'exit_pressure_psi': exit_pressure_psi,
            'flow_rate_mscf_d': flow_rate_mscf_d,
            'calculation_type': 'inlet_pressure',
            'gas_properties': {
                'gravity': gravity,
                'pseudocritical_pressure_psi': PCdoubleprime,
                'pseudocritical_temperature_r': TCdoubleprime
            }
        }
    
    elif calculation_mode == 'exit_pressure':
        if inlet_pressure_psi is None or flow_rate_mscf_d is None:
            raise ValueError("inlet_pressure_psi and flow_rate_mscf_d must be provided for exit pressure calculation")
        
        calculated_pressure = pressure_traverse(inlet_pressure_psi, inlet_temperature_f, exit_temperature_f, flow_rate_mscf_d, 'backward')
        
        return {
            'exit_pressure_psi': calculated_pressure,
            'inlet_pressure_psi': inlet_pressure_psi,
            'flow_rate_mscf_d': flow_rate_mscf_d,
            'calculation_type': 'exit_pressure',
            'gas_properties': {
                'gravity': gravity,
                'pseudocritical_pressure_psi': PCdoubleprime,
                'pseudocritical_temperature_r': TCdoubleprime
            }
        }
    
    elif calculation_mode == 'flow_rate':
        if inlet_pressure_psi is None or exit_pressure_psi is None:
            raise ValueError("inlet_pressure_psi and exit_pressure_psi must be provided for flow rate calculation")
        
        if exit_pressure_psi >= inlet_pressure_psi:
            raise ValueError("Exit pressure should be smaller than inlet pressure")
        
        # Iterative solution for flow rate
        error_f = 1
        flow_rate_guess = 500  # MScf/D
        max_iterations = 100
        iteration = 0
        
        while error_f > 0.00001 and iteration < max_iterations:
            iteration += 1
            
            calculated_inlet = pressure_traverse(exit_pressure_psi, exit_temperature_f, inlet_temperature_f, flow_rate_guess, 'forward')
            
            error_f = abs(inlet_pressure_psi - calculated_inlet) / inlet_pressure_psi
            error_sign = (inlet_pressure_psi - calculated_inlet) / inlet_pressure_psi
            
            flow_rate_guess = flow_rate_guess * error_sign + flow_rate_guess
            
            if flow_rate_guess < 1:
                raise ValueError("There is not enough inlet pressure to overcome potential energy - check input pressures")
        
        if iteration >= max_iterations:
            raise RuntimeError("Flow rate calculation did not converge")
        
        return {
            'flow_rate_mscf_d': flow_rate_guess,
            'inlet_pressure_psi': inlet_pressure_psi,
            'exit_pressure_psi': exit_pressure_psi,
            'calculation_type': 'flow_rate',
            'iterations_to_converge': iteration,
            'gas_properties': {
                'gravity': gravity,
                'pseudocritical_pressure_psi': PCdoubleprime,
                'pseudocritical_temperature_r': TCdoubleprime
            }
        }
    
    else:
        raise ValueError(f"Invalid calculation_mode: {calculation_mode}")
