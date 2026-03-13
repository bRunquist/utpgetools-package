"""
utpgetools: A collection of tools for UT PGE python projects.

This package provides a comprehensive suite of petroleum engineering tools and
utilities designed for academic and professional use in reservoir engineering,
production engineering, drilling engineering, and facilities design.

Modules:
    artificial_lift: Artificial lift systems analysis (VLP, IPR, gas lift, plunger lift, PCP)
    facilities: Oil and gas processing facilities design and analysis
    forecasting: DCA-based production forecasting with ML prediction for new wells
    formation_evaluation: Formation evaluation and petrophysical analysis (under development)
    general: General-purpose utility functions for mathematical operations
    geomechanics: Geomechanical analysis including stress visualization and fault analysis
    geostats: Geostatistical analysis and spatial modeling (under development)
    numerical_methods: Numerical methods and computational tools (under development)
    production: Production engineering analysis and optimization (under development)
    res_2: Advanced reservoir engineering methods (under development)
    utilities_package: Core utilities for fluid properties and multiphase flow calculations
    well_testing: Comprehensive well testing analysis and pressure transient interpretation

Key Features:
    - Comprehensive artificial lift system analysis and design
    - Multiphase flow calculations using industry-standard correlations
    - PVT property calculations for oil and gas systems
    - Geomechanical analysis with 3D Mohr's circle visualization
    - Facilities design tools for separators and process equipment
    - Well testing and pressure transient analysis capabilities
    - Well-documented functions with extensive examples and theory
    - Consistent unit systems and comprehensive error checking

Applications:
    - Reservoir characterization and modeling
    - Well performance analysis and optimization
    - Artificial lift system design and troubleshooting
    - Facilities engineering and process design
    - Geomechanical analysis for wellbore stability
    - Production optimization and decline analysis
    - Pressure transient analysis and well test interpretation
    - ML-driven production forecasting for new wells
    - Academic research and teaching

Dependencies:
    - numpy: Numerical calculations and array operations
    - scipy: Scientific computing including special functions (for well_testing module)
    - matplotlib: Plotting and visualization (for geomechanics module)
    - rich: Enhanced console output formatting (for artificial_lift module)
    - csv: Data file processing (for geomechanics module)
    - typing: Type hints and annotations
    - pandas: Data manipulation (for forecasting module)
    - scipy: Curve fitting (for forecasting module)
    - scikit-learn: Machine learning models (for forecasting module)
    - seaborn: Statistical visualization (for forecasting module)

Notes:
    This package is developed by and for the University of Texas Petroleum
    and Geosystems Engineering program. All functions include comprehensive
    documentation, examples, and theoretical background to support both
    learning and professional application.

Version: Current development version
Author: UT PGE Program
License: See LICENSE.txt for details
"""

