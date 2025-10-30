# Color Formatting Update Summary

## Overview
Updated all functions in `artificial_lift.py` starting from `echometer_fl_bhp` and after to use the Rich package for colored console output.

## Changes Made

### 1. Added Rich Package Import
- Added `from rich.console import Console` and `from rich.text import Text` at the top
- Created a global `console = Console()` instance

### 2. Updated Functions with Color Formatting

#### IPR Function (existing print statements)
- `qb` and `qmax` variable names are now colored **blue**
- Values are now colored **red**

#### echometer_fl_bhp Function
All print statements updated to use rich formatting:
- Variable names (`z`, `vg`, `L`, `q/A`, `fl`, `fo`, `fw`, `fg`, `rhof`, `BHP`) are colored **blue**
- Calculated values are colored **red**

#### plunger_rate_calculation Function
All print statements updated to use rich formatting:
- Variable names (`t_rise`, `t_fall`, `gamma_l`, `pg`, etc.) are colored **blue**
- Calculated values are colored **red**

#### pcp_design Function
All print statements updated to use rich formatting:
- Variable names (`Rs`, `z`, `GLR`, `pump_speed`, `T`, `BHPin`, `HHPout`, `gamma_l`, `Fr`, `Fb`, `bearing_life`, etc.) are colored **blue**
- Calculated values are colored **red**

### 3. Formatting Pattern
- Variable names: `[blue]variable_name[/blue]`
- Values: `[red]{value:.2f}[/red]`
- Equations remain uncolored for clarity

## Dependencies
- Requires `rich` package: `pip install rich`
- Functions will still work without rich, but without colors

## Example Output
Instead of:
```
Sonic velocity in gas vg = 1234.56 ft/s
```

Now displays:
```
Sonic velocity in gas vg = 1234.56 ft/s
```
Where "vg" appears in blue and "1234.56" appears in red in the terminal.

## Test File
Created `test_color_formatting.py` to demonstrate the color formatting functionality.