"""
Sunrise and Sunset Visualization Script

This script creates a figure with two subplots showing:
- Sunrise times
- Sunset times  
- Hours of daylight (on secondary y-axis)

One subplot for Anchorage, AK and one for Houston, TX
Data is calculated for an entire year.
"""

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from astral import LocationInfo
from astral.sun import sun
import matplotlib.dates as mdates
import pytz


def calculate_sun_data(location_info, year=2025):
    """
    Calculate sunrise, sunset, and daylight hours for a given location and year.
    
    Parameters:
    location_info: astral LocationInfo object
    year: int, year to calculate data for
    
    Returns:
    dates: list of datetime objects
    sunrise_times: list of datetime objects
    sunset_times: list of datetime objects
    daylight_hours: list of float (hours)
    """
    dates = []
    sunrise_times = []
    sunset_times = []
    daylight_hours = []
    
    # Generate dates for entire year
    start_date = dt.date(year, 1, 1)
    end_date = dt.date(year, 12, 31)
    current_date = start_date
    
    while current_date <= end_date:
        try:
            # Calculate sun times with UTC, then convert to local
            s = sun(location_info.observer, date=current_date)
            sunrise_utc = s['sunrise']
            sunset_utc = s['sunset']
            
            # Convert to local time properly
            sunrise_local = sunrise_utc.astimezone(pytz.timezone(location_info.timezone))
            sunset_local = sunset_utc.astimezone(pytz.timezone(location_info.timezone))
            
            # Calculate daylight hours
            daylight_duration = (sunset_utc - sunrise_utc).total_seconds() / 3600
            
            dates.append(current_date)
            sunrise_times.append(sunrise_local.time())
            sunset_times.append(sunset_local.time())
            daylight_hours.append(daylight_duration)
            
        except Exception as e:
            # For extreme polar conditions, use more sophisticated calculation
            print(f"Special handling for {current_date}: {str(e)[:50]}...")
            
            # Day of year for seasonal calculation
            day_of_year = current_date.timetuple().tm_yday
            
            # Solar declination angle (simplified)
            declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
            
            # Hour angle for sunrise/sunset
            lat_rad = np.radians(location_info.latitude)
            decl_rad = np.radians(declination)
            
            # Calculate hour angle
            try:
                cos_hour_angle = -np.tan(lat_rad) * np.tan(decl_rad)
                
                if cos_hour_angle < -1:
                    # Polar day (midnight sun) - sun never sets
                    hour_angle = np.pi  # 24 hours of daylight
                    daylight_hrs = 24.0
                    # Set sunrise to midnight (0:00) and sunset to end of day (24:00 = 0:00 next day)
                    sunrise_decimal = 0.0
                    sunset_decimal = 24.0
                elif cos_hour_angle > 1:
                    # Polar night - sun never rises
                    hour_angle = 0  # 0 hours of daylight
                    daylight_hrs = 0.0
                    # Set both to noon as arbitrary times
                    sunrise_decimal = 12.0
                    sunset_decimal = 12.0
                else:
                    hour_angle = np.arccos(cos_hour_angle)
                    daylight_hrs = 2 * hour_angle * 12 / np.pi
                    
                    # Calculate sunrise and sunset times
                    solar_noon = 12.0  # Approximate solar noon
                    sunrise_decimal = solar_noon - daylight_hrs / 2
                    sunset_decimal = solar_noon + daylight_hrs / 2
                
                # Ensure times are within 24-hour range and handle midnight sun properly
                if sunrise_decimal < 0:
                    sunrise_decimal = 0
                if sunset_decimal > 24:
                    sunset_decimal = 24
                    
                # Convert to time objects, handling 24:00 as 23:59
                sunrise_hr = int(sunrise_decimal)
                sunrise_min = int((sunrise_decimal - sunrise_hr) * 60)
                sunset_hr = int(min(sunset_decimal, 23.99))  # Cap at 23:59 for time object
                sunset_min = int((min(sunset_decimal, 23.99) - sunset_hr) * 60)
                
                dates.append(current_date)
                sunrise_times.append(dt.time(sunrise_hr, sunrise_min))
                sunset_times.append(dt.time(sunset_hr, sunset_min))
                daylight_hours.append(daylight_hrs)
                
            except Exception as e2:
                print(f"Skipping {current_date}: {e2}")
                continue
                
        current_date += dt.timedelta(days=1)
    
    return dates, sunrise_times, sunset_times, daylight_hours


def time_to_decimal_hours(time_obj):
    """Convert time object to decimal hours for plotting."""
    return time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600


def main():
    # Define locations with proper timezone strings
    anchorage = LocationInfo("Anchorage", "Alaska", "America/Anchorage", 61.2181, -149.9003)
    anchorage.timezone = "America/Anchorage"
    
    houston = LocationInfo("Houston", "Texas", "America/Chicago", 29.7604, -95.3698)
    houston.timezone = "America/Chicago"
    
    # Calculate sun data for both cities
    print("Calculating data for Anchorage...")
    anc_dates, anc_sunrise, anc_sunset, anc_daylight = calculate_sun_data(anchorage)
    
    print("Calculating data for Houston...")
    hou_dates, hou_sunrise, hou_sunset, hou_daylight = calculate_sun_data(houston)
    
    print(f"Anchorage data points: {len(anc_dates)}")
    print(f"Houston data points: {len(hou_dates)}")
    
    # Convert times to decimal hours for plotting
    anc_sunrise_decimal = [time_to_decimal_hours(t) for t in anc_sunrise]
    anc_sunset_decimal = [time_to_decimal_hours(t) for t in anc_sunset]
    hou_sunrise_decimal = [time_to_decimal_hours(t) for t in hou_sunrise]
    hou_sunset_decimal = [time_to_decimal_hours(t) for t in hou_sunset]
    
    # Calculate daylight hours as simple difference between sunset and sunrise
    # Handle special case where sunset is 23:59 (representing 24:00 for midnight sun)
    anc_daylight_calculated = []
    for sunrise, sunset in zip(anc_sunrise_decimal, anc_sunset_decimal):
        if sunset >= 23.99 and sunrise <= 0.01:  # Midnight sun condition
            daylight_hrs = 24.0
        elif sunset > sunrise:
            daylight_hrs = sunset - sunrise
        else:  # Sunset is next day
            daylight_hrs = (24 - sunrise + sunset)
        anc_daylight_calculated.append(daylight_hrs)
    
    hou_daylight_calculated = []
    for sunrise, sunset in zip(hou_sunrise_decimal, hou_sunset_decimal):
        if sunset >= 23.99 and sunrise <= 0.01:  # Midnight sun condition (unlikely for Houston)
            daylight_hrs = 24.0
        elif sunset > sunrise:
            daylight_hrs = sunset - sunrise
        else:  # Sunset is next day
            daylight_hrs = (24 - sunrise + sunset)
        hou_daylight_calculated.append(daylight_hrs)
    
    # Create the figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Sunrise, Sunset, and Daylight Hours Throughout the Year', fontsize=16, fontweight='bold')
    
    # Plot Anchorage data
    ax1.plot(anc_dates, anc_sunrise_decimal, 'orange', linewidth=2, label='Sunrise')
    ax1.plot(anc_dates, anc_sunset_decimal, 'red', linewidth=2, label='Sunset')
    
    # Create secondary y-axis for daylight hours
    ax1_twin = ax1.twinx()
    ax1_twin.plot(anc_dates, anc_daylight_calculated, 'blue', linewidth=2, label='Daylight Hours')
    
    # Format Anchorage subplot
    ax1.set_title('Anchorage, Alaska', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time of Day (24-hour)', color='black')
    ax1_twin.set_ylabel('Daylight Hours', color='blue')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 24)
    ax1_twin.set_ylim(0, 24)
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot Houston data
    ax2.plot(hou_dates, hou_sunrise_decimal, 'orange', linewidth=2, label='Sunrise')
    ax2.plot(hou_dates, hou_sunset_decimal, 'red', linewidth=2, label='Sunset')
    
    # Create secondary y-axis for daylight hours
    ax2_twin = ax2.twinx()
    ax2_twin.plot(hou_dates, hou_daylight_calculated, 'blue', linewidth=2, label='Daylight Hours')
    
    # Format Houston subplot
    ax2.set_title('Houston, Texas', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time of Day (24-hour)', color='black')
    ax2_twin.set_ylabel('Daylight Hours', color='blue')
    ax2.set_xlabel('Month')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 24)
    ax2_twin.set_ylim(0, 24)
    
    # Format x-axis
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # Add legends
    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper right')
    
    # Adjust layout and show
    plt.tight_layout()
    plt.show()
    
    # Optional: Save the figure
    # plt.savefig('sunrise_sunset_comparison.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()