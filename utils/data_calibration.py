import os
import pandas as pd
import numpy as np

try:
    from fredapi import Fred
    HAS_FRED = True
except ImportError:
    HAS_FRED = False

def calibrate_parameters(fred_api_key=None):
    """
    Calibrates macroeconomic parameters using the FRED API if available.
    Returns a dictionary of calibrated parameters.
    """
    calibrated_params = {}
    
    if fred_api_key is None:
        fred_api_key = os.environ.get('FRED_API_KEY')
        
    if HAS_FRED and fred_api_key:
        print("Using FRED Data for Calibration...")
        try:
            fred = Fred(api_key=fred_api_key)
            
            # Fetch CPI (CPIAUCSL) - compute quarterly inflation
            # FRED CPI is monthly, we can resample to quarterly and compute pct change
            cpi = fred.get_series('CPIAUCSL')
            cpi_q = cpi.resample('Q').mean()
            inflation_q = cpi_q.pct_change() * 4 # Annualized quarterly rate
            
            inflation_mean = inflation_q.mean()
            inflation_std = inflation_q.std()
            
            # Fetch Output Gap (GDPGAP or calculate from GDP and POT)
            # Since GDPGAP series may be discontinued/limited, let's try reading it
            try:
                gdp_gap = fred.get_series('GDPGAP')
                # Usually given as percent, so convert to decimal
                gdp_gap = gdp_gap / 100.0
                output_gap_std = gdp_gap.std()
            except Exception:
                output_gap_std = 0.03 # fallback if series fails
                
            # Fetch Fed Funds Rate
            fedfunds = fred.get_series('FEDFUNDS')
            # Convert percentage to decimal
            fedfunds = fedfunds / 100.0
            
            rate_mean = fedfunds.mean()
            rate_min = max(0.0, float(fedfunds.min())) # don't go below zero
            rate_max = min(0.20, float(fedfunds.max())) # cap at 20%
            rate_range = [rate_min, rate_max]
            
            calibrated_params = {
                'inflation_mean': inflation_mean,
                'inflation_std': inflation_std,
                'output_gap_std': output_gap_std,
                'rate_mean': rate_mean,
                'rate_range': rate_range
            }
        except Exception as e:
            print(f"Failed to fetch from FRED API: {e}. Using fallback values.")
            HAS_FRED = False # trigger fallback
            
    if not HAS_FRED or not fred_api_key:
        print("FRED API key not found or failed. Using fallback calibrated values.")
        calibrated_params = {
            'inflation_mean': 0.035,
            'inflation_std': 0.025,
            'output_gap_std': 0.03,
            'rate_mean': 0.05,
            'rate_range': [0.0, 0.20]
        }
        
    return calibrated_params

def print_calibration_summary(params):
    """
    Format and print a calibration summary table.
    """
    print("+---------------------------------------------------+")
    print("| Parameter Calibration Summary                     |")
    print("+---------------------------------------------------+")
    print(f"| Target Inflation Mean      : {params['inflation_mean']:6.3f}            |")
    print(f"| Inflation Volatility (Std) : {params['inflation_std']:6.3f}            |")
    print(f"| Output Gap Volatility (Std): {params['output_gap_std']:6.3f}            |")
    print(f"| Average Policy Rate        : {params['rate_mean']:6.3f}            |")
    print(f"| Policy Rate Range          : [{params['rate_range'][0]:.2f}, {params['rate_range'][1]:.2f}]       |")
    print("+---------------------------------------------------+")

if __name__ == "__main__":
    params = calibrate_parameters()
    print_calibration_summary(params)
