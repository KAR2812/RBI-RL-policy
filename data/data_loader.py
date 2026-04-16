"""
Module to load and clean historical RBI datasets from DBIE Excel sheets.
Specifically designed to handle "RBIB Table No. 19 _ Consumer Price Index (Base 2010=100).xlsx".
"""
import pandas as pd
import numpy as np

def load_rbi_historical_data(filepath: str) -> pd.DataFrame:
    """
    Loads empirical CPI data and synthesizes mock output gap/repo rate sequences
    for the Historical Gym environment if proxies don't exist yet in the DB.
    
    Args:
        filepath (str): Path to the DBIE Excel spreadsheet.
        
    Returns:
        pd.DataFrame: Cleaned dataframe aligned row-by-row containing:
            ['date', 'inflation', 'output_gap', 'interest_rate']
            Note: output_gap and historical interest rates may be synthetic 
            for validation purposes if absent from the specific Excel snippet.
    """
    # Load sheet, skip initial metadata rows
    df_raw = pd.read_excel(filepath, header=None)
    
    # Extract only the "A) General Index" rows.
    # Column 1 has Date/Month string, Column 2 has "A) General Index"
    df_general = df_raw[df_raw[2] == "A) General Index"].copy()
    
    # Based on DBIE's format:
    # Column 1 = Month (e.g., 'DEC-2025')
    # Column 9 = Combined Inflation (%)
    
    dates = df_general[1].values
    combined_inflation = pd.to_numeric(df_general[9], errors='coerce').values
    
    # Store in a cleaner format
    df_clean = pd.DataFrame({
        "date": dates,
        "inflation_percent": combined_inflation,
    })
    
    # Convert 'DEC-2025' to datetime and sort chronologically (oldest to newest)
    df_clean['date'] = pd.to_datetime(df_clean['date'], format="%b-%Y")
    df_clean = df_clean.sort_values(by="date").dropna().reset_index(drop=True)
    
    # In RL agent, inflation is used as a decimal (e.g., 2% = 0.02)
    # The RBI data ranges from 2% to 18% in anomalous times. 
    # To keep the agent functioning smoothly inside its training bounds,
    # we convert to absolute decimal.
    df_clean['inflation'] = df_clean['inflation_percent'] / 100.0
    
    # IMPORTANT: Real-world RL needs Output Gap & Historical Policy Rates. 
    # Since the single Excel file provided is strictly CPI, we will synthesize 
    # correlated proxies to complete the MDP state space for the historical validator.
    # In a real environment, you'd join IIP and Repo Rate CSVs here.
    
    np.random.seed(42)
    n_steps = len(df_clean)
    
    # Synthesize Output Gap (Random Walk bounded around 0 with some correlation to inflation shocks)
    # Synthesize Output Gap (Random Walk bounded around 0 with some correlation to inflation shocks)
    output_gaps = np.zeros(n_steps)
    for t in range(1, n_steps):
        inf_change = df_clean['inflation'].iloc[t] - df_clean['inflation'].iloc[t-1]
        output_gaps[t] = 0.8 * output_gaps[t-1] + np.random.normal(0, 0.01) + 2.0 * inf_change
    df_clean['output_gap'] = np.clip(output_gaps, -0.1, 0.1)
    
    # Synthesize actual Historical Interest Rate (A rough Taylor rule approximation for base comp)
    df_clean['interest_rate'] = 0.02 + df_clean['inflation'] + 0.5 * (df_clean['inflation'] - 0.02) + 0.5 * df_clean['output_gap']
    df_clean['interest_rate'] = np.clip(df_clean['interest_rate'], 0.0, 0.10)
    
    return df_clean[['date', 'inflation', 'output_gap', 'interest_rate']]
