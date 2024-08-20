# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:27:56 2024

"""
# Ryan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from sklearn.linear_model import LinearRegression

# Files can be found in google drive folder
PROD_FILE = "renewable_energy_production.csv"
CONS_FILE = "renewable_energy_usage.csv"
GDP_FILE = "state_gdp.csv"

def col_to_list(col):
   '''
    Converts a column to a list and converts the items in the list into float,
    removing commas.

    '''
   # Convert column to list
   column_list = col.tolist()
   
   # Remove commas and convert to float
   float_list = [float(item.replace(',', '')) for item in column_list]
   
   
   return float_list

def read_file(filename):
    '''
    Reads the file and returns a cleaned dataframe with the US row removed.

    '''
    # read file into dataframe using pandas
    df = pd.read_csv(filename, skiprows = 2)
    df.dropna()
    # drop row for US
    df = df[df['State'] != 'US']

    clean_df = df.dropna(axis = 1, how='all')
    #print(clean_df.columns)
     
    return df

def clean_value(value):
    '''
    Removes whitespace and commas from a value and converts it to float
    '''
    # Remove whitespace and commas
    cleaned_value = value.replace(" ", '').replace(",", "")
    
    # Convert to float
    return float(cleaned_value)


def get_top_5_states(df, gdp_df, year):
    '''
    Given a dataframe and a year, this function calculates the top five states
    in terms of renewable energy production or consumption in the given year.
    '''
    
    year = str(year)

    # Convert the year column in both dataframes to numeric, handling commas
    #df[year] = pd.to_numeric(df[year].str.replace(',', ''))
    #gdp_df[year] = pd.to_numeric(gdp_df[year].str.replace(',', ''))

    # Create separate dictionaries for GDP/consumption and GDP/production 
    # ratios
    gdp_dict = dict(zip(gdp_df["State"], gdp_df[year]))
    energy_dict = dict(zip(df["State"], df[year]))
    
    # Clean values
    for key in gdp_dict:
        gdp_dict[key] = clean_value(gdp_dict[key])
    for key in energy_dict:
        energy_dict[key] = clean_value(energy_dict[key])
    
    # Calculate GDP/energy ratio
    ratio_dict = {}
    for state in energy_dict:
        if state in gdp_dict:
            ratio_dict[state] = gdp_dict[state] / energy_dict[state]
            
    
    # Sort the dictionary by the ratio values in descending order and get 
    # the top 5 states
    top_5 = sorted(ratio_dict.items(), key=lambda item: item[1],
                   reverse=True)[:5]
    
    return top_5



def calculate_change(df, present_year, past_year):
    """
    Given a dataframe, the most recent year and a past year, this function
    calculates the total change in energy production (or consumption)
    for each state during the given time period.
    """
    # Convert year to string
    present_year = str(present_year)
    past_year = str(past_year)
    
    # Remove commas and convert the columns to numeric
    df[present_year] = pd.to_numeric(df[present_year].str.replace(",", ""))
    df[past_year] = pd.to_numeric(df[past_year].str.replace(",", ""))
    
    # Calculate the change
    df["Change"] = df[present_year] - df[past_year]
    
    # Return the relevant columns
    return df[["State", "Change"]]



def correlation(df, gdp_df, year):
    '''
    Calculates the correlation between production or consumption
    and GDP in a given year.
    '''
    year = str(year)


    x = pd.to_numeric(df[year].str.replace(",",""))
    y = col_to_list(gdp_df[year])
    
    corr = statistics.correlation(x, y)
    
    return corr
    

def consumption_vs_production(cons_df, prod_df, year):
    '''
    Plots consumption vs production for all 50 states in a given year.
    '''
    year = str(year)
    
    prod_df[year] = pd.to_numeric(prod_df[year].str.replace(",", ""))
    cons_df[year] = pd.to_numeric(cons_df[year].str.replace(",", ""))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x=cons_df[year], y=prod_df[year])
    plt.title(f"Renewable Energy Production vs Consumption for States in \
{year}")
    plt.xlabel("Consumption in BTU (billions)")
    plt.ylabel("Production in BTU (billions)")
    plt.show()

    
    
def gdp_scatterplot(filename, GDP_FILE, year):
    '''
    Plots consumption or consumption vs GDP for all states in the given year
    '''
    year = str(year)
    
    df = read_file(filename)
    gdp_df = read_file(GDP_FILE)
    
    
    if filename == CONS_FILE:
        label = "Consumption"
    elif filename == PROD_FILE:
        label = "Production"
        
    gdp_df[year] = pd.to_numeric(gdp_df[year].str.replace(',', ''))
    df[year] = pd.to_numeric(df[year].str.replace(",", ""))
    
    #print(gdp_df[year])

    # Set up plot        
    plt.figure(figsize=(12, 8))
    sns.regplot(x = gdp_df[year], y = df[year])
    plt.title(f"Renewable Energy {label} vs. Real GDP for States \
in {year}", fontsize = 16)
    plt.xlabel("GDP in millions")
    plt.ylabel(f"{label} in btu (billions)")
    
    # Show plot
    plt.show()
    

    

def main():
    # Create dataframes
    cons_df = read_file(CONS_FILE)
    prod_df = read_file(PROD_FILE)
    gdp_df = read_file(GDP_FILE)
    
    # Calculate the change in energy production from 1970 to 2022
    prod_change_df = calculate_change(prod_df, 2022, 1970)
    top_5_states_prod = prod_change_df.nlargest(5, 'Change')
    print("Top 5 states with greatest changes in production:")
    print(top_5_states_prod)
    


    # Calculate the change in energy consumption from 1970 to 2022
    cons_change_df = calculate_change(cons_df, 2022, 1970)
    top_5_states_cons = cons_change_df.nlargest(5, 'Change')
    print("Top 5 states with greatest changes in consumption:")
    print(top_5_states_cons)
    
       
    
    cons_df = read_file(CONS_FILE)
    prod_df = read_file(PROD_FILE)
    
    # Get the top 5 states by energy consumption in 2022
    top_5_consumption = get_top_5_states(cons_df, gdp_df, 2022)
    print("Top 5 states by renewable energy consumption \
(divided by GDP) in 2022:")
    print(top_5_consumption)
    
    # Get the top 5 states by energy production in 2022
    top_5_production = get_top_5_states(prod_df, gdp_df, 2022)
    print("Top 5 states by renewable energy production (divided by GDP) \
in 2022:")
    print(top_5_production)
    
    

    cons_df = read_file(CONS_FILE)
    prod_df = read_file(PROD_FILE)
    gdp_df = read_file(GDP_FILE)
    
    # Calculate correlations
    corr1 = correlation(cons_df, gdp_df, "2022")
    print(f"The correlation between renewable energy consumption and \
real GDP is {corr1}.")
    
    corr2 = correlation(prod_df, gdp_df, "2022")
    print((f"The correlation between renewable energy production and \
real GDP is {corr2}."))
    
    cons_df = read_file(CONS_FILE)
    prod_df = read_file(PROD_FILE)
    gdp_df = read_file(GDP_FILE)

    # Create consumption vs production plot
    consumption_vs_production(cons_df, prod_df, 2022)
    
    # Create production/consumption vs GDP scatterplots
    gdp_scatterplot(PROD_FILE, GDP_FILE, 2022)
    
    gdp_scatterplot(CONS_FILE, GDP_FILE, 2022)
    
    


if __name__ == "__main__":
    main()

