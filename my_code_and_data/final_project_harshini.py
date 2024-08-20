# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:08:05 2024

@author: Harshini Dinesh,
@teammates: Hang Hang, Ryan Jiang, Hersh Joshi
Course: DS 2500 Int. Programming with Data
Assignment: Final Project
Date: August 5, 2024
Name of file: final_project_harshini.py
"""

from utils import get_file_path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics 

ELEC_FILE = get_file_path("electric_power.csv")
IND_FILE  = get_file_path("industrial.csv")
RESCOMM_FILE = get_file_path("resid_and_commercial.csv")
AFS_FILE = get_file_path("alt_fuel_stations.csv")

ELEC_ENERGY = "Total Renewable Energy Consumed by the Electric Power Sector"
IND_ENERGY = "Total Renewable Energy Consumed by the Industrial Sector"
RES_ENERGY = "Total Renewable Energy Consumed by the Residential Sector"
COMM_ENERGY = "Total Renewable Energy Consumed by the Commercial Sector"


def process_data(filename, energy_column, sector_name):
    '''
    Reads and cleans data. Then finds total renewable energy consumed 
    by each sector from 1949 to 2023
    '''
    # total renewable energy consumed by each sector
    # from 1949 to 2023
    df = pd.read_csv(filename, skiprows = 10)
    df = df[9:626]
    
    # prevents years from being converted from int to float
    if "Annual Total" in df.columns:
        df["Annual Total"] = df["Annual Total"].astype(
            int, errors= "ignore")
        
    # converts numbers from string to float for sum calculation
    energy_column = pd.to_numeric(df[energy_column], errors = "coerce")
    total_energy = energy_column.sum()   
    print(f"Total renewable energy consumed by the {sector_name} sector:"
          f" {round(total_energy, 2)} Trillion btu")
    return df, total_energy


def analyze_sector_energy(df, columns_to_sum, sector_name):
    '''
    Further cleans data columns replacing missing/unavailable data with NaN,
    and then getting rid of all NaN values. Then finds the most consumed energy
    type by each sector. 
    '''
    # replaces unavailable data with NaN
    df.replace(["Not Available", "No Data Reported"], np.nan, inplace=True)
    # drops NaN values
    df.dropna()
    
    # Convert columns to numeric where possible (necessary for summing)
    df = df.apply(pd.to_numeric, errors="ignore")
    
    # Select columns to sum
    column_sums = df[columns_to_sum].sum()
    # Find the column with the highest sum
    max_column = column_sums.idxmax()
    max_sum = round(column_sums.max(), 2)
    
    # the most consumed energy type by sector
    print(f"{max_sum} Trillion btu of {max_column}")
    return max_column, max_sum


def energy_cons_over_time(dfs, energy_cols, sector_names):
    '''
    Plot 1: Plots the total renewable energy consumed by each sector,
    over time. 
    '''
    sns.set_palette("Set2")
    plt.figure(figsize=(10, 6))
    for df, energy_col, sector_name in zip(dfs, energy_cols, sector_names):
        energy_data = pd.to_numeric(df[energy_col], errors="coerce")
        sns.lineplot(x = df["Annual Total"], y = energy_data, 
             label = sector_name)
    plt.title("Total Renewable Energy Consumed by Sector, Over Time")
    plt.xlabel("Years")
    plt.ylabel("Total Renewable Energy Consumed (Trillion btu)")
    plt.legend(title = "Sector")
    plt.show()  


def plot_energy_consumption(df, columns, sector_name):
    ''' 
    Plots 2-5: Plots the energy consumption of each sector, showing 
    different lines for different types of energy sources
    '''
    sns.set_palette("Set2")
    plt.figure(figsize=(10, 10))
    for col in columns:
        if col in df.columns:
            # converts column values to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # plots lines for each energy source used by the sector
            sns.lineplot(x=df["Annual Total"], y=df[col], label=col)
            
    # labels the lineplot and displays it
    plt.title(f"{sector_name} Sector Energy Consumption")
    plt.xlabel("Years")
    plt.ylabel("Consumption (Trillion btu)")
    plt.legend(title="Energy Types")
    plt.show()


def plot_afs_by_fuel_type(afs_counts, fuel_labels, afs_type):
    '''
    Plots 6-7: Plots the number of public/private alternative fuel stations 
    to see which fuel type most alternative fuel stations provide
    '''
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=afs_counts.values, y=afs_counts.index,
                       palette = "pastel")
    plt.title(f"Number of {afs_type} Alternative Fuel Stations by Fuel Type")
    plt.xlabel("Number of Stations")
    plt.ylabel("Fuel Type")
    # adds a legend
    fuel_types = [fuel_labels.get(code, code) for code in afs_counts.index]
    plt.legend(handles=bars.patches, labels=fuel_types, title="Fuel Types",
               loc="lower right")
    plt.show()
    

def plot_mean_energy_consumption_by_year(df, energy_column, sector_name):
    ''' 
    Plots 8-11: Ensures all data is numeric values and not strings. 
    Then calculates the mean energy consumption for each year by sector, 
    the correlation, and plots the regression.
    '''
    # makes sure 'Annual Total' is numeric and handle any missing values
    df['Annual Total'] = pd.to_numeric(df['Annual Total'], errors='coerce')
    df[energy_column] = pd.to_numeric(df[energy_column], errors='coerce')

    # groups by year and calculate the mean energy consumption for each year
    df_grouped = df.groupby('Annual Total')[energy_column
                                            ].mean().reset_index()
    
    # print r values (correlation) for each regression
    r = statistics.correlation(df_grouped["Annual Total"], 
                               df_grouped[energy_column])
    print(f"Correlation of year vs. mean total renewable energy consumption"
          f" for {sector_name} sector: {round(r, 3)}")
    
    # plots the regression
    plt.figure(figsize=(10, 6))
    sns.regplot(x='Annual Total', y=energy_column, data=df_grouped, 
                line_kws={'color':'mediumpurple'})
    
    # labels the plot and displays it
    plt.title(f"Relationship between Year and Mean Total Renewable Energy "
              f" Consumption Levels for the {sector_name} Sector")
    plt.xlabel("Year")
    plt.ylabel("Mean Total Renewable Energy Consumption Levels"
               " (Trillion Btu)")
    plt.show()
    
def main():
    '''
    Analyzing datasets to test Hypothesis 6
    Hypothesis 6: Given the growth of the EV industry, the electric power
    sector has consumed more renewable energy than the industrial, 
    commercial, and residential sectors.
    '''
    
    electric_df, total_elect = process_data(ELEC_FILE, ELEC_ENERGY,
                                            "Electric Power")
    indust_df, total_indust = process_data(IND_FILE, IND_ENERGY,
                                           "Industrial")
    rescomm_df, total_res = process_data(RESCOMM_FILE, RES_ENERGY,
                                         "Residential")
    rescomm_df, total_comm = process_data(RESCOMM_FILE, COMM_ENERGY,
                                          "Commercial")
     
        # Compare the totals to determine the highest
    totals = {
       "Electric Power": total_elect,
       "Industrial": total_indust,
       "Residential": total_res,
       "Commercial": total_comm
    }
   
    highest_sector = max(totals, key=totals.get)
    highest_value = totals[highest_sector]
   
    print(f"\nSector with the highest total renewable energy consumption:"
          f" {highest_sector} at {round(highest_value, 2)} Trillion btu")
    if highest_sector != "Electric Power":
        print("The electric power sector has not consumed more renewable "
                  "energy than all other sectors. Hypothesis 6 "
                  "is inaccurate.")
    if highest_sector == "Electric Power":
        print("The electric power sector has consumed more renewable "
                  "energy than all other sectors. Hypothesis 6 is accurate.")
    
    '''
    Plot 1
    Plots the total renewable energy consumed by each sector,
    over time. 
    '''
    # defines energy columns for plotting
    energy_columns = [ELEC_ENERGY, IND_ENERGY, RES_ENERGY, COMM_ENERGY]
    sector_names = ["Electric Power", "Industrial", "Residential",
                    "Commercial"]
    dfs = [electric_df, indust_df, rescomm_df, rescomm_df]
    energy_cons_over_time(dfs, energy_columns, sector_names)
    
    '''
    What type of energy is most consumed by the electric power, 
    residential, commercial, and industrial sectors, respectively 
    (cross-comparison)
    '''
    print("\nMost consumed renewable energy source by sector:")
    # filters out different energy source columns for each sector
    energy_cols_electric = electric_df.columns[6:-1]
    energy_cols_indust = indust_df.columns[1:-1]
    energy_cols_res = rescomm_df.columns[1:4]
    energy_cols_comm = rescomm_df.columns[5:13]

    analyze_sector_energy(electric_df, energy_cols_electric, "Electric Power")
    analyze_sector_energy(rescomm_df, energy_cols_res, "Residential")
    analyze_sector_energy(rescomm_df, energy_cols_comm, "Commercial")
    analyze_sector_energy(indust_df, energy_cols_indust, "Industrial")

    
    '''
    Plots 2-5
    How has energy consumption changed for different energy sources over 
    the years (1949 to present) in each sector?
    '''
    
    # Electric Power Sector Plot
    plot_energy_consumption(electric_df, energy_cols_electric,
                                            "Electric Power")    
    # Industrial Sector Plot
    plot_energy_consumption(indust_df, energy_cols_indust,
                                          "Industrial")
    # Residential Sector Plot
    plot_energy_consumption(rescomm_df, energy_cols_res,
                                       "Residential")
    
    # Commercial Sector Plot
    plot_energy_consumption(rescomm_df, energy_cols_comm,
                                        "Commercial")    
    
    '''Analyzing datasets to test Hypothesis 7'''
    '''Dealing with public AFS data'''
    
    '''
    Hypothesis 7: Given the growth of the EV industry, there are more 
    alternative fuel stations available to the public for electric cars 
    than for stations providing other fuels.
    '''
    
    '''
    Are there more alternative fuel stations available to the public 
    for electric cars than for stations providing other fuels?
    '''
    alt_df = pd.read_csv(AFS_FILE, low_memory = False)
    # shows all the types of fuels under the fuel type code column
    fuels = alt_df["Fuel Type Code"].unique()
    print(f"\nThe different types of fuels: {fuels}.")
    # filters out the alternative fuel stations (AFS) which are available 
    # to the public
    public_afs_df = alt_df[alt_df["Access Code"] == "public"]
    # counts how many alt. fuel stations are available to the public
    # for each type of fuel
    afs_counts = public_afs_df["Fuel Type Code"].value_counts()
    # finds name of fuel type with max alt. fuel stations
    max_afs_fuel = afs_counts.idxmax()
    # finds the maximum number of alt. fuel stations 
    max_afs_count = max(afs_counts)
    print(f"In 2021, the fuel type with the most alternative fuel stations"
          f" available to the public is '{max_afs_fuel}' with "
          f"{max_afs_count} stations.")
    if max_afs_fuel == "ELEC":
        print("In 2021, there are more alternative fuel stations available"
              " to the public providing electricity (for electric cars)"
              " than other fuels. Hypothesis 7 is accurate.")
    if max_afs_fuel != "ELEC":
        print("In 2021, there are more alternative fuel stations available"
              f" to the public for cars using {max_afs_fuel} as their fuel,"
              f" not electricity. Hypothesis 7 is inaccurate.")
        
    '''
    Plot 6
    Plots the number of public alternative fuel stations 
    to see which fuel type most alternative fuel stations provide
    '''
    fuel_labels = {
        'CNG': 'Compressed Natural Gas (CNG)',
        'E85': 'E85 (85% Ethanol, 15% Gasoline)',
        'LPG': 'Propane/Liquefied Petroleum Gas (LPG)',
        'BD': 'Biodiesel (BD)',
        'ELEC': 'Electricity (ELEC)',
        'HY': 'Hydrogen (HY)',
        'LNG': 'Liquefied Natural Gas (LNG)'
    }
    plot_afs_by_fuel_type(afs_counts, fuel_labels, "Public")    
    
    '''Dealing with private AFS data'''
    '''Do private stations provide more electricity than other fuels?'''
    # filters out the alternative fuel stations (AFS) which are available 
    # to the public
    private_afs_df = alt_df[alt_df["Access Code"] == "private"]
    # counts how many alt. fuel stations are available to the public
    # for each type of fuel
    afs_counts_priv = private_afs_df["Fuel Type Code"].value_counts()
    # finds name of fuel type with max alt. fuel stations
    max_afs_fuel_priv = afs_counts_priv.idxmax()
    # finds the maximum number of alt. fuel stations 
    max_afs_count_priv = max(afs_counts_priv)
    print(f"\nIn 2021, the fuel type with the most private alternative fuel"
          f" stations is '{max_afs_fuel_priv}' with {max_afs_count_priv}"
          f" stations.")
    if max_afs_fuel_priv == "ELEC":
        print("In 2021, there are more private alternative fuel stations"
              " providing electricity (for electric cars)"
              " than other fuels.")
    if max_afs_fuel_priv != "ELEC":
        print("In 2021, there are more private alternative fuel stations"
              f" for cars using {max_afs_fuel_priv} as their fuel, not"
              f" electricity.")
        
    '''
    Plot 7
    Plots the number of private alternative fuel stations 
    to see which fuel type most private alternative fuel stations provide
    '''
    plot_afs_by_fuel_type(afs_counts_priv, fuel_labels, "Private")
    
    # comparing public AFS to private AFS by fuel type
    if max_afs_count > max_afs_count_priv:
        print(f"\nThere are more public alternative fuel stations,"
              f" {max_afs_count}, which provide"
              f" {max_afs_fuel} than private alternative"
              f" fuel stations, {max_afs_count_priv}.")
    
    '''
    Do states with stations with longer access times have less alternative 
    fuel stations (electric)?
    '''
    # shows all the types of time availabilities for alt. fuel stations that 
    # provide electric energy
    elect_afs = alt_df[alt_df["Fuel Type Code"] == "ELEC"]
    elect_counts = elect_afs["Access Days Time"].value_counts()
    # shows how access times are too varied:
    # print(elect_counts)
    print("\nSince the data for access times is inconsistent,"
          " performing analysis to find out if states with longer access"
          " times have less AFS providing electricity would be inefficient.\n")
          
    '''
    Plots 8-11
    Plots the regression analysis showing the relationship
    between the year and mean total renewable energy consumption levels
    for the different sectors
    '''
    # Electric Power Sector: its regression analysis shows how the 
    # electric power sector's energy consumption levels are increasing over
    # the years (1949-2023)
    plot_mean_energy_consumption_by_year(electric_df, ELEC_ENERGY,
                                         "Electric Power")
    
    # Industrial Sector: its regression analysis shows how the 
    # industrial sector's energy consumption levels are increasing over
    # the years (1949-2023) but the curves in the scatter points suggest
    # there are slight variations in energy consumptions levels from 1990-2010
    plot_mean_energy_consumption_by_year(indust_df, IND_ENERGY,
                                         "Industrial")
    
    # Residential Sector: its regression analysis shows how the 
    # residential sector's energy consumption levels are not increasing 
    # nor decreasing over the years (1949-2023) because its scatter points
    # keep going up and down in an inconsistent sinusoidal pattern
    plot_mean_energy_consumption_by_year(rescomm_df, RES_ENERGY,
                                         "Residential")
    
    # Commercial Sector: its regression analysis shows how the 
    # commercial sector's energy consumption levels are increasing over
    # the years (1949-2023) but there is a large gap between 1990-2000
    # where energy consumption levels spiked as shown by the scatter points
    plot_mean_energy_consumption_by_year(rescomm_df, COMM_ENERGY,
                                         "Commercial")

    '''
    Further Research Ideas:
    Forecast future energy consumption levels by sector and future AFS 
    accessibility using correlation, regression analysis, monte carlo 
    simulations, and perhaps kFold.
    
    Gather climate change (i.e. emissions) data and look into how consumption 
    of each energy source and how increases in AFS impact climate change.
    '''

if __name__ == "__main__":
    main()
