#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 00:55:53 2024

@author: hershjoshi
"""
#importing generic libraries
import statistics 
import seaborn as sns 
import csv
import os
from datetime import datetime
import operator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")



df_electricity_data = pd.read_csv('Electricity_data.csv')
df_fuel_data = pd.read_csv('Fuel_data.csv')


YEARS = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', 
          '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
vehicle_types = ['Electricity data', 'Fuel data']

def clean_data():
    '''Cleans electricity data'''
    df_electricity_data_clean = df_electricity_data[['Year', 'State', 
                                                     'Fuel Consumed', 
                                                     'Weight Class', 
                                                     'Number of Vehicles']]
    df_electricity_data_clean['Electric State'] = df_electricity_data_clean[
        'State']
    df_electricity_data_clean[
        'Electric Fuel Consumed'] = df_electricity_data_clean['Fuel Consumed']
    df_electricity_data_clean[
        'Electric Weight Class'] = df_electricity_data_clean['Weight Class']

    df_fuel_data_clean = df_fuel_data[['Year', 'State', 'Fuel Consumed',
                                       'Weight Class', 'Number of Vehicles']]
    df_fuel_data_clean['Diesel State'] = df_fuel_data_clean['State']
    df_fuel_data_clean['Diesel Fuel Consumed'] = df_fuel_data_clean[
        'Fuel Consumed']
    df_fuel_data_clean['Diesel Weight Class'] = df_fuel_data_clean[
        'Weight Class']
     #features needed here   
    
    return df_electricity_data_clean, df_fuel_data_clean
    
def lst_to_dct(lst):
    ''' given a 2d list, create and return a dictionary where each key is 
        a header and each value is the rest of that column.
        Assumes the first row of the 2d list is the header
    '''
    dct = {}
    for header in lst[0]:
        dct[header] = []
    for row in lst[1:]:
        for i in range(len(row)):
            dct[lst[0][i]].append(row[i])
    return dct


def emissions_vehicles():
    """
    This function loads and organizes vehicle emissions data based on weight 
    category and year.
    Parameters:
    directory (str): The directory where the data files are located.
    Returns:
    dict: A dictionary with data organized by weight category and year.
    """

    df_electricity_data_clean, df_fuel_data_clean = clean_data()
    # Define weight class categories
    weight_class_mapping = {
        'Light Duty': 'Light',
        'Medium Duty': 'Medium',
        'Heavy Duty': 'Heavy'
    }

    # Apply the weight class mapping
    df_electricity_data_clean['Weight Category'] = df_electricity_data_clean[
        'Weight Class'].map(lambda wc: weight_class_mapping.get(wc, 'Other'))
    df_fuel_data_clean['Weight Category'] = df_fuel_data_clean[
        'Weight Class'].map(lambda wc: weight_class_mapping.get(wc, 'Other'))
    data_by_weight_and_year_electric = {}
    data_by_weight_and_year_diesel = {}


    # Group the data by Weight Category and Year
    grouped_data_electric = df_electricity_data_clean.groupby([
        'Weight Category', 'Year'])
    grouped_data_diesel = df_fuel_data_clean.groupby([
        'Weight Category', 'Year'])

    
    # Store the data in the dictionaries for each dataset 
    for (weight_category, year), group in grouped_data_electric:
        if weight_category not in data_by_weight_and_year_electric:
            data_by_weight_and_year_electric[weight_category] = {}
        data_by_weight_and_year_electric[weight_category][year] = group

    for (weight_category, year), group in grouped_data_diesel:
        if weight_category not in data_by_weight_and_year_diesel:
            data_by_weight_and_year_diesel[weight_category] = {}
        data_by_weight_and_year_diesel[weight_category][year] = group
        
    return data_by_weight_and_year_electric, data_by_weight_and_year_diesel
    print(data_by_weight_and_year_electric)


def scatter_plot_comparison():
    '''Scatter Plot to compare electricity data with fuel data'''
    df_electricity_data_clean, df_fuel_data_clean = clean_data()
    
    # Ensure data types are correct
    # df_electricity_data_clean['Electric Year'] = pd.to_numeric(
    # df_electricity_data_clean['Electric Year'], errors='coerce')
    df_electricity_data_clean['Electric Fuel Consumed'] = pd.to_numeric(
        df_electricity_data_clean['Electric Fuel Consumed'])
    df_electricity_data_clean.dropna(subset=['Year', 'Electric Fuel Consumed']
                                     , inplace=True)
    
    # Define colors for each weight class
    color_map = {
        'Heavy': 'red',
        'Medium': 'yellow',
        'Light': 'green'
    }
    
    plt.figure(figsize=(12, 8))
    
    # Plot each weight class with a different color
    for weight_class, color in color_map.items():
        subset = df_electricity_data_clean[df_electricity_data_clean[
            'Electric Weight Class'] == weight_class]
        plt.scatter(subset['Year'], subset['Electric Fuel Consumed'], 
                    color=color, label=weight_class, alpha=0.7,
                    edgecolors='w')
    
    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Fuel Consumed')
    plt.title('Scatter Plot of Fuel Consumption by Weight Class Over Years')
    plt.legend(title='Weight Class')
    plt.grid(True)
    
    # Show plot
    plt.show()


def plot_vehicles_by_weight_class(data):
    '''Plots vehicle by weight class'''
    # Filter and group data by State and Weight Class
    grouped_data = data.groupby(['State', 'Weight Class'])[
        'Number of Vehicles'].sum().unstack()
    
    # Plotting
    grouped_data.plot(kind='bar', stacked=True, figsize=(14, 8))
    
    # Add labels and title
    plt.xlabel('State')
    plt.ylabel('Number of Vehicles')
    plt.title('Number of Vehicles by Weight Class in Each State')
    plt.legend(title='Weight Class')
    plt.xticks(rotation=45, ha='right')
    
    # Show plot
    plt.tight_layout()
    plt.show()


def main():
    df_electricity_data_clean, df_fuel_data_clean = clean_data()
    print(emissions_vehicles())
    
    #Assuming df_electricity_data_clean is already cleaned and available
    print(plot_vehicles_by_weight_class(df_electricity_data_clean))
    
    #Assuming df_electricity_data_clean is already cleaned and available
    print(plot_vehicles_by_weight_class(df_fuel_data_clean))

    
if __name__ == "__main__":
    main()