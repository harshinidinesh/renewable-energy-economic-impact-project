#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: Hang Hang
Course: DS2500
Assignment: final project
Date: 8/16/2024
File name: final_project_hang.py
"""

import pandas as pd
import geopandas as gpd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, classification_report
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# File paths
cost_of_charging_xlsx = 'cost_of_charging_2019_v1.xlsx'
energy_generation_csv = 'net_generation_for_all_sectors.csv'
costal_inland_states_csv = 'costal_and_inland_states.csv'
median_household_income_csv = 'median_household_income.csv'
us_states_map = 'cb_2018_us_state_500k.zip'

# Column names and constants
STATE_AND_ENERGY_SOURCE = 'state and energy generation source'
GENERATION_STATE = 'state'
ENERGY_SOURCE = 'energy generation source'
LOCATION_STNAME = 'state'
COASTAL_INLAND_STATUS = 'coastal/inland status'
RENEWABLE_PERC = 'renewable_perc'
COST_CHARGING_STNAME = 'state'
DCFC_STATION_P1_LCOC = 'baseline_dcfc_station_p1_lcoc'

# sheets' names within excel
B_BEV = 'B-BEV'
C_PHEV = 'C-PHEV'

# Define the renewable sources
RENEWABLE_SOURCES = ['conventional hydroelectric', 'other renewables', 
                     'wind', 'geothermal', 'biomass', 
                     'wood and wood-derived fuels', 'other biomass',
                     'all solar']

# Excluded areas for mapping
EXCLUDE_AREAS = ['Puerto Rico', 'United States Virgin Islands', 
    'District of Columbia', 'Guam', 
    'Commonwealth of the Northern Mariana Islands', 
    'American Samoa', 'Alaska', 'Hawaii']

# Grouped regions
REGIONS = {'Northeast': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA', 
                         'DC'],
           'Midwest': ['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 
                       'OH', 'SD', 'WI'], 
           'South': ['AL', 'AR', 'DE', 'FL', 'GA', 'KY', 'LA', 'MD', 'MS', 
                     'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV'],
           'West': ['AK', 'AZ', 'CA', 'CO', 'HI', 'ID', 'MT', 'NV', 'NM', 'OR',         
                    'UT', 'WA', 'WY']}

# List of regions to exclude for the knn classifier's 
# features and label preparation
EXCLUDE_REGIONS = [
    'United States', 'Middle Atlantic', 'East North Central', 
    'West North Central',
    'South Atlantic', 'East South Central', 'West South Central', 'Mountain', 
    'Pacific Contiguous', 'Pacific Noncontiguous']

def load_and_clean_data(filepath, drop_columns = None, header_row = 0):
    '''
    Load and clean data from CSV or Excel file.

    Parameters
    ----------
    filename (str): The file name to clean and load data for.
    drop_columns (list, optional): A list of columns to be dropped from 
    the DataFrame. Default is None.
    header_row (int, optional): The row number to be used as the header. 
    Default is 0.

    Returns
    -------
    DataFrame or dict: A cleaned pandas DataFrame or a dictionary of DataFrames 
    if the input is an Excel file with multiple sheets.
    '''
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, header = header_row)
        if drop_columns:
            df = df.drop(columns = drop_columns)
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x) 
        return df
    elif filepath.endswith('.xlsx'):
        sheets_dict = pd.read_excel(filepath, sheet_name = None)
        for sheet_name, df in sheets_dict.items():
            df = df.drop(df.index[-1], axis = 0) if sheet_name == B_BEV else df 
            sheets_dict[sheet_name] = df
        return sheets_dict
    
def load_clean_energy_gen_data(filename):
    '''
    Loads and cleans the energy generation data from the CSV file. After 
    executing this function, only 'state', 'energy generation source', and 
    year columns will be displayed, with year columns as integers.

    Parameters
    ----------
    filename (str): The file name for the energy generation data.

    Returns
    -------
    cleaned_df (dataframe): The cleaned and filtered energy generation 
    dataframe.
    '''
    df = pd.read_csv(filename, header = 4)
    df = df.drop(columns = ['units', 'source key']).drop(index = [5, 6])
    # drop the rows where 'state and energy generation source' contains
    # only the region name
    df = df[df[STATE_AND_ENERGY_SOURCE].str.contains(':', na = False)]
    # split the 'state and energy source info' column into two separate 
    # columns and drop the original column 
    df[[GENERATION_STATE, ENERGY_SOURCE]] = df[STATE_AND_ENERGY_SOURCE]. \
    str.split(':', expand = True) 
    df = df.drop(columns = [STATE_AND_ENERGY_SOURCE])
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # convert the year column header to int and reorder the columns as pandas 
    # add the new columns to the end
    df.columns = [int(col) if col.isdigit() else col for col in df.columns]
    cols = [GENERATION_STATE, ENERGY_SOURCE] + [
        col for col in df.columns if isinstance(col, int)]
    cleaned_df = df[cols]
    return cleaned_df

def calculate_renewable_percentage(df, state_name, year):
    '''
    Given the energy generation data for all sources dataframe, calculate the
    total percentage of the renewable energy generation out of the total energy 
    generation for each state for the specified year.

    Parameters
    ----------
    df (dataframe): The renewable energy generation dataframe.
    state (str): The name of the state for which to calculate the renewable
    energy percentage.
    year (int): The specified year for calculating the renewable energy
    percentage.

    Returns
    -------
    float: The percentage of renewable energy generation per state.
    '''
    # create the copy to aviod modifying a dataframe that is a slice of 
    # another dataframe
    state_df = df[df[GENERATION_STATE] == state_name].copy()
    renewable_df = state_df[state_df[ENERGY_SOURCE].isin(
        RENEWABLE_SOURCES)].copy()
    # Replace NaN, "NM", and "--" with 0 in the specified year column
    state_df.loc[:, year] = state_df[year].replace(['NM', '--'], 0).fillna(0)
    renewable_df.loc[:, year] = renewable_df[year].replace(['NM', '--'], 
                                                           0).fillna(0)
    state_df.loc[:, year] = pd.to_numeric(state_df[year], errors='coerce')
    renewable_df.loc[:, year] = pd.to_numeric(renewable_df[year], 
                                              errors='coerce')        
    renewable_percentage = (renewable_df[year].sum() / 
                            state_df[year].sum()) * 100
    return renewable_percentage

def merge_energy_gen_with_location(energy_gen_df, state_loc_df, year):
    '''
    Merge the energy generation dataframe with the state location dataframe 
    and calculates the renewable energy percentage for each state.

    Parameters
    ----------
    energy_gen_df (dataframe): The energy generation data.
    state_loc_df (dataframe): The state location data (coastal/inland).
    year (int): The year of the energy generation data we select to merge with
    state_loc_df.
    
    Returns
    -------
    merged_df (dataframe): The merged dataframe containing the renewable 
    energy percentage and location data (coastal/inland) for each state.
    '''
    # create a new dataframe (state_energy_gen_df) that pairs each state 
    # with its renewable energy percentage
    state_energy_gen_df = pd.DataFrame({'state': state_loc_df[LOCATION_STNAME], 
                             'renewable_perc': state_loc_df[LOCATION_STNAME].
                             apply(lambda x: calculate_renewable_percentage(
                                 energy_gen_df, x, year))})
    merged_data = pd.merge(state_energy_gen_df, state_loc_df, on = 'state')              
    return merged_data

def correlation_location_with_renewables(merged_df):
    '''
    Calculates the correlation between state location (coastal/inland) 
    and renewable energy production.

    Parameters
    ----------
    merged_df (DataFrame): The merged dataframe containing the renewable 
    energy percentage and location data (coastal/inland) for each state.

    Returns
    -------
    float: The correlation between the renewable energy generation for a state
    and the geographical location of that state.
    '''
    merged_df['location_binary'] = merged_df[COASTAL_INLAND_STATUS]. \
    map({'Coastal': 1, 'Inland': 0})
    correlation = merged_df[[RENEWABLE_PERC, 'location_binary']].corr(). \
    iloc[0, 1]
    return correlation

def plot_box_plot(merged_df, year):
    '''
    Plots a box plot to visualize the distribution of renewable energy
    percentages by state location (coastal/inland).

    Parameters
    ----------
    merged_df (dataframe): The dataframe containing the renewable energy 
    percentage and location data for each state.
    year (int): The year for which the data is being plotted.

    Returns
    -------
    None
    '''
    plt.figure(dpi=400)
    sns.boxplot(x=merged_df[COASTAL_INLAND_STATUS], 
                y=merged_df[RENEWABLE_PERC], palette='Set2')
    plt.xlabel('State Location')
    plt.ylabel('Renewable Energy Percentage (%)')
    plt.title(f'Renewable Energy Percentage by State Location ({year})')
    plt.grid(True)
    plt.show()

def plot_renewable_energy_map(merged_df, year):
    '''
    Plots a U.S. map with states colored according to the percentage of 
    renewable energy generated for a specified year.

    Parameters
    ----------
    merged_df (dataframe): The dataframe containing the renewable 
    energy percentage for each state.
    year (int): The year for which the renewable energy percentage is plotted.

    Returns
    -------
    None
    '''
    us_states = gpd.read_file(us_states_map)
    us_states = us_states[us_states['NAME'].isin(EXCLUDE_AREAS) == False]
    # Merge the two DataFrames based on the 'NAME' index and the 'state' index
    gdf = us_states.set_index('NAME').join(merged_df.set_index('state'))
    # Plot using a fixed aspect ratio to prevent distortion
    fig, ax = plt.subplots(1, 1, figsize = (18, 8), dpi = 400)
    gdf.boundary.plot(ax = ax, linewidth = 1) 
    gdf.plot(column = RENEWABLE_PERC, cmap = 'YlGnBu', linewidth = 0.8, 
             edgecolor = '0.8', legend = True, ax = ax, 
             legend_kwds = {'label': 'Percentage of Renewable Energy \
Generation(%)', 'orientation': 'vertical'})  
    # Add percentage labels on each state
    for x, y, label in zip(gdf.geometry.centroid.x, 
                           gdf.geometry.centroid.y, 
                           gdf[RENEWABLE_PERC].round(2).astype(str) + '%'):
        ax.text(x, y, label, fontsize = 6, ha = 'center', color = 'red')   
    # set title and adjust axis limits
    ax.set_title(f'Renewable Energy Percentage by State ({year})', fontsize=22)
    # Adjust to focus on the contiguous U.S.
    ax.set_xlim([-126, -67])
    ax.set_ylim([25, 50])
    ax.set_aspect('auto') 
    ax.axis('off')
    plt.show()
    
def perform_linear_regression(df, energy_source):
    '''
    Train and evaluate a linear regression model on energy generation data 
    for a specific energy source in Vermont, then calculate the RMSE.

    Parameters
    ----------
    df (dataframe): The cleaned Vermont energy generation data.
    energy_source (str): The energy source to analyze (e.g., 'wind', 'solar').

    Returns
    -------
    tuple: The RMSE value, trained model, test features, actual test values, 
    and predicted test values.
    '''
    source_df = df[df[ENERGY_SOURCE] == energy_source]
    # Prepare the data for regression: X is the years, y is the energy 
    # generation values
    X = source_df.columns[2:].astype(int).values.reshape(-1, 1)
    y = source_df.iloc[0, 2:].replace(['NM', '--'], 0).astype(float).values. \
    reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
    return rmse, model, X_test, y_test, y_pred

def residual_plot_energy_data(X_test, y_test, y_pred):
    '''
    Create a residual plot to assess the goodness-of-fit for the linear 
    regression model.

    Parameters
    ----------
    X_test (Series): The test data features (years).
    y_test (Series): The actual values for the test data (energy generation).
    y_pred (Series): The predicted values from the model.
    
    Returns
    -------
    None.
    '''
    plt.figure(dpi=400)
    sns.residplot(x=X_test, y=y_test, color="blue")
    plt.xticks(ticks=range(2001, 2024), rotation=45, ha='right')
    plt.xlabel('Year')
    plt.ylabel('Residuals (Energy Generation) in Thosuand MWh')
    plt.title('Residual Plot for Vermont Energy Generation')
    plt.grid(True)
    plt.show()

def analyze_cost_variation(sheet_name, cost_col, state_col):
    '''
    Analyze the levelized cost of charging battery electric vehicle (BEV) 
    in both residential and workplace settings for each state.

    Parameters
    ----------
    sheet_name (str): The sheet name containing the cost data.
    cost_col (str): The column containing the charging cost (either 
    residential or workplace).
    state_col (str): The column containing the state names.

    Returns
    -------
    dict: A dictionary with statistics, highest and lowest cost state.
    '''
    cost_df = load_and_clean_data(cost_of_charging_xlsx)[sheet_name]
    # costs stats
    cost_stats = cost_df[cost_col].describe()
    highest_cost_state = cost_df.loc[cost_df[cost_col].idxmax(), state_col]
    lowest_cost_state = cost_df.loc[cost_df[cost_col].idxmin(), state_col]
    return cost_stats, highest_cost_state, lowest_cost_state

def analyze_regional_variation(sheet_name, cost_column, state_column):
    '''
    Analyze the average Levelized Cost of Charging (LCOC) for EVs across 
    different regions.

    Parameters
    ----------
    sheet_name (str): The name of the sheet containing the BEV LCOC data.
    cost_column (str): The column name within the sheet representing the LCOC 
    values (e.g., 'baseline_residential_lcoc', 'baseline_workplace_lcoc').
    state_column (str): The state name columns.

    Returns
    -------
    regional_avg_lcoc (Series): A Series where the index represents the 
    regions and the values represent the average LCOC for each region.
    '''
    cost_df = load_and_clean_data(cost_of_charging_xlsx, B_BEV)[B_BEV]
    cost_df = cost_df.drop(cost_df.index[-1])
    # maps each state in the state_column to its corresponding region using 
    # the predefined REGIONS dictionary
    cost_df['Region'] = cost_df[state_column].map(
        {state: region for region, states in REGIONS.items() for state in 
         states})
    # groups the dataframe by the 'Region' column and calculate average 
    # LCOC for each group (region)
    regional_avg_lcoc = cost_df.groupby('Region')[cost_column].mean()
    return regional_avg_lcoc

def plot_regional_lcoc_comparison(regional_lcoc_residential, 
                                  regional_lcoc_workplace):
    '''
    Plots a bar chart comparing the average Levelized Cost of Charging (LCOC) 
    for BEVs in residential and workplace settings by region.

    Parameters
    ----------
    regional_lcoc_residential (Series): The LCOC for residential setting 
    by region.
    regional_lcoc_workplace (Series): The LCOC for workplace setting 
    by region.

    Returns
    -------
    None.
    '''
    plt.figure(dpi=400)
    lcoc_data = pd.DataFrame({'Residential': regional_lcoc_residential,
                              'Workplace': regional_lcoc_workplace})
    lcoc_data.plot(kind='bar', figsize=(10, 6))
    plt.title('Regional Comparison of Levelized Cost of Charging (LCOC) for \
BEV')
    plt.ylabel('Average LCOC ($)')
    plt.xlabel('U.S. Regions')
    plt.grid(True)
    plt.xticks(rotation=0)
    plt.show()
    
def prepare_features_and_labels(df, year):
    '''
    Prepare the features and labels for the KNN classifier from the dataset.
    Features: the amounts of various renewable energy sources generated per
    state within the year range. Labels: categorical values like "High wind" 
    or "High biomass."

    Parameters
    ----------
    df (dataframe): The energy generation dataset.
    year (int): The year to prepare the features and labels.

    Returns
    -------
    tuple: A tuple containing the features (X) and labels (y).
    '''
    df = df.replace(['NM', '--'], 0).fillna(0)
    # Filter to include only renewable sources and exclude unwanted regions
    renewable_df = df[df[ENERGY_SOURCE].isin(RENEWABLE_SOURCES)]
    renewable_df = renewable_df[renewable_df[GENERATION_STATE].isin(
        EXCLUDE_REGIONS) == False]
    # Create the features by selecting the renewable sources for a year
    X = renewable_df.pivot_table(
        index=GENERATION_STATE, columns=[ENERGY_SOURCE], values=year, 
        aggfunc='sum')
    X = X.apply(pd.to_numeric, errors='coerce')
    # Identify the dominant renewable energy source for the state
    y = X.idxmax(axis=1).apply(lambda x: f'High {x}')
    return X, y

def find_best_k(X, y, k_range = (4, 11), n_splits = 5, random_state = 0):
    '''
    Find the best value of k for KNN using cross-validation.

    Parameters
    ----------
    X (DataFrame): The features.
    y (Series): The labels.
    k_range (tuple): A tuple specifying the range of k values to test.
    n_splits (int): The number of splits for cross-validation.
    random_state (int): The random state for reproducibility.

    Returns
    -------
    tuple: A tuple containing the optimal k for recall, the lowest recall,
    the optimal k for precision, the lowest precision, and the results 
    dataframe.
    '''
    kf = KFold(n_splits = n_splits, random_state=random_state, shuffle = True)
    results = {'k': [], 'mean_precision': [], 'mean_recall': []}
    highest_precision = -1
    highest_recall = -1
    lowest_precision = 1
    lowest_recall = 1
    for k in range(k_range[0], k_range[1]):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_validate(knn, X, y, cv = kf, scoring=['precision_macro',
                                                             'recall_macro'])
        mean_precision = scores['test_precision_macro'].mean()
        mean_recall = scores['test_recall_macro'].mean()    
        results['k'].append(k)
        results['mean_precision'].append(mean_precision)
        results['mean_recall'].append(mean_recall)
        
        if mean_precision > highest_precision:
            highest_precision = mean_precision
            optimal_k_precision = k
        if mean_recall > highest_recall:
            highest_recall = mean_recall
            optimal_k_recall = k
        if mean_precision < lowest_precision:
            lowest_precision = mean_precision
        if mean_recall < lowest_recall:
            lowest_recall = mean_recall
            
    return optimal_k_recall, lowest_recall, optimal_k_precision, \
highest_precision, pd.DataFrame(results) 

def evaluate_knn(X_train, y_train, X_test, y_test, k):
    '''
    Train and evaluate the KNN classifier.

    Parameters
    ----------
    X_train (dataframe): The training features.
    y_train (Series): The training labels.
    X_test (dataframe): The testing features.
    y_test (Series): The testing labels.
    k (int): The number of neighbors for KNN.

    Returns
    -------
    tuple: A tuple containing the trained KNN model, the predicted labels, 
    the classification report, and accuracy score.
    '''
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = knn.score(X_test, y_test)
    return knn, y_pred, report, accuracy

def plot_confusion_matrix(y_test, y_pred, labels, k_type, k_value):
    '''
    Plot a heatmap of the confusion matrix for the given true and predicted 
    labels.

    Parameters
    ----------
    y_test (Series): The testing labels.
    y_pred (Series): The predicted labels.
    labels (list): The list of label names for display on the matrix.
    k_type (str): The type of optimal k used ('precision' or 'recall').
    k_value (int): The value of k used for the KNN classifier.
    
    Returns
    -------
    None.
    '''
    # Shorten the labels
    display_labels = [label.replace("High conventional hydroelectric", "High hydroelectric") for label in labels]
    plt.figure(dpi = 400)
    cm = metrics.confusion_matrix(y_test, y_pred, labels=labels)
    sns.heatmap(cm, cmap='Blues', annot=True, xticklabels=display_labels, 
                yticklabels=display_labels)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix For Year Of 2023 (Optimal k {k_type} = \
{k_value})')
    plt.show()

def main():

  # load the data
  energy_gen_df = load_clean_energy_gen_data(energy_generation_csv)
  state_loc_df = load_and_clean_data(costal_inland_states_csv, header_row = 2)
  
  # tests the renewable energy percentage function on specific state
  state_MA = 'Massachusetts'
  year = 2023
  ma_renewable_percentage = calculate_renewable_percentage( 
       energy_gen_df, state_MA, year)
  print(f'Renewable energy percentage for {state_MA} in {year}: '
           f'{ma_renewable_percentage:.4f}%')

  # print the merged dataset
  merged_dataset = merge_energy_gen_with_location(energy_gen_df, state_loc_df, 
                                                   2023)
  
  # print the correlation between state location (coastal/inland) 
  # and renewable energy production
  correlation = correlation_location_with_renewables(merged_dataset)
  print(f'Correlation between the state location and renewable energy '
        f'generation percentage: {correlation:.4f}')
  
  # plot the correlation between state locations and energy generation
  plot_box_plot(merged_dataset, 2023)
  
  # plot the renewable energy generation map
  plot_renewable_energy_map(merged_dataset, 2023)
  
  # get the values of interest for the cost of charging BEVs
  residential_stats, highest_residential_state, lowest_residential_state = \
  analyze_cost_variation(B_BEV, 'baseline_residential_lcoc', 'state')
  workplace_stats, highest_workplace_state, lowest_workplace_state = \
  analyze_cost_variation(B_BEV, 'baseline_workplace_lcoc', 'state')
  print('Residential Stats:', residential_stats) 
  print('Workplace Stats:', workplace_stats) 
  print('Highest Residential State:', highest_residential_state) 
  print('Lowest Residential State:', lowest_residential_state) 
  print('Highest Workplace State:', highest_workplace_state) 
  print('Lowest Workplace State:', lowest_workplace_state)
 
  # Plot the bar plot to compare the residential and workplace charging cost
  residential_lcoc = analyze_regional_variation(B_BEV, 
                                                 'baseline_residential_lcoc', 
                                                 'state')
  workplace_lcoc = analyze_regional_variation(B_BEV, 
                                               'baseline_workplace_lcoc',
                                               'state')
  plot_regional_lcoc_comparison(residential_lcoc, workplace_lcoc)
  
  # perform the linear regression analysis on Vermont's wind energy
  vermont_df = energy_gen_df[energy_gen_df[GENERATION_STATE] == 'Vermont']
  rmse, model, X_test, y_test, y_pred = perform_linear_regression(vermont_df, 
                                                                  'wind')
  print(f'RMSE for Vermont wind energy: {rmse:.2f}')
  
  # Plot the residual plot
  residual_plot_energy_data(X_test, y_test, y_pred)
  
  # prepare features and labels
  X, y = prepare_features_and_labels(energy_gen_df, 2023)
  print(X)
  print(y)
  
  # Find the best k values
  optimal_k_recall, lowest_recall, optimal_k_precision, highest_precision, \
  results_df = find_best_k(X, y)
  print(f'Optimal k for overall recall for 2023: {optimal_k_recall}')
  print(f'Optimal k for overall precision in 2023: {optimal_k_precision}')
  
  # Split the data
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

  # Evaluate the KNN classifier with optimal k precision
  knn_model_recall, y_pred_recall, report_recall, accuracy_recall = \
  evaluate_knn(X_train, y_train, X_test, y_test, optimal_k_recall)
  print('Accuracy of the KNN classifier (recall):', accuracy_recall.round(4))
  print("Classification Report:\n", report_recall)
  
  # plot the confusion matrix with optimal k precision
  unique_labels = y.unique()
  plot_confusion_matrix(y_test, y_pred_recall, labels = unique_labels, 
                        k_type='recall', k_value = optimal_k_recall)
  
  knn_model_precision, y_pred_precision, report_precision, \
  accuracy_precision = evaluate_knn(X_train, y_train, X_test, y_test, 
                                    optimal_k_precision)
  print('Accuracy of the KNN classifier (precision):', 
        accuracy_precision.round(4))
  print("Classification Report:\n", report_precision)
  
  # plot the confusion matrix with optimal k recall
  plot_confusion_matrix(y_test, y_pred_precision, labels = unique_labels, 
                        k_type='precision', k_value = optimal_k_precision)

if __name__ == "__main__": 
    main()
            
    