# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:14:54 2024

@author: Harshini Dinesh
DS 2500 Int Programming with Data
Utility functions that I commonly use. 
"""

import csv 
import os 

def read_csv(filename):
    '''Reads the CSV file and returns a list of lists'''
    data = []
    with open(filename, "r") as infile: 
        csvfile = csv.reader(infile)
        for row in csvfile:
            data.append(row)
    return data
#   to write to a file
#     with open(filename, "w") as outfile:
#         outfile.write("what you want to replace file with or write")

def lst_to_dct(lst):
    '''Convert from 2d list to dictionary where keys are the header (column)
    and data is list of the values in the column
    '''
    data_dct = {}
    for i in range(len(lst[0])):
        header_col = lst[0][i]
        data_dct[header_col] = []
        for row in lst[1:]:
            data_dct[header_col].append(row[i])
    return data_dct

def filter_data(filter_value, data_lst, other_lst):
    '''
    Filters other_lst data down by filter value, checking
    using data_lst
    
    Parameters
    -------
    filter_value: str 
    The value to look for
    
    data_lst: list of strings
    The column of data which has the filter_value that needs
    to be checked
    
    other_lst: list of obj
    The data to filter and return.

    Returns
    -------
    list of filtered data from other_list.

    '''
    return [other_lst[i] for i in range(len(data_lst))
        if filter_value.lower() in data_lst[i].lower()]

def median(old_lst):
    ''' Finds the median of the data in my list. This function will NOT
    update/change the old list.'''
    # protects original code from getting changed
    lst = old_lst.copy() 
    lst.sort()
    mid = len(lst)//2
    return lst[mid]

def get_filenames(dirname, ext = ".csv"):
    ''' given the name of a directory (string), return a list
        of paths to all the  ****.ext files in that directory
    '''
    filenames = []
    files = os.listdir(dirname)
    for file in files:
        if file.endswith(ext):
            filenames.append(dirname + "/" + file)
    return filenames

def get_file_path(default_filename):
    '''Gets a file path relative to the script
    or asks the user to input it.'''
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, default_filename)
    
    if not os.path.exists(file_path):
        file_path = input(f"Please provide the path for {default_filename}: ")
    
    return file_path

def main():
    lst_of_ints = [5,3,6,1,7,5,6,9,8]
    # tests functions
    print(f"Old list is {lst_of_ints}")
    print(f"Median is {median(lst_of_ints)}")
    print(f"After calculating median, list is {lst_of_ints}")



if __name__ == "__main__":
    main() 

