# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 20:30:10 2021

@author: PatCa
"""

def list_fcn(data):
    """
    Description
    -----------
    The function takes a list of tuples and returns index number of the elements grouped
    by elements that points to the same person.
    This function will not work if elements are list,tuple, arrays or dataframes etc.

    Parameters
    ----------
    data : List of tuples of str, int or float

    Returns
    -------
    List of list of indexes pointing to persons

    """
    
    # Useful variables and list
    tuple_length = len(data[0])
    data_length = len(data)
    idx_lst = [] #Final output list
    skip_list = [] #List to remove tuples to avoid double counting
    
    for nr,i in enumerate(data):
        #Iterate through data list to get indexes
        if nr not in skip_list:
            #Add current index to list
            current_index = [nr]
            #move current tuple elements to list and make type check
            current_tuple_list = read_tuple(i)
            
            #Go through all tuples infront of current to get same person tuples
            temp_current_index, temp_skip_list = tuple_search(nr+1,data_length,
                                        current_tuple_list, data, tuple_length)
            
            #Save all from the pointing to the current person
            current_index.extend(temp_current_index)
            #Save the tuple number of the indexes moved to list so not to double count
            skip_list.extend(temp_skip_list)
            #Add current indexes to master list
            idx_lst.append(current_index)
       
    return idx_lst


def tuple_search(current_pos, data_length, current_tuple_list, data, tuple_length):
    """
    Description
    -----------
    Main function, goes through data list and compares current tuple to all 
    other tuples. Check that tuples are the same length as the first one.

    Parameters
    ----------
    current_pos: Position of current tuple. Start at position +1
    data_length: length of Data list
    current_tuple_list: Elements of current tuple
    data: List of tuples
    tuple_length: length of first tuple

    Returns
    -------
    List of indexes to sane
    List of indexes to skip

    """
    temp_current_index = []
    temp_skip_list = []
    for m in range(current_pos,data_length):
        if m not in temp_skip_list:
            #Get list of current o_tuple
            other_tuple_list = read_tuple(data[m])
            
            #Check tuple length
            if len(other_tuple_list) == tuple_length:
                
                #Compare the two lists in any items are identica
                identical = check_same(current_tuple_list,other_tuple_list)
    
                #If there were identical elements, save index and remove from future search
                #Also compare the new tuple with all tuples infront
                if identical:
                    #Save index of tuple for list
                    temp_current_index.append(m)
                    temp_skip_list.append(m)
                    #Iterative call to check new tuple
                    temp_new_index, temp_new_skip = tuple_search(m+1,data_length,
                                                other_tuple_list, data,tuple_length)
                    #Save new new indexes
                    temp_current_index.extend(temp_new_index)
                    temp_skip_list.extend(temp_new_skip)
            #Raise error if tuples have different size
            else: raise IndexError("Tuples in Data are of different length")
    return temp_current_index, temp_skip_list


def read_tuple(tuple1):
    """
    Description
    -----------
    Takes in tuple, checks types of elemnts and puts elements in a list

    Parameters
    ----------
    tuple1: takes in a tuple of information

    Returns
    -------
    List of elements from tuple

    """
    item_list = []
    for tup in tuple1:
        if  type(tup) is int or type(tup) is float or type(tup) is str:
            item_list.append(str(tup))
        else: raise TypeError("At least one element in the list is not str, int or float")
    return item_list


def check_same(tuple1, tuple2):
    """
    Description
    -----------
    Takes in two tuples and determines if there are identical elements between
    two tuples

    Parameters
    ----------
    tuple1: tuple of information
    tuple2: tuple of information

    Returns
    -------
    Identical: Boolean. True if there are identical elements

    """
    identical = False
    for j in range(len(tuple1)):
        if tuple1[j] == tuple2[j]:
            identical = True   
    return identical
