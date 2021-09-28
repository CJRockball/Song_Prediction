# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 20:55:15 2021

@author: PatCa
"""


import sqlite3
import numpy as np
import pandas as pd


def make_db():
    #Connect to db
    conn = sqlite3.connect("music_data.db")
    conn.execute("PRAGMA foreign_keys = 1")
    cur = conn.cursor()
    #Create a table
    cur.execute("""DROP TABLE IF EXISTS song_data""")
    cur.execute("""DROP TABLE IF EXISTS genre_table""")
    cur.execute("""DROP TABLE IF EXISTS title_table""")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS song_data (
                    song_id INTEGER UNIQUE,
                    tempo REAL,
                    duration REAL,
                    genre INTEGER);""")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS genre_table (
                    genre_id INTEGER UNIQUE,
                    genre_name TEXT);""")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS title_table (
                    song_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL UNIQUE);""")
 
    
    # Write changes
    conn.commit()
    conn.close()
    
    #Make tuple for stock_hist
    genre_tuple = ((1,'soul and reggae'), (2,'pop'), (3,'punk'),(4,'jazz and blues'),
                   (5,'dance and electronica'), (6,'folk'),(7,'classic pop and rock'),
                   (8,'metal'))
    #Open database
    conn = sqlite3.connect("music_data.db")
    cur = conn.cursor()    
    #Inset new records
    cur.executemany("INSERT OR IGNORE INTO genre_table (genre_id,genre_name) VALUES (?,?);"""
                    ,genre_tuple)
    # Write change
    conn.commit()
    conn.close()       
    return

def populate_db(num_rows:int):
    #Get data
    source_feature = pd.read_csv('data/features.csv')
    source_label = pd.read_csv('data/labels.csv')
    
    # ----Check repeated rows
    result = pd.merge(source_feature, source_label, on="trackID")
    clean_data = result.dropna()
    clean_data = clean_data.drop_duplicates()
    
    label_dict = {'soul and reggae':1, 'pop':2, 'punk':3, 'jazz and blues':4, 
              'dance and electronica':5,'folk':6, 'classic pop and rock':7, 'metal':8}
    
    clean_data['genre_num'] = clean_data['genre'].replace(label_dict)
    clean_data = clean_data.iloc[:num_rows,:]
    
    
    #Make lists
    title_list = clean_data.title.astype(str).to_list()
    tempo_list = clean_data.tempo.round(0).to_list()
    duration_list = clean_data.duration.round(0).to_list()
    genre_list = clean_data.genre_num.round(2).to_list()
    
    #Make tuple for stock_hist
    for i in range(len(title_list)):
        name_list = ((title_list[i]),)
        #Open database
        conn = sqlite3.connect("music_data.db")
        cur = conn.cursor()    
        #Inset new records
        cur.execute("INSERT OR IGNORE INTO title_table (title) VALUES (?);""",name_list)
        # Write change
        conn.commit()
        conn.close()      
            
        #Open database
        conn = sqlite3.connect("music_data.db")
        cur = conn.cursor()
        cur.execute("""SELECT song_id FROM title_table WHERE title = ?;""", name_list)
        new_song_id = cur.fetchone()
        result_tuple = (new_song_id[0], tempo_list[i], duration_list[i], genre_list[i])
        #Inset new records
        cur.execute("INSERT OR IGNORE INTO song_data (song_id,tempo,duration,genre) VALUES (?,?,?,?);""",
                        result_tuple)
        # Write change
        conn.commit()
        conn.close()
    return



if __name__ == "__main__":  
    make_db()
    populate_db(100)








