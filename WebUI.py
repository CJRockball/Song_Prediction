# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 09:17:38 2021

@author: PatCa
"""

from flask import Flask, request,render_template

import webbrowser
import os
import sqlite3
import pandas as pd
import pickle
import joblib


def db_action(db_command, db_params="", one_line=False, db_name='music_data.db'):
    #Open db connection
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    
    #Run action
    cur.execute(db_command, db_params)
    if one_line:
        result = cur.fetchone()
    else:
        result = cur.fetchall()
    
    # Write change
    conn.commit()
    conn.close()
    return result

def get_genre_sample(submit_genre):
    # Get more from the same genre
    sample_genre = ((submit_genre),)
    db_command = """SELECT title_table.title \
                 FROM song_data \
                     LEFT JOIN title_table \
                         ON song_data.song_id = title_table.song_id \
                 WHERE genre = ? LIMIT(5);"""
    song_samples = db_action(db_command=db_command, db_params=sample_genre, one_line=False)

    #Put songs in list for easy handling
    song_list = []
    for i in song_samples:
        song_list.append(i[0])   
    return song_list

def check_song_in_db(song_name):
    
    param = ((song_name),)
    db_command = """Select title FROM title_table WHERE title = ?;"""
    name_check = db_action(db_command=db_command, db_params=param, one_line=True)     
    
    if name_check is None:
        return False
    else: return True
    
    
def nr_songs_in_db():
    #Check if song already in the db
    db_command = "Select count(title) FROM title_table;"
    num_titles = db_action(db_command, one_line=True)
    
    return  num_titles[0]    

def get_db_genre(song_name):
    param = ((song_name),)
    #Get genre and genre number
    db_command = """SELECT genre_table.genre_name, song_data.genre\
            FROM title_table \
                LEFT JOIN song_data \
                    ON title_table.song_id = song_data.song_id \
                        LEFT JOIN genre_table \
                            ON song_data.genre = genre_table.genre_id \
                        WHERE title_table.title = ?;"""
    genre_data = db_action(db_command, db_params=param, one_line=True)    
    return genre_data     

def load_song_to_db(song_title, tempo, duration, genre:int):
        title_tuple = ((song_title),)
        #Open database
        conn = sqlite3.connect("music_data.db")
        cur = conn.cursor()    
        #Inset new records
        cur.execute("INSERT OR IGNORE INTO title_table (title) VALUES (?);""",title_tuple)
        # Write change
        conn.commit()
        conn.close()      
        
        #Open database
        conn = sqlite3.connect("music_data.db")
        cur = conn.cursor()
        cur.execute("""SELECT song_id FROM title_table WHERE title = ?;""", title_tuple)
        new_song_id = cur.fetchone()
        result_tuple = (new_song_id[0], tempo, duration, genre)
        #Inset new records
        cur.execute("INSERT OR IGNORE INTO song_data (song_id,tempo,duration,genre) VALUES (?,?,?,?);""",
                        result_tuple)
        # Write change
        conn.commit()
        conn.close()
        return

def get_data_sample(sample_row):
    #Import pipeline and prediction model
    pipe = joblib.load('model_artifacts/pipe.joblib')
    file_name = "model_artifacts/xgb_baseline.pkl"
    imported_xgb_baseline = pickle.load(open(file_name,'rb'))
    
    #get dataset
    test_data = pd.read_csv('data/test.csv')
    test_data = test_data.iloc[sample_row,:].to_frame().T
    song_name = test_data['title'].values[0]
    tempo = test_data['tempo'].values[0]
    duration = test_data['duration'].values[0]
    
    if check_song_in_db(song_name):
        genre_data = get_db_genre(song_name)
        suggested_genre, num_genre = genre_data[0], genre_data[1]
    else:
        #preprocess data
        test_data = test_data.copy().drop(columns=['title', 'tags', 'trackID'])
        test_data = test_data.astype({'time_signature':int,'key':int,'mode':int})
        #Rename categorical values
        mode_dict = {0:'minor', 1:'major'}
        key_dict = {0:'C', 1:'D', 2:'E',3:'F', 4:'G', 5:'H', 6:'I', 7:'J', 8:'K', 9:'L',
                    10:'M', 11:'N'}
        test_data['mode'] = test_data['mode'].replace(mode_dict)
        test_data['key'] = test_data['key'].replace(key_dict)

        test_pipe = pipe.transform(test_data)
        y_pred = imported_xgb_baseline.predict(test_pipe)
        num_genre = y_pred[0]
        
        rev_label_dict = {1:'soul and reggae', 2:'pop', 3:'punk', 4:'jazz and blues', 
                      5:'dance and electronica',6:'folk', 7:'classic pop and rock', 8:'metal'}
        suggested_genre = rev_label_dict[y_pred[0]]
        load_song_to_db(song_name, tempo, duration, int(num_genre))         
    return (song_name, suggested_genre), num_genre


app = Flask(__name__)
app.secret_key = os.urandom(28)

@app.route('/', methods=['GET', 'POST'])
def index():    
    if request.method == 'POST':
       if request.form.get('Genre') == 'Genre':
           choose_sample = request.form['choose_sample']
           if len(choose_sample) < 1:
               return render_template("index.html")
           elif not choose_sample.isnumeric():
               return render_template("index.html")
           elif int(choose_sample) > 428:
               return render_template("index.html")
           elif int(choose_sample) < 0:
               return render_template("index.html")
           else:    
               return_tuple,num_genre = get_data_sample(int(choose_sample))
               no_titles = nr_songs_in_db()
               sample = get_genre_sample(str(num_genre))
               return render_template("suggest_genre2.html", return_tuple=return_tuple,
                               genre_samples=sample, no_titles=no_titles)
       
    elif request.method == 'GET':
        return render_template('index.html')
    
    else: return "GET/POST Error"
    

if __name__ == "__main__":  
    webbrowser.open('http://localhost:5000/')
    app.run(host='0.0.0.0')































