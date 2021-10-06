# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 13:17:25 2021

@author: PatCa
"""

from flask import Flask, make_response, jsonify, request
from flask_restful import Resource, Api
import sqlite3

def db_action(db_name, db_command, db_params=""):
    #Open db connection
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    
    #Run action
    cur.execute(db_command, db_params)
    result = cur.fetchall()
    
    # Write change
    conn.commit()
    conn.close()
    return result


def get_five():
    #Get the five latest title entries in the database    
    db_name = str("music_data.db")
    db_command = str("""SELECT song_id,title \
                     FROM title_table \
                     ORDER BY song_id DESC
                     LIMIT(5);""")
    result = db_action(db_name, db_command)

    #Add category names song nr in the db and song title
    song_list = []
    for sample in result:
        song_list.append({'nr':sample[0], 'title':sample[1]})

    return song_list


app = Flask(__name__)
api = Api(app)

@app.route('/api/five_latest', methods=['GET'])
def api_latest():
    top_five = get_five()
    return make_response(jsonify(top_five),200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)













