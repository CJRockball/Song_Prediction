# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 13:17:25 2021

@author: PatCa
"""

from flask import Flask, make_response, jsonify, request
from flask_restful import Resource, Api
import sqlite3


def get_five():
    
    #Get the five most recent songs
    conn = sqlite3.connect("music_data.db")
    cur = conn.cursor()
    #Inset new records
    cur.execute("""SELECT song_id,title \
                    FROM title_table \
                    ORDER BY song_id DESC
                    LIMIT(5);""",
                )
    song_samples = cur.fetchall()
    # Write change
    conn.commit()
    conn.close()
    #print(song_samples)
    song_list = []
    for sample in song_samples:
        song_list.append({'nr':sample[0], 'title':sample[1]})
    #print(song_list)
    return song_list


app = Flask(__name__)
api = Api(app)

@app.route('/api/five_latest', methods=['GET'])
def api_latest():
    top_five = get_five()
    return make_response(jsonify(top_five),200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)













