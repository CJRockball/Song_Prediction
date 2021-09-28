# Song Prediction
Predicts song genre from a set of properties. The model is analyzed and optimized and served through a web interface

## Dataset
Dataset is derived from:

T. Bertin-Mahieux, D. P.W. Ellis, B. Whitman, and P. Lamere. The million song dataset. In
Proceedings of the 12th International Conference on Music Information Retrieval (ISMIR),
2011.
A. Schindler and A. Rauber. Capturing the temporal domain in Echonest Features for improved
classification effectiveness. In Proceedings of the 10th InternationalWorkshop on Adaptive Multimedia Retrieval (AMR), 2012.


### Features

* trackID: unique identifier for each song (Maps features to their labels)
* title: title of the song. Type: text.
* tags: A comma-separated list of tags representing the words that appeared in the lyrics of the song and are assigned by human annotators. Type: text / categorical.
* loudness: overall loudness in dB. Type: float / continuous.
* tempo: estimated tempo in beats per minute (BPM). Type: float / continuous.
* time_signature: estimated number of beats per bar. Type: integer.
* key: key the track is in. Type: integer/ nominal. 
* mode: major or minor. Type: integer / binary.
* duration: duration of the song in seconds. Type: float / continuous.
* vect_1 ... vect_148: 148 columns containing pre-computed audio features of each song. 
	- These features were pre-extracted (NO TEMPORAL MEANING) from the 30 or 60 second snippets, and capture timbre, chroma, and mfcc aspects of the audio. \
	- Each feature takes a continuous value. Type: float / continuous.
 

### Labels

* trackID: unique id for each song (Maps features to their labels)
* genre: the genre label
	1. Soul and Reggae
	2. Pop
	3. Punk
	4. Jazz and Blues
	5. Dance and Electronica
	6. Folk
	7. Classic Pop and Rock
	8. Metal


## Filestructure

1. Data exploration and modeldevelopment is described in the [Data_Exploration_and_Model_development](https://github.com/CJRockball/Song_Prediction/blob/main/Data_Exploration_and_Model_Development.ipynb) notebook. This file will also predict all the samples in the test file and save to test_answers.
2. To do data preparation and model training run the data_model_setup.py script. It will save pipeline and model to model_artifacts and filtered csv file to data_artifacts.
3. To setup and preload a database run the set_up_db.py script. It will reset the database and preload 100 songs from the training data.
4. To test the prediction algo run the WebUI.py script. It will draw songs from the test csv file, predict genre, commit data to database and suggest other songs from the same genre

### Other Files and Folders

* data; contains raw data and description
* data_artifacts; contains processed data from ETL pipeline
* model_artifacts; contains pickeled models
* templates; contains html templates
* environment.yml; file specifying python requirements
* music_data.db; database used by web ui
* test_answers.csv; predicted genres for test data
