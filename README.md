# Disaster Response Pipeline Project

## Purpose

The purpose of this project is to train a machine learning model to classify messages of individuals during disasters.
To get to people who are in dire need of help first, help teams/organisations need to know where they are needed.
Text messages shared on social media platforms etc. are a good source to identify peoples needs.

## Project Structure

### Data Processing

In `data/process_data.py` real messages and their classifications are processed for the machine learning pipeline and
stored in a SQLite database.

### Model Training

In `modes/train_classifier.py` the machine learning model is trained through a pipeline. The steps include:

1. Tokenization
2. Tfidf Transformation
3. Classification using the AdaBoostClassifier

The model fitting is done via the GridSearch method. And the parametrisation is stored in `classifier.pkl`.

### Visualisation

The message classification can be tested by running the `run.py` file in the `/app` directory.
Additionally, plotly visualisations are included.

## How to use the Model

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Open the URL displayed in your terminal

## Acknowledgement

The idea and instructions for this project are part of the Udacity [Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
Furthermore, the data is provided by [appen](https://appen.com/).
