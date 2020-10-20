# Disaster-Response-Pipelines

## Overview
This project used processed data originally from Figure Eight's Disaster Response data to build a model for an API that classifies disaster messages. This project, a web app is created to read in a user message input and then assign up to 36 categories to this message input. In this classification process, I used random forest classifier combined with multipleoutput classifier to perform the task

## Install
This project requires Python 3.x and the following Python libraries installed:
1.	NumPy
2.	Pandas
3.	Matplotlib
4.	Json
5.	Plotly
6.	Nltk
7.	Flask
8.	Sklearn
9.	Sqlalchemy
10.	Sys
11.	Re
12.	Pickle

## ETL pipeline process(process_data.py)
•	This code extract Data from  files: messages.csv (containing message data) and categories.csv (classes of messages) that read into  panda Dataframe , merge together , cleaned the data and tokenized the data.
•	Dataframe was then loaded to a SQLite database to be loaded by the next step of the project

## ML pipeline process(train_classifier.py)
•	This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.

## Web Application (run.py)


