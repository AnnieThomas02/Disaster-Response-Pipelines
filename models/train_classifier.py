import sys
# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

# import libraries
import sqlite3
import pandas as pd
import pickle
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql('DisasterResponse',engine)
    X = df ['message']
    Y = df.iloc[:,4:]
    return X,Y


def tokenize(text):
    url_regex='http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''Builds a parameter optimized model
    
    Returns:
        model (GridSearchCV object)
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'clf__estimator__n_estimators': [10],
    'clf__estimator__min_samples_split': [2]   
    #'clf__estimator__n_estimators': [10, 50],
    #'clf__estimator__min_samples_leaf': [2, 5,10]
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1, verbose=3)
    return model

def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print('{}:'.format( Y_test.columns[i]))
        print(classification_report(Y_test[col], Y_pred[:, i]))
    #class_report = classification_report(Y_test, Y_pred, target_names=Y_test.columns)
    #print(class_report)


def save_model(model, model_filepath):
    
     '''
    INPUT
    model - the name in which the file would be saved as
    model_filepath - the filepath in which the file would be saved
    
    OUTPUT
    File saved with desired name and filepath
    '''
     with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        # X, Y, category_names = load_data(database_filepath)
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()