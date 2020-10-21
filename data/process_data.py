import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
      Function:
      load data from two csv file and then merge them
      Args:
      messages_filepath (str): the file path of messages csv file
      categories_filepath (str): the file path of categories csv file
      Return:
      df (DataFrame): Dataframe obtained from merging the two input data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on=['id'])
    return df


def clean_data(df):
    """
    Clean data included in the DataFrame and transform categories part
    
    Args:
    df : Merged dataframe returned from load_data() function
    Returns:
    df pandas_dataframe: Cleaned data to be used by ML model
    """
    #categories = df.categories.str.split(';', expand=True)
    categories = df.categories.str.split(';', expand=True)
    row = categories.head(1)
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:]
    category_colnames = category_colnames.tolist()
    #rename the columns of `categories`
    categories .columns = category_colnames    
    
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] =  categories[column].astype(int)
    
    # drop the original categories column from `df`
    
    df.drop(['categories'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1, join = 'inner' )  

 # drop duplicates
    df.drop_duplicates(inplace = True)
    return df


def save_data(df, database_filename):
    """
    Saves cleaned data to an SQL database
    Args:
    df pandas_dataframe: Cleaned data returned from clean_data() function
    database_file_name : File path of SQL Database into which the cleaned\
    data is to be saved
    Returns:
    None
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponse', engine, index=False,if_exists = 'replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()