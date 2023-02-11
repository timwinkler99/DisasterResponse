import sys
import pandas as pd
import sqlite3


def load_data(messages_filepath, categories_filepath):
    """
    :param messages_filepath:
    :param categories_filepath:
    :return: merged dataframe
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """
    :param df: uncleaned df containing messages and categories
    :return: cleaned df
    """

    # split categories column
    categories = df['categories'].str.split(';', expand=True)

    # get first row and remove unwanted characters
    row = categories.iloc[0]
    category_colnames = list(row.apply(lambda x: x[:-2]))

    # rename categories columns
    categories.columns = category_colnames

    # remove category from 0,1 indicator
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # replace unformatted categories column against formatted categories df
    df = df.drop('categories', axis=1)

    df = pd.concat([df, categories], axis=1)

    # remove duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    :param df:
    :param database_filename:
    :return: saves dataframe under specified database filename in sqlite database
    """
    # used sqlite3 instead of SQLAlchemy, because database was not readable in train_classifier.py and run.py
    conn = sqlite3.connect(database_filename)
    df.to_sql('DisasterResponse', con=conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
