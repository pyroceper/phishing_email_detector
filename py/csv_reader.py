import pandas 

# returns a dataframe
def open_csv(file_name): # TODO: try catch
    df = pandas.read_csv(file_name)
    # remove rows with empty/missing email text and include them in the dataframe
    df['Email Text'].fillna('Missing', inplace=True) 
    return df