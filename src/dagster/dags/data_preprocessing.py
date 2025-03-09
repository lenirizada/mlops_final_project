from ucimlrepo import fetch_ucirepo

from dagster import asset

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

@asset
def load_data():
    """
    Load and preprocess the Online Shoppers Purchasing Intention dataset from UCI repo.
    
    This applies OneHotEncoding to categorical columns and then SMOTE to the training data.
    
    Returns:
        X_train_resampled: pandas DataFrame
        y_train_resampled: pandas Series
        X_test: pandas DataFrame
        y_test: pandas Series
    """
    # fetch dataset 
    online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468) 
  
    # data (as pandas dataframes) 
    X = online_shoppers_purchasing_intention_dataset.data.features 
    y = online_shoppers_purchasing_intention_dataset.data.targets 

    # Identify categorical columns
    categorical_cols = ['Administrative', 'Informational', 'ProductRelated',
                        'SpecialDay', 'Month', 'VisitorType', 'Weekend',
                        'OperatingSystem', 'Browser', 'Region', 'TrafficType']

    # Apply OneHotEncoder to categorical columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_categorical_data = encoder.fit_transform(X[categorical_cols])

    # Convert encoded data to DataFrame
    encoded_categorical_df = pd.DataFrame(
        encoded_categorical_data,
        columns=encoder.get_feature_names_out(categorical_cols))

    # Drop original categorical columns and concatenate encoded columns
    X = X.drop(columns=categorical_cols).reset_index(drop=True)
    X = pd.concat([X, encoded_categorical_df], axis=1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2)
    
    # Apply SMOTE to training data
    smote = SMOTE(random_state=2)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test