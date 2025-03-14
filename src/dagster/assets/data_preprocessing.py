from ucimlrepo import fetch_ucirepo

from dagster import op, Out

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

@op(
    out={
        "features": Out(pd.DataFrame),
        "targets": Out(pd.Series)
    },
    description=("Load the Online Shoppers Purchasing Intention dataset from "
                 "the UCI repo.")
)
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the Online Shoppers Purchasing Intention dataset from UCI repo.
    
    Returns:
        X: pandas DataFrame, features
        y: pandas Series, targets
    """
    # fetch dataset 
    online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468) 
  
    # data (as pandas dataframes) 
    X = online_shoppers_purchasing_intention_dataset.data.features 
    y = online_shoppers_purchasing_intention_dataset.data.targets['Revenue']

    return X, y


@op(
    out=Out(dict),
    description=("Preprocess the data by applying OneHotEncoding to "
                 "categorical columns, splitting the data to train and test "
                 "sets, and resampling the train set using SMOTE.")
)
def preprocess_data(features: pd.DataFrame, targets: pd.Series) -> dict:
    """
    Preprocess the data by:
        - applying OneHotEncoding to categorical columns,
        - splitting the data to train and test sets, and 
        - resampling the train set using SMOTE.

    Args:
        data(features: pandas DataFrame, targets: pandas Series)
        
    Returns:
        X_train_resampled: pandas DataFrame
        y_train_resampled: pandas Series
        X_test: pandas DataFrame
        y_test: pandas Series
    """
    X, y = features, targets

    # Identify categorical columns
    categorical_cols = ['Administrative', 'Informational', 'ProductRelated',
                        'SpecialDay', 'Month', 'VisitorType', 'Weekend',
                        'OperatingSystems', 'Browser', 'Region', 'TrafficType']

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

    return {
        "X_train": X_train_resampled, 
        "X_test": X_test, 
        "y_train": y_train_resampled, 
        "y_test": y_test
    }

if __name__ == '__main__':
    features, target = load_data()
    
    print(features.columns)
    print(features.shape)
    print(target.shape)