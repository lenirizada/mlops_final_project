from dagster import op

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

## TODO: Add parameter_space as input to train functions

@op
def train_gb_model_grid_search(X_train, y_train):
    """
    Train a GradientBoostingClassifier model on the training data with GridSearchCV.

    Args:
        X_train: pandas DataFrame
        y_train: pandas Series

    Returns:
        GridSearchCV
    """
    gb = GradientBoostingClassifier(random_state=2)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 3, 4]
    }
    gb_grid = GridSearchCV(gb, param_grid, cv=3)
    gb_grid.fit(X_train, y_train)
    return gb_grid

@op
def train_rf_model_grid_search(X_train, y_train):
    """
    Train a RandomForestClassifier model on the training data with GridSearchCV.

    Args:
        X_train: pandas DataFrame
        y_train: pandas Series

    Returns:
        GridSearchCV
    """
    rf = RandomForestClassifier(random_state=2)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 3, 4]
    }
    rf_grid = GridSearchCV(rf, param_grid, cv=3)
    rf_grid.fit(X_train, y_train)
    return rf_grid

@op
def train_svc_model_grid_search(X_train, y_train):
    """
    Train a SVC model on the training data with GridSearchCV.

    Args:
        X_train: pandas DataFrame
        y_train: pandas Series

    Returns:
        GridSearchCV
    """
    svc = SVC(random_state=2)
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly']
    }
    svc_grid = GridSearchCV(svc, param_grid, cv=3)
    svc_grid.fit(X_train, y_train)
    return svc_grid

@op
def train_lr_model_grid_search(X_train, y_train):
    """
    Train a LogisticRegression model on the training data with GridSearchCV.

    Args:
        X_train: pandas DataFrame
        y_train: pandas Series

    Returns:
        GridSearchCV
    """
    lr = LogisticRegression(random_state=2)
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }
    lr_grid = GridSearchCV(lr, param_grid, cv=3)
    lr_grid.fit(X_train, y_train)
    return lr_grid


model_list = {
    'GradientBoostingClassifier': train_gb_model_grid_search,
    'RandomForestClassifier': train_rf_model_grid_search,
    'SVC': train_svc_model_grid_search,
    'LogisticRegression': train_lr_model_grid_search
}