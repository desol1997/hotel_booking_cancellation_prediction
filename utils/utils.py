from typing import List, Any, Tuple, Union, Mapping, Callable
from collections.abc import Iterable
import pickle

import numpy.typing as npt
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import xgboost as xgb
from xgboost import DMatrix, Booster


def metrics_factory(metric: str) -> Callable[..., float]:
    factory = {
        'AUC': roc_auc_score,
        'Accuracy': accuracy_score,
        'Precision': precision_score,
        'Recall': recall_score,
        'F1': f1_score
    }

    if metric in factory:
        return factory[metric]
    else:
        raise ValueError(f"Metric '{metric}' is not supported. Supported metrics: {', '.join(factory.keys())}")


def trainers_factory(trainer: str) -> Callable[..., Union[BaseEstimator, Booster]]:
    factory = {
        'logistic_regression': train_logistic_regression,
        'decision_tree': train_decision_tree,
        'random_forest': train_random_forest,
        'xgboost': train_xgboost
    }

    if trainer in factory:
        return factory[trainer]
    else:
        raise ValueError(f"Trainer '{trainer}' is not supported. Supported trainers: {', '.join(factory.keys())}")
    

def extract_features_and_target(
        data: pd.DataFrame,
        target: str,
        features: List[str] | None = None
    ) -> Tuple[pd.DataFrame, npt.NDArray]:
        
        y = data[target].values

        if features is not None:
            X = data[features]
        else:
            X = data.drop(columns=[target])

        return X, y


def standardize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    for c in [*df.dtypes[df.dtypes == 'object'].index]:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    return df

    
def reset_index(*dfs: pd.DataFrame) -> pd.DataFrame | List[pd.DataFrame]:
    if len(dfs) == 1:
        return dfs[0].reset_index(drop=True)
    else:
        return [df.reset_index(drop=True) for df in dfs]


def convert_target_to_binary(df: pd.DataFrame, target: str, target_value: str) -> pd.DataFrame:
    df[target] = (df[target] == target_value).astype(int)
    return df


def fill_missing_with_values(
        df: pd.DataFrame,
        cols_with_missing: Iterable[str],
        numerical_columns: Iterable[str],
        categorical_columns: Iterable[str],
        fill_numerical_value: float,
        fill_categorical_value: str
    ) -> pd.DataFrame:

    for col in cols_with_missing:
        if col in numerical_columns:
            df[col] = df[col].fillna(fill_numerical_value)
        elif col in categorical_columns:
            df[col] = df[col].fillna(fill_categorical_value)

    return df


def _train_clissifier(classifier: BaseEstimator, X: npt.ArrayLike, Y: npt.ArrayLike, **kwargs) -> BaseEstimator:
    model = classifier(**kwargs)
    model.fit(X, Y)

    return model


def fit_dict_vectorizer(data, sparse=False):
    dv = DictVectorizer(sparse=sparse)
    dv.fit(data)

    return dv


def create_feature_matrix(data: Mapping[str, Any], dv: DictVectorizer) -> npt.NDArray:
    feature_names = dv.get_feature_names_out()

    if len(feature_names) == 0:
        raise ValueError('The provided DictVectorizer does not have any feature names.')
    
    X = dv.transform(data)

    return X


def train_logistic_regression(X: npt.ArrayLike, Y: npt.ArrayLike, **kwargs) -> LogisticRegression:
    return _train_clissifier(LogisticRegression, X=X, Y=Y, **kwargs)


def train_decision_tree(X: npt.ArrayLike, Y: npt.ArrayLike, **kwargs) -> DecisionTreeClassifier:
    return _train_clissifier(DecisionTreeClassifier, X=X, Y=Y, **kwargs)


def train_random_forest(X: npt.ArrayLike, Y: npt.ArrayLike, **kwargs) -> RandomForestClassifier:
    return _train_clissifier(RandomForestClassifier, X=X, Y=Y, **kwargs)


def predict(X: npt.ArrayLike, model: Any) -> npt.NDArray:

    if isinstance(model, BaseEstimator):
        return model.predict_proba(X)[:, 1] # For scikit-learn classifiers
    
    if isinstance(model, Booster):
        return model.predict(DMatrix(X)) # For XGBoost models


def train_xgboost(params, X, y, num_boost_round, **kwargs):
    model = xgb.train(params=params, dtrain=xgb.DMatrix(X, label=y), num_boost_round=num_boost_round, **kwargs)
    return model


def parse_xgboost_output(output):
    results = []

    for line in output.stdout.strip().split('\n'):
        iteration_line, train_line, validation_line = line.split('\t')

        iteration = int(iteration_line.strip('[]'))
        train = float(train_line.split(':')[1])
        validation = float(validation_line.split(':')[1])

        results.append((iteration, train, validation))

    columns = ['num_iter', 'train_auc', 'val_auc']
    df_results = pd.DataFrame(results, columns=columns)

    return df_results


def save_model_asset(asset: Any, filename: str) -> None:
    with open(filename, 'wb') as f:
        pickle.dump(asset, f)


def read_model_asset(filename: str) -> Any:
    with open(filename, 'rb') as f:
        return pickle.load(f)
