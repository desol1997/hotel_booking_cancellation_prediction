from pathlib import Path
from typing import List, Dict, Any, Tuple, Mapping
from collections.abc import Iterable
from collections import defaultdict, namedtuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold

from logger import logger

from utils import (
    standardize_dataset,
    fill_missing_with_values,
    extract_features_and_target,
    reset_index,
    fit_dict_vectorizer,
    create_feature_matrix,
    metrics_factory,
    trainers_factory,
    predict,
    save_model_asset
)


def extract_data(filepath: Path | str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def clean_and_standardize_data(
        data: pd.DataFrame,
        numerical_columns: Iterable[str],
        categorical_columns: Iterable[str],
        columns_to_drop: List[str] | None = None,
        fill_numerical_value: float = 0,
        fill_categorical_value: str = 'undefined'
    ) -> pd.DataFrame:

    if columns_to_drop is not None:
        logger.info(f"Dropping unnecessary columns: {', '.join(columns_to_drop)}...")
        data = data.drop(axis=1, columns=columns_to_drop)

    logger.info('Standardizing data...')
    data = standardize_dataset(data)
    
    logger.info('Checking for missing values...')
    if data.isnull().any().any():
        cols_with_missing = [*data.columns[data.isnull().any()]]
        logger.warning(f"Missing values detected in the following columns: {', '.join(cols_with_missing)}.")
        logger.info('Filling missing values...')
        data = fill_missing_with_values(data, cols_with_missing, numerical_columns, categorical_columns,
                                        fill_numerical_value, fill_categorical_value)
    else:
        logger.info('No missing values found.')

    return data


def prepare_data_for_model(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target: str,
        features: List[str] | None = None,
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, DictVectorizer]:

    logger.info('Extracting features and target from train and test datasets...')
    train_features, train_target = extract_features_and_target(df_train, target=target, features=features)
    test_features, test_target = extract_features_and_target(df_test, target=target, features=features)

    logger.info('Resetting index...')
    train_features, test_features = reset_index(train_features, test_features)

    logger.info('Transforming train and test features into dictionary...')
    train_features_dict = train_features.to_dict(orient='records')
    test_features_dict = test_features.to_dict(orient='records')

    logger.info('Fitting DictVectorizer for building feature matrix...')
    dv = fit_dict_vectorizer(train_features_dict, sparse=False)

    logger.info('Transforming train and test data into feature matrices...')
    X_train = create_feature_matrix(train_features_dict, dv)
    X_test = create_feature_matrix(test_features_dict, dv)

    return X_train, X_test, train_target, test_target, dv


def evaluate_model(
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        metrics: Iterable[str] = ('AUC', 'Accuracy', 'Precision', 'Recall', 'F1'),
        threshold: float = 0.5
    ) -> Dict[str, float]:

    scores = {}

    for metric in metrics:
        logger.info(f'Calculating {metric} score...')
        metric_func = metrics_factory(metric)
        if metric == 'AUC':
            scores[metric] = metric_func(y_true, y_pred)
        else:
            scores[metric] = metric_func(y_true, y_pred > threshold)

    return scores


def train_model_and_evaluate(
        data: pd.DataFrame,
        features: List[str],
        target: str,
        trainer: str,
        model_params: Dict[str, Any],
        kfold_splits: int = 5,
        seed: int | None = None,
        num_boost_round: int = 100 # only for XGBoost
    ):

    logger.info('Initializing KFold for cross-validation...')
    kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=seed)
    metrics = defaultdict(list)
    MeanStdMetric = namedtuple('MeanStdMetric', 'mean std')

    for fold, (train_idx, test_idx) in enumerate(kfold.split(data), 1):
        logger.info(f'Processing fold {fold} out of {kfold_splits}...')

        df_train = data.iloc[train_idx]
        df_test = data.iloc[test_idx]
        
        logger.info('Preparing data for the model...')
        X_train, X_test, y_train, y_test, dv = prepare_data_for_model(df_train, df_test, features=features, target=target)
        
        model_func = trainers_factory(trainer)
        
        logger.info(f'Training {trainer} model for fold {fold}....')
        if trainer == 'xgboost':
            model = model_func(model_params, X_train, y_train, num_boost_round=num_boost_round)
        else:
            model = model_func(X_train, y_train, **model_params)

        logger.info(f'Predicting for fold {fold}...')
        y_pred = predict(X_test, model)

        logger.info(f'Calculating metrics for fold {fold}...')
        scores = evaluate_model(y_test, y_pred)

        for metric, value in scores.items():
            metrics[metric].append(value)

    logger.info('Performing metric aggregation: Calculating mean and standard deviation for all evaluation metrics...')
    avg_metrics = {metric: MeanStdMetric(np.mean(values), np.std(values)) for metric, values in metrics.items()}

    return model, dv, avg_metrics


def save_model_assets(model: Any, dv: DictVectorizer, model_filename: str, dv_filename: str) -> None:
    logger.info(f'Saving model into {model_filename}...')
    save_model_asset(model, model_filename)
    logger.info(f'Saving DictVectorizer into {dv_filename}...')
    save_model_asset(dv, dv_filename)


def save_model_metrics(model_metrics: Mapping, filename: str) -> None:
    with open(filename, 'w') as f:
        for metric, value in model_metrics.items():
            f.write(f'{metric} of the model: {value.mean}.\n')
    