import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.logger import logger
from utils.utils import convert_target_to_binary

from train import (
    extract_data,
    clean_and_standardize_data,
    train_model_and_evaluate,
    save_model_assets,
    save_model_metrics
)

DATA_DIR = Path(__file__).parent.parent / 'data'
FILE_PATH = DATA_DIR / 'hotel_reservations.csv'
DROP_COLUMNS = ['Booking_ID', 'arrival_year']
SEED = 1
FEATURES = [
    'no_of_adults',
    'no_of_children',
    'no_of_weekend_nights',
    'no_of_week_nights',
    'type_of_meal_plan',
    'required_car_parking_space',
    'room_type_reserved',
    'lead_time',
    'arrival_month',
    'arrival_date',
    'market_segment_type',
    'repeated_guest',
    'no_of_previous_cancellations',
    'no_of_previous_bookings_not_canceled',
    'avg_price_per_room',
    'no_of_special_requests'
]
NUMERICAL_FEATURES = (
    'no_of_adults',
    'no_of_children',
    'no_of_weekend_nights',
    'no_of_week_nights',
    'lead_time',
    'no_of_previous_cancellations',
    'no_of_previous_bookings_not_canceled',
    'avg_price_per_room',
    'no_of_special_requests'
)
CATEGORICAL_FEATURES = (
    'type_of_meal_plan',
    'required_car_parking_space',
    'room_type_reserved',
    'arrival_month',
    'arrival_date',
    'market_segment_type',
    'repeated_guest'
)
TARGET = 'booking_status'
TARGET_VALUE = 'canceled'
TRAINER = 'xgboost'
NUM_BOOST_ROUND = 100
MODEL_PARAMS = {
    'eta': 0.3, 
    'max_depth': 10,
    'min_child_weight': 1,

    'objective': 'binary:logistic',

    'nthread': 8,
    'seed': SEED,
    'verbosity': 1
}
KFOLD_SPLITS = 5
MODEL_FILENAME = 'model.bin'
DV_FILENAME = 'DictVectorizer.bin'
METRICS_FILENAME = 'model_metrics.txt'


def main():
    logger.info('Data extraction started...')
    data = extract_data(FILE_PATH)
    logger.info('Data extraction completed.')

    logger.info('Cleaning and standardizing dataset...')
    data = clean_and_standardize_data(data, numerical_columns=NUMERICAL_FEATURES,
                                      categorical_columns=CATEGORICAL_FEATURES, columns_to_drop=DROP_COLUMNS)
    logger.info('Dataset cleaning and standardization completed.')

    logger.info(f'Convering {TARGET} to binary format...')
    data = convert_target_to_binary(data, target=TARGET, target_value=TARGET_VALUE)
    logger.info('Converting is completed.')

    logger.info('Model training and evaluation in progress...')
    model, dv, metrics = train_model_and_evaluate(data, features=FEATURES, target=TARGET, trainer=TRAINER,
                                                  model_params=MODEL_PARAMS, kfold_splits=KFOLD_SPLITS, seed=SEED,
                                                  num_boost_round=NUM_BOOST_ROUND)
    logger.info('Model training and evaluation completed.')

    logger.info('Saving model asssets...')
    save_model_assets(model=model, dv=dv, model_filename=MODEL_FILENAME, dv_filename=DV_FILENAME)
    logger.info('Model assets saved.')

    logger.info('Saving model metrics...')
    save_model_metrics(metrics, filename=METRICS_FILENAME)
    logger.info('Model metrics saved.')
    

if __name__ == '__main__':
    main()
