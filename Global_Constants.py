# Contains constant values that are used in more than one script
import os

NUMBER_OF_YEARLY_TRADING_DAYS = 252
BOOTSTRAP_SAMPLE_SIZE = 30
NUMBER_OF_BOOTSTRAP_SAMPLES = 1000
TOTAL_NUMBER_OF_BOOTSTRAP_RETURNS = BOOTSTRAP_SAMPLE_SIZE * NUMBER_OF_BOOTSTRAP_SAMPLES
NUMBER_OF_DAYS_IN_HISTORICAL_DATA = 4520

SCRIPT_PATH = os.path.dirname(__file__)
TREND_INDEX_CSV_FILE_PATH = os.path.join(SCRIPT_PATH, 'Trend And Put Index Datasets/masters trend index dataset.csv')
PUT_INDEX_CSV_FILE_PATH = os.path.join(SCRIPT_PATH, 'Trend And Put Index Datasets/masters put index dataset.csv')
BOOTSTRAP_TESTING_TREND_INDEX_CSV_FILE_PATH = \
    os.path.join(SCRIPT_PATH, 'Trend And Put Index Datasets/bootstrap_trend_index_test_set.csv')
BOOTSTRAP_TESTING_PUT_INDEX_CSV_FILE_PATH = \
    os.path.join(SCRIPT_PATH, 'Trend And Put Index Datasets/bootstrap_put_index_test_set.csv')
BOOTSTRAP_TRAINING_TREND_INDEX_CSV_FILE_PATH = \
    os.path.join(SCRIPT_PATH, 'Trend And Put Index Datasets/bootstrap_trend_index_training_set.csv')
BOOTSTRAP_TRAINING_PUT_INDEX_CSV_FILE_PATH = \
    os.path.join(SCRIPT_PATH, 'Trend And Put Index Datasets/bootstrap_put_index_training_set.csv')

