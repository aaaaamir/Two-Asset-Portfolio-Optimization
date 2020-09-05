# Includes functions and attributes used to collect and model the data from the csv files. Also runs functions to create
# and save the bootstrap dataset we use to test on.
import matplotlib.pyplot as plt
import csv
import pandas as pd
from Global_Constants import NUMBER_OF_YEARLY_TRADING_DAYS, BOOTSTRAP_TESTING_PUT_INDEX_CSV_FILE_PATH, \
    BOOTSTRAP_TRAINING_PUT_INDEX_CSV_FILE_PATH, BOOTSTRAP_TRAINING_TREND_INDEX_CSV_FILE_PATH, \
    BOOTSTRAP_TESTING_TREND_INDEX_CSV_FILE_PATH, TREND_INDEX_CSV_FILE_PATH, PUT_INDEX_CSV_FILE_PATH, \
    BOOTSTRAP_SAMPLE_SIZE, NUMBER_OF_BOOTSTRAP_SAMPLES, NUMBER_OF_DAYS_IN_HISTORICAL_DATA
import numpy as np

NUMBER_OF_PUT_RETURN_ENTRIES = 4361
MISSING_RETURN_INDICATOR = float(-1000.0)
FIRST_CONSECUTIVE_MISSING_INDEX_LEFT = 1303
FIRST_CONSECUTIVE_MISSING_INDEX_RIGHT = 1304
SECOND_CONSECUTIVE_MISSING_INDEX_LEFT = 2823
SECOND_CONSECUTIVE_MISSING_INDEX_RIGHT = 2824
ANNUALIZED_VOLATILITY_CONSTANT = np.sqrt(252)
NUMBER_OF_YEARS_IN_HORIZON = 18

PUT_ESTIMATED_EXPECTED_BLOCK_SIZE = 7.08103397
TREND_ESTIMATED_EXPECTED_BLOCK_SIZE = 4.60082382
COMBINED_EXPECTED_BLOCK_SIZE = (TREND_ESTIMATED_EXPECTED_BLOCK_SIZE + PUT_ESTIMATED_EXPECTED_BLOCK_SIZE) / 2


# Plots both indices
def plot_indices_histograms(put_index_to_graph, trend_index_to_graph):

    plt.hist(trend_index_to_graph, bins=100, color='purple', range=(-0.1, 0.1))
    plt.xlabel('Daily Return Value')
    plt.ylabel('Frequency Of Returns')
    plt.title('Histogram Of Daily Returns For Trend-Following Index')
    plt.show()

    plt.hist(put_index_to_graph, bins=150, color='purple', range=(-0.1, 0.1))
    plt.xlabel('Daily Return Value')
    plt.ylabel('Frequency Of Returns')
    plt.title('Histogram Of Daily Returns For Put Index')
    plt.show()


# Plots the returns for a trading strategy into a histogram
def plot_returns_histogram(returns_vector, trading_strategy_string, x_axis_range):

    plt.hist(returns_vector, range=x_axis_range, bins=100, color='red')
    plt.xlabel("Return Value")
    plt.ylabel('Frequency Of Returns During Testing Period')
    plt.title('Histogram Of Daily Returns For ' + trading_strategy_string)
    plt.show()


# Opens a csv file to store the dates in one array, and the daily returns in the other.
def save_csv_information(csv_string, array_size):

    returns_index = np.zeros(array_size)
    dates = ["exampleStr"] * array_size

    with open(csv_string, 'r') as csv_file:

        input_data = csv.reader(csv_file, delimiter=',')
        counter = 0

        for row in input_data:
            daily_return_value = float(row[1])
            returns_index[counter] = daily_return_value
            date_value = str(row[0])
            dates[counter] = date_value
            counter = counter + 1

    # Correcting the '\ufeff' prefix added to the first string in dates list.
    if dates[0] == '\ufeff2002-01-01':
        dates[0] = '2002-01-01'
    if dates[0] == '\ufeff2002-01-02':
        dates[0] = '2002-01-02'

    return returns_index, dates


# Opens a csv file to store the daily returns -- used for generated datasets
def save_csv_information_no_dates(csv_string, array_size):

    returns_index = np.zeros(array_size)

    with open(csv_string, 'r') as csv_file:

        input_data = csv.reader(csv_file, delimiter=',')
        counter = 0

        for row in input_data:
            daily_return_value = float(row[0])
            returns_index[counter] = daily_return_value
            counter = counter + 1

    return returns_index


# takes put and trend indices and extends the put one to accommodate missing values
def extend_put_index(put_dates_list, trend_dates_list, put_returns_array):

    put_returns_with_missing_data = np.zeros(NUMBER_OF_DAYS_IN_HISTORICAL_DATA)
    trend_counter = 0
    put_counter = 0

    # We scan through both lists of dates, anytime a date is missing from the put returns array, we insert a missing
    # - value denoted by MISSING_RETURN_INDICATOR = -1000.

    while put_counter < NUMBER_OF_PUT_RETURN_ENTRIES:

        if put_dates_list[put_counter] == trend_dates_list[trend_counter]:
            put_returns_with_missing_data[trend_counter] = put_returns_array[put_counter]
            put_counter = put_counter + 1
        else:
            put_returns_with_missing_data[trend_counter] = MISSING_RETURN_INDICATOR
        trend_counter = trend_counter + 1

    return put_returns_with_missing_data


# imputes novel two sequences of data where consecutive days of returns are missing from the put index.
def impute_consecutive_special_missing_put_returns(put_returns_array, standard_deviation):

    normal_dist_std_dev = standard_deviation * np.sqrt(float(1.5) / NUMBER_OF_YEARLY_TRADING_DAYS)

    normal_dist_mean = (2 * put_returns_array[FIRST_CONSECUTIVE_MISSING_INDEX_LEFT - 1] +
                        put_returns_array[FIRST_CONSECUTIVE_MISSING_INDEX_RIGHT + 1]) / 3
    put_returns_array[FIRST_CONSECUTIVE_MISSING_INDEX_LEFT] = np.random.normal(normal_dist_mean, normal_dist_std_dev)

    normal_dist_mean = (put_returns_array[FIRST_CONSECUTIVE_MISSING_INDEX_LEFT - 1] +
                        2 * put_returns_array[FIRST_CONSECUTIVE_MISSING_INDEX_RIGHT + 1]) / 3
    put_returns_array[FIRST_CONSECUTIVE_MISSING_INDEX_RIGHT] = np.random.normal(normal_dist_mean, normal_dist_std_dev)

    normal_dist_mean = (2 * put_returns_array[SECOND_CONSECUTIVE_MISSING_INDEX_LEFT - 1] +
                        put_returns_array[SECOND_CONSECUTIVE_MISSING_INDEX_RIGHT + 1]) / 3
    put_returns_array[SECOND_CONSECUTIVE_MISSING_INDEX_LEFT] = np.random.normal(normal_dist_mean, normal_dist_std_dev)

    normal_dist_mean = (put_returns_array[SECOND_CONSECUTIVE_MISSING_INDEX_LEFT - 1] +
                        2 * put_returns_array[SECOND_CONSECUTIVE_MISSING_INDEX_RIGHT + 1]) / 3
    put_returns_array[SECOND_CONSECUTIVE_MISSING_INDEX_RIGHT] = np.random.normal(normal_dist_mean, normal_dist_std_dev)

    return put_returns_array


# imputes missing returns data using Brownian bridge method.
def impute_missing_returns(put_returns_array, standard_deviation):

    counter = 0
    normal_dist_std_dev = standard_deviation * np.sqrt(1 / NUMBER_OF_YEARLY_TRADING_DAYS)

    while counter < NUMBER_OF_DAYS_IN_HISTORICAL_DATA:
        return_value = put_returns_array[counter]
        if return_value == MISSING_RETURN_INDICATOR:
            normal_dist_mean = (put_returns_array[counter + 1] + put_returns_array[counter - 1]) / 2
            put_returns_array[counter] = np.random.normal(normal_dist_mean, normal_dist_std_dev)
        counter = counter + 1

    return put_returns_array


# function to get annualized std deviations and means for the final daily returns over 18 years of data
def get_annualized_means_deviations(index_to_analyze):

    counter = 0
    starter_index = 0
    stop_index = NUMBER_OF_YEARLY_TRADING_DAYS
    means_array = np.zeros(NUMBER_OF_YEARS_IN_HORIZON)
    std_devs_array = np.zeros(NUMBER_OF_YEARS_IN_HORIZON)

    while counter < NUMBER_OF_YEARS_IN_HORIZON:

        if counter < NUMBER_OF_YEARS_IN_HORIZON - 1:
            returns_subarray = index_to_analyze[starter_index:stop_index]
            starter_index = starter_index + NUMBER_OF_YEARLY_TRADING_DAYS
            stop_index = stop_index + NUMBER_OF_YEARLY_TRADING_DAYS
        else:
            returns_subarray = index_to_analyze[starter_index:NUMBER_OF_DAYS_IN_HISTORICAL_DATA]

        # Calculates the annualized return by multiplying using all daily returns
        annualized_return = 1
        for return_value in returns_subarray:
            annualized_return = annualized_return * (1 + return_value)
        annualized_return = annualized_return - 1

        subarray_std_dev = np.std(returns_subarray) * ANNUALIZED_VOLATILITY_CONSTANT

        means_array[counter] = annualized_return
        std_devs_array[counter] = subarray_std_dev

        counter = counter + 1

    return means_array, std_devs_array


# Prints the annualized standard deviations and means in percentage format
def print_annualized_means_vols_percentage_format(put_index, trend_index):

    trend_index_annualized_means, trend_index_annualized_standard_deviations \
        = get_annualized_means_deviations(trend_index)
    trend_index_annualized_means = trend_index_annualized_means * 100
    trend_index_annualized_standard_deviations = trend_index_annualized_standard_deviations * 100

    print("trend index returns: ")
    print(trend_index_annualized_means)
    print()
    print("trend index annualized vol: ")
    print(trend_index_annualized_standard_deviations)
    print()

    put_index_annualized_means, put_index_annualized_standard_deviations = get_annualized_means_deviations(put_index)
    put_index_annualized_means = put_index_annualized_means * 100
    put_index_annualized_standard_deviations = put_index_annualized_standard_deviations * 100

    print("put index returns: ")
    print(put_index_annualized_means)
    print()
    print("put index annualized vol: ")
    print(put_index_annualized_standard_deviations)
    print()


# Draws a block size from a geometric probability distribution to be used in the bootstrap resampling algorithm
def draw_block_size(geometric_dist_parameter):
    return np.random.geometric(geometric_dist_parameter)


# Draws a starting position from a uniform probability distribution to be used in the bootstrap resampling algorithm
def draw_starting_spot():
    return np.random.randint(0, NUMBER_OF_DAYS_IN_HISTORICAL_DATA)


# Uses stationary block bootstrap method to take one single sample from historical data
def generate_single_bootstrap_sample_both_indices(put_index_to_sample, trend_index_to_sample, bootstrap_sample_size,
                                                  expected_block_size):

    put_bootstrap_sample = np.zeros(bootstrap_sample_size)
    trend_bootstrap_sample = np.zeros(bootstrap_sample_size)

    sampled_returns_counter = 0
    geometric_dist_parameter = float(1 / expected_block_size)

    # Loop continues drawing block sizes and sub-samples to add to the bootstrap until it is completely full
    while sampled_returns_counter < bootstrap_sample_size:

        sampled_block_size = draw_block_size(geometric_dist_parameter)
        historical_data_selection_spot = draw_starting_spot()

        # Loops through historical data to save returns to add to the bootstrap
        for i in range(sampled_block_size):

            put_bootstrap_sample[sampled_returns_counter] = put_index_to_sample[historical_data_selection_spot]
            trend_bootstrap_sample[sampled_returns_counter] = trend_index_to_sample[historical_data_selection_spot]
            sampled_returns_counter = sampled_returns_counter + 1
            historical_data_selection_spot = (historical_data_selection_spot + 1) % NUMBER_OF_DAYS_IN_HISTORICAL_DATA

            # Returns bootstrap sample if we've sampled enough from the historical dataset.. can be moved to outer scope
            if sampled_returns_counter >= bootstrap_sample_size:
                return put_bootstrap_sample, trend_bootstrap_sample

    # Code will never get to this point - added return statement to end to avoid infinite loop in case of error
    return put_bootstrap_sample, trend_bootstrap_sample


# Uses stationary block bootstrap method to take sample from historical data (assuming puts have already been imputed)
# All of the bootstraps are saved in one long array. Every 'bootstrap_sample_size' long string represents one sample.
def generate_all_bootstrap_samples(put_index_to_sample, trend_index_to_sample, expected_block_size,
                                   bootstrap_sample_size, number_of_bootstrap_samples):

    total_number_of_returns_to_generate = bootstrap_sample_size * number_of_bootstrap_samples
    put_bootstrap_samples = np.zeros(total_number_of_returns_to_generate)
    trend_bootstrap_samples = np.zeros(total_number_of_returns_to_generate)

    number_of_bootstrap_samples_taken = 0
    sampled_returns_counter = 0

    # iteratively creates bootstrap samples of size 'bootstrap_sample_size' and copies the resulting
    while number_of_bootstrap_samples_taken < number_of_bootstrap_samples:

        put_subsample, trend_subsample \
            = generate_single_bootstrap_sample_both_indices(put_index_to_sample, trend_index_to_sample,
                                                            bootstrap_sample_size, expected_block_size)

        number_of_bootstrap_samples_taken = number_of_bootstrap_samples_taken + 1

        subsample_returns_counter = 0

        for i in range(bootstrap_sample_size):

            put_bootstrap_samples[sampled_returns_counter] = put_subsample[subsample_returns_counter]
            trend_bootstrap_samples[sampled_returns_counter] = trend_subsample[subsample_returns_counter]

            subsample_returns_counter = subsample_returns_counter + 1
            sampled_returns_counter = sampled_returns_counter + 1

    return put_bootstrap_samples, trend_bootstrap_samples


# Creates and saves bootstrap sets (same sizes), to use to train Sharpe ratio strategy, and test strategies
# importing historical data and imputing missing returns
historical_trend_index, trend_dates = save_csv_information(TREND_INDEX_CSV_FILE_PATH, NUMBER_OF_DAYS_IN_HISTORICAL_DATA)
initial_put_index, put_dates = save_csv_information(PUT_INDEX_CSV_FILE_PATH, NUMBER_OF_PUT_RETURN_ENTRIES)
historical_put_index_standard_deviation = np.std(initial_put_index)
imputed_historical_put_index = extend_put_index(put_dates, trend_dates, initial_put_index)
imputed_historical_put_index = impute_consecutive_special_missing_put_returns(imputed_historical_put_index,
                                                                              historical_put_index_standard_deviation)
imputed_historical_put_index = \
   impute_missing_returns(imputed_historical_put_index, historical_put_index_standard_deviation)

# plots historical data histograms and prints the annualized means and vols 
plot_indices_histograms(imputed_historical_put_index, historical_trend_index)
print_annualized_means_vols_percentage_format(imputed_historical_put_index, historical_trend_index)

# Creates and saves bootstrap samples
bootstrap_put_index_test_set, bootstrap_trend_index_test_set \
    = generate_all_bootstrap_samples(imputed_historical_put_index, historical_trend_index, COMBINED_EXPECTED_BLOCK_SIZE,
                                     BOOTSTRAP_SAMPLE_SIZE, NUMBER_OF_BOOTSTRAP_SAMPLES)

pd.DataFrame(bootstrap_put_index_test_set).to_csv(BOOTSTRAP_TESTING_PUT_INDEX_CSV_FILE_PATH, header=None, index=None)
pd.DataFrame(bootstrap_trend_index_test_set)\
   .to_csv(BOOTSTRAP_TESTING_TREND_INDEX_CSV_FILE_PATH, header=None, index=None)
pd.DataFrame(imputed_historical_put_index)\
   .to_csv(BOOTSTRAP_TRAINING_PUT_INDEX_CSV_FILE_PATH, header=None, index=None)
pd.DataFrame(historical_trend_index)\
   .to_csv(BOOTSTRAP_TRAINING_TREND_INDEX_CSV_FILE_PATH, header=None, index=None)
