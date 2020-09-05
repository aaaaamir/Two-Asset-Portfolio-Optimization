# Contains functions that facilitate analysis of a set of returns from a given trading strategy
import numpy as np
from Static_Strategy_Functions import get_equal_proportion_strategy_returns, get_sharpe_ratio_strategy_returns, \
    get_fully_put_static_strategy_returns, get_fully_trend_static_strategy_returns, \
    generate_sharpe_ratio_maximization_weighting
from Data_Collector import save_csv_information_no_dates, plot_returns_histogram
from Global_Constants import BOOTSTRAP_TESTING_PUT_INDEX_CSV_FILE_PATH, BOOTSTRAP_TRAINING_PUT_INDEX_CSV_FILE_PATH, \
    BOOTSTRAP_TRAINING_TREND_INDEX_CSV_FILE_PATH, BOOTSTRAP_TESTING_TREND_INDEX_CSV_FILE_PATH, BOOTSTRAP_SAMPLE_SIZE, \
    NUMBER_OF_YEARLY_TRADING_DAYS, NUMBER_OF_BOOTSTRAP_SAMPLES, TOTAL_NUMBER_OF_BOOTSTRAP_RETURNS, \
    NUMBER_OF_DAYS_IN_HISTORICAL_DATA

BETA_FOR_95_PERCENT_CVAR = 0.95
SHARPE_TRAINED_PUT_PROPORTION = 0.63501958
SHARPE_TRAINED_TREND_PROPORTION = 0.36498042
STATIC_STRATEGY_HISTOGRAMS_X_AXIS_RANGE = (-0.05, 0.05)


# Returns the beta-level CVaR for a set of unsorted daily returns (we use 95% in analysis)
def calculate_cvar(returns_vector, beta):

    returns_vector = np.sort(returns_vector)
    var_index = np.floor((1 - beta) * returns_vector.size)
    var_index = int(var_index)

    cvar_sum = 0

    for i in range(var_index):
        cvar_sum = cvar_sum + returns_vector[i]

    cvar_float = float(cvar_sum / var_index)

    return cvar_float


# estimates annualized volatility of a trading strategy from a set of bootstrap samples
def estimate_annualized_vol_from_bootstrap(trading_strategy_returns, bootstrap_sample_size,
                                           number_of_bootstrap_samples):

    counter = 0
    bootstrap_sample_volatilities = np.zeros(number_of_bootstrap_samples)

    for i in range(number_of_bootstrap_samples):

        subarray_left_index = counter
        subarray_right_index = counter + bootstrap_sample_size
        subarray = trading_strategy_returns[subarray_left_index:subarray_right_index]
        volatility = np.std(subarray)

        bootstrap_sample_volatilities[i] = volatility
        counter = counter + bootstrap_sample_size

    mean_vol_over_bootstraps = np.mean(bootstrap_sample_volatilities)
    estimated_annualized_vol = np.sqrt(NUMBER_OF_YEARLY_TRADING_DAYS) * mean_vol_over_bootstraps

    return estimated_annualized_vol


# estimates daily returns CVaR of a trading strategy from a set of bootstrap samples
def estimate_daily_cvar_from_bootstrap(trading_strategy_returns, bootstrap_sample_size, number_of_bootstrap_samples,
                                       cvar_beta):

    counter = 0
    bootstrap_sample_cvars = np.zeros(number_of_bootstrap_samples)

    for i in range(number_of_bootstrap_samples):
        subarray_left_index = counter
        subarray_right_index = counter + bootstrap_sample_size
        subarray = trading_strategy_returns[subarray_left_index:subarray_right_index]
        cvar = calculate_cvar(subarray, cvar_beta)

        bootstrap_sample_cvars[i] = cvar
        counter = counter + bootstrap_sample_size

    mean_cvar_over_bootstraps = np.mean(bootstrap_sample_cvars)

    return mean_cvar_over_bootstraps


# Analyzes the results of a trading strategy's returns, giving estimate of annualized mean / vol, and the daily CVar
def analyze_trading_strategy(trading_strategy_returns, bootstrap_sample_size, number_of_bootstrap_samples, cvar_beta):

    mean_daily_return = np.mean(trading_strategy_returns)
    annualized_return = mean_daily_return * NUMBER_OF_YEARLY_TRADING_DAYS

    annualized_volatility = estimate_annualized_vol_from_bootstrap(trading_strategy_returns, bootstrap_sample_size,
                                                                   number_of_bootstrap_samples)

    cvar_value = estimate_daily_cvar_from_bootstrap(trading_strategy_returns, bootstrap_sample_size,
                                                    number_of_bootstrap_samples, cvar_beta)

    return annualized_return, annualized_volatility, cvar_value


# imports the testing datasets from the datasets folder
put_testing_set = save_csv_information_no_dates(BOOTSTRAP_TESTING_PUT_INDEX_CSV_FILE_PATH,
                                                TOTAL_NUMBER_OF_BOOTSTRAP_RETURNS)
trend_testing_set = save_csv_information_no_dates(BOOTSTRAP_TESTING_TREND_INDEX_CSV_FILE_PATH,
                                                  TOTAL_NUMBER_OF_BOOTSTRAP_RETURNS)


# Trains the sharpe ratio strategy and prints the distribution of wealth between the two assets to be maintained
put_training_set = save_csv_information_no_dates(BOOTSTRAP_TRAINING_PUT_INDEX_CSV_FILE_PATH, NUMBER_OF_DAYS_IN_HISTORICAL_DATA)
trend_training_set = save_csv_information_no_dates(BOOTSTRAP_TRAINING_TREND_INDEX_CSV_FILE_PATH,
                                                   NUMBER_OF_DAYS_IN_HISTORICAL_DATA)
sharpe_ratio_proportion = generate_sharpe_ratio_maximization_weighting(put_training_set, trend_training_set)
print("put proportion: " + str(sharpe_ratio_proportion))

# test on static strategies -- Calculating results for static trading strategies and plotting histograms
equal_proportion_returns = get_equal_proportion_strategy_returns(put_testing_set, trend_testing_set)
print(analyze_trading_strategy(equal_proportion_returns, BOOTSTRAP_SAMPLE_SIZE, NUMBER_OF_BOOTSTRAP_SAMPLES,
                               BETA_FOR_95_PERCENT_CVAR))
plot_returns_histogram(equal_proportion_returns, "Equal Proportion Strategy",
                       STATIC_STRATEGY_HISTOGRAMS_X_AXIS_RANGE)

full_put_returns = get_fully_put_static_strategy_returns(put_testing_set, trend_testing_set)
print(analyze_trading_strategy(full_put_returns, BOOTSTRAP_SAMPLE_SIZE, NUMBER_OF_BOOTSTRAP_SAMPLES,
                               BETA_FOR_95_PERCENT_CVAR))
plot_returns_histogram(full_put_returns, "100% Put Index Strategy",
                       STATIC_STRATEGY_HISTOGRAMS_X_AXIS_RANGE)

full_trend_returns = get_fully_trend_static_strategy_returns(put_testing_set, trend_testing_set)
print(analyze_trading_strategy(full_trend_returns, BOOTSTRAP_SAMPLE_SIZE, NUMBER_OF_BOOTSTRAP_SAMPLES,
                               BETA_FOR_95_PERCENT_CVAR))
plot_returns_histogram(full_trend_returns, "100% Trend-Following Index Strategy",
                       STATIC_STRATEGY_HISTOGRAMS_X_AXIS_RANGE)

sharpe_ratio_returns = get_sharpe_ratio_strategy_returns(put_testing_set, trend_testing_set)
print(analyze_trading_strategy(sharpe_ratio_returns, BOOTSTRAP_SAMPLE_SIZE, NUMBER_OF_BOOTSTRAP_SAMPLES,
                               BETA_FOR_95_PERCENT_CVAR))
plot_returns_histogram(sharpe_ratio_returns, "Sharpe Ratio Maximization Strategy",
                       STATIC_STRATEGY_HISTOGRAMS_X_AXIS_RANGE)
