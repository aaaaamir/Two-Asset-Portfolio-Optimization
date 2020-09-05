# Contains functions to calculate returns from historical returns data data on a few static trading strategies.
# We consider putting wealth into just two assets: a put option index and a trend-following index.
# Trading is done daily. We work with daily returns data over a 20-year period, with the assumption of no commission or trading fees.
# Author: Amir Farrag
import numpy as np
from scipy import optimize
from Global_Constants import NUMBER_OF_YEARLY_TRADING_DAYS

PUT_WEIGHT_LOWER_BOUND = 0.000001
PUT_WEIGHT_UPPER_BOUND = 0.999999
PUT_WEIGHT_INITIAL_GUESS = 0.5

ALL_IN_PUT_INDEX_CONSTANT = 1
ALL_IN_TREND_INDEX_CONSTANT = 0
EQUAL_PROPORTION_STRATEGY_CONSTANT = 0.5
RISK_FREE_RATE = 0.025


# Given a weighting, computes the set of returns (not percentage) for a static trading strategy over the entire inputted investment horizon
def static_trading_strategy(put_returns_array, trend_returns_array, put_weighting):

    wealth = 1
    counter = 0
    trend_weighting = 1 - put_weighting

    number_of_trading_days_in_test = len(put_returns_array)
    strategy_returns_array = np.zeros(number_of_trading_days_in_test)

    while counter < number_of_trading_days_in_test:

        old_wealth = wealth
        wealth = put_weighting * wealth * (1 + put_returns_array[counter]) + \
            trend_weighting * wealth * (1 + trend_returns_array[counter])

        return_value = (wealth - old_wealth) / old_wealth
        strategy_returns_array[counter] = return_value

        counter = counter + 1

    return strategy_returns_array


# Gives returns for a strategy that is equally invested into the put and trend-following indices, with daily trading
def get_equal_proportion_strategy_returns(put_returns_array, trend_returns_array):
    return static_trading_strategy(put_returns_array, trend_returns_array, EQUAL_PROPORTION_STRATEGY_CONSTANT)


# Gives returns for a strategy that is fully invested into the put index, with no trading
def get_fully_put_static_strategy_returns(put_returns_array, trend_returns_array):
    return static_trading_strategy(put_returns_array, trend_returns_array, ALL_IN_PUT_INDEX_CONSTANT)


# Gives returns for a strategy that is fully invested into the trend-following index, with no trading
def get_fully_trend_static_strategy_returns(put_returns_array, trend_returns_array):
    return static_trading_strategy(put_returns_array, trend_returns_array, ALL_IN_TREND_INDEX_CONSTANT)


# Gives returns for the sharpe-ratio maximization defined trading strategy
def get_sharpe_ratio_strategy_returns(put_returns_array, trend_returns_array):
    sharpe_ratio_put_weighting = generate_sharpe_ratio_maximization_weighting(put_returns_array, trend_returns_array)
    return static_trading_strategy(put_returns_array, trend_returns_array, sharpe_ratio_put_weighting)


# Defines a static trading strategy that picks an allocation of wealth into each asset such that the portfolio's 
# - expected Sharpe Ratio is maximized. We find an optimal weight in the put portfolio: The optimizer used is 
# - Sequential Least Squares Programming
def generate_sharpe_ratio_maximization_weighting(put_returns_array, trend_returns_array):

    expected_annualized_put_returns = np.mean(put_returns_array) * NUMBER_OF_YEARLY_TRADING_DAYS
    expected_annualized_trend_returns = np.mean(trend_returns_array) * NUMBER_OF_YEARLY_TRADING_DAYS
    annualized_put_sample_vol = np.std(put_returns_array) * np.sqrt(NUMBER_OF_YEARLY_TRADING_DAYS)
    annualized_trend_sample_vol = np.std(put_returns_array) * np.sqrt(NUMBER_OF_YEARLY_TRADING_DAYS)
    estimated_covariance = np.cov(put_returns_array, trend_returns_array)[0, 1] * NUMBER_OF_YEARLY_TRADING_DAYS

    sharpe_ratio_arguments = (expected_annualized_put_returns, expected_annualized_trend_returns,
                              annualized_put_sample_vol, annualized_trend_sample_vol, estimated_covariance)
    weight_bounds = [(PUT_WEIGHT_LOWER_BOUND, PUT_WEIGHT_UPPER_BOUND)]

    opt_result = optimize.minimize(sharpe_ratio, x0=np.array(PUT_WEIGHT_INITIAL_GUESS),
                                   args=sharpe_ratio_arguments, bounds=weight_bounds)
    optimized_put_weighting = opt_result.x

    return optimized_put_weighting


# Returns an estimation of expected sharpe ratio multiplied by negative one
def sharpe_ratio(put_weight, expected_put_returns, expected_trend_returns, put_vol, trend_vol, covariance):
    numerator = put_weight * expected_put_returns + (1 - put_weight) * expected_trend_returns - RISK_FREE_RATE
    denominator = np.sqrt(np.square(put_weight * put_vol) + np.square((1 - put_weight) * trend_vol) +
                          2 * put_weight * (1 - put_weight) * covariance)
    return - numerator / denominator
