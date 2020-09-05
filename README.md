# Two-Asset-Portfolio-Optimization

Portfolio optimization tool for use on two assets -- Calculates summary statistics for inputted data including annualized return and volatility and generates histograms of returns given daily historical returns data for either asset. Also imputes (estimates) missing data points for any daily return value from either asset not found in the given data using a Brownian bridge time series estimation method [1]. Once dataset is finished collection, bootstrap re-sampling from the historical datasets is done to make a testing set for trading strategy assessment. This is a neccessary move since directly re-sampling data prevents us from having to make assumptions about the stochastic process that either asset follows. The bootstrapping is done in a way to ensure the distribution of returns in the testing sets will match those of the historical sets, by utilizing sampling method by Patton et. al [2]. Once the test sets are ready for use, the optimizer computes and prints performance results including estimations of 95% CVaR (a risk measure), monthly return and monthly volatility for four static trading strategies:

- All-in-one full investment into the either index with no trading (2 possible portfolios doing this)
- 50% proportion of wealth invested into each index, with daily portfolio rebalancing
- Division of wealth in each index defined by choosing proportions that will maximize the estimated annual Sharpe ratio, with daily portfolio rebalancing

Implementation is done in python using matplotlib, numPy and sciPy. Unit testing files not included in the repo.

**Note:** The original analysis was done on a put option index and a trend following index, however this code can be easily generalized to fit any two assets. Just change the constants containing the filename of two datasets you intend on using. You can also rename the functions to correspond to two general assets as opposed to a put and trend index respectively, for easier readability if you'd like. I can commit to this process if there is any interest in these modifications!

## More Details of Implementation:

**Data Imputation**

We use a Brownian bridge imputation method to estimate missing returns as presented by Durham and Gallant [1]. For trading days with missing returns data, we make the assumption that the index with missing data will follow a geometric Brownian motion at that time. This allows us to use the Brownian bridge imputation algorithm: 

Let the i(th) day be represented using t(i), and let t(0) < t(1) < t(2). Suppose we are missing the return x(1) that occurred on day t(1), however we have the recorded returns x(0) for day t(0), and x(2) for day t(2). We estimate all of the missing returns by drawing x(1) values from a *normal distribution* with the following attributes:

mean = x(0) + (x(2) - x(0)) / (t(2) - t(0)) * (t(1) - t(0))

standard deviation = sigma ^ 2 * (t(1) - t(0)), where sigma represents the annualized volatility from that year.  

**Portfolio Assessment**

To optimize the portfolio, we first gather testing sets by doing bootstrap re-sampling on our original index datasets; This is done to create 1000 testing datasets, each spanning 30 days of returns. We then use the bootstrap samples to assess the performance of any given portfolio and trading strategy. The bootstrapping is done to add systematic, controlled randomness into our testing, and to have a larger pool of data to test the strategies on. We use the stationary block bootstrap re-sampling method, which is a method presented by Andrew Patton, Dimitris N Politis, and Halbert White in their article "Correction to 'Automatic block-length selection for the dependent bootstrap'" [2].

*Protocol to get the Bootstrap Samples for Testing:*

To create each 30-day bootstrap sample, varied length sub-sequences of consecutive returns are continuously sampled from the historical datasets with replacement. To most accurately emulate the correlation between either index during sampling, the starting point in time and the length of each sampled sub-sequence is held to be identical for either index. The starting point of time for each sample is chosen uniformly at random. Sampled returns wrap around to the beginning of the historical dataset in the case that returns on any day after the final recorded trading day from the historical datasets are chosen in the bootstrap re-sampling process. Each of the sampled sub-sequences for either index are concatenated together until we get two resulting sequences that reach a length of 30 returns or more. Extra returns appearing after the 30th recorded one are then omitted from either bootstrap sample.
To make each bootstrap sample best represent the distribution of historical returns, the length of each sampled time sub-sequence is individually sampled from a geometric distribution. The algorithm for this process is presented in the paper by Patton et. al [2].

*Strategy Evaluation Procedure*

We record each trading strategy’s overall performance by gathering daily returns for the strategy over the bootstrap testing set, then calculating an estimate of the annualized sample mean and annualized sample volatility. We also estimate the daily 95% CVaR of daily returns for each strategy to get a view of the downside risk. Each of the statistics are gathered by taking an arithmetic mean of estimates over each sampled bootstrap test set.
The Sequential Least Squares Programming algorithm from the Python library sciPy was used to solve the optimization problem of maximizing the Sharpe ratio between the two assets.

References:
[1] -- Garland  B  Durham  and  A  Ronald  Gallant.  “Numerical  techniques  formaximum likelihood estimation of continuous-time diffusion processes”. In:Journal of Business & Economic Statistics20.3 (2002), pp. 297–338.
[2] -- Andrew  , Dimitris N Politis, and Halbert White. “Correction to “Automatic block-length selection for the dependent bootstrap” by D. Politis and H. White”. In: Econometric Reviews 28.4 (2009), pp. 372–375.
