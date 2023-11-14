'''
Library for volatilty and variance estimation functions. 

Supported Estimation Algorithms:
1) Exponentially-Weighted Volatility Estimation
2) HAR Volatility Estimation

Future Estimation Algorithms:
1) GARCH
2) TGARCH
3) Etc.
'''
import pandas as pd
import numpy as np
import statsmodels.api as sm

# -------------------------------- Volatility Models --------------------------------

def ewma_vol(returns: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """
    Compute the exponentially-weighted moving average estimate of volatility.
    
    Args:
        returns (pd.DataFrame): historical returns.
        lookback (int, optional): exponentially-weighted daily returns with a "lookback" day center-of-mass.
        
    Returns:
        pd.Series: EWMA volatility estimates.
    """
            
    # Calculate volatility estimates
    vols = returns.ewm(span=lookback).std().iloc[-1]    
        
    return vols


def ewma_vol_manual(returns: pd.DataFrame, alpha: float = .95, span: float = None) -> pd.Series:
    """
    Without libraries, compute the exponentially-weighted moving average estimate of volatility.

    Generalize form: alpha * (1 - alpha)^k
    
    Args:
        returns (pd.DataFrame): historical returns.
        alpha (float, optional): decay factor.
        span (float, optional): represents the decay factor in terms of the number of observations. 
                                Specifically, span is defined as the number of periods required for 
                                the EWMA to span the entire range of the data.
        
    Returns:
        pd.Series: EWMA volatility estimates.
    """
    
    # If decay is specified in terms of number of observations
    if span:
        alpha = 1 - 2 / (1 + span)

    # Square returns        
    squared_returns = np.square(returns)

    # Compute weights for squared returns (must reverse numerical range to capture reverse time weighting)
    weights = pd.Series(np.power(alpha, np.arange(len(returns)-1, -1, -1)) * (1 - alpha), 
                        index=squared_returns.index)          

    # Calculate EWMA variance
    # ewma_variance = np.sum(weights * squared_returns) 
    ewma_variance = pd.Series() 
    for col, ret in squared_returns.items():
        ewma_variance[col] = np.sum(weights*ret)

    # Compute volatility as the square root of EWMA variance
    vols = np.sqrt(ewma_variance)

    return vols

def har_vol(returns: pd.DataFrame, lags: list, realized_vol_n: int = 21, rolling_window: int = 12) -> pd.Series:
    """
    Applies the Heteroskedastic Autoregressive model to forecast "realized_vol_n" day volatility. Leverages a rolling
    multiple regression of lagged realized volatilty.    

    Args:
        data (pd.DataFrame): Dataframe containing the historical volatility series.
        lags (list): List of lag periods for the HAR model.
        forecast_n (int): Number of periods ahead to forecast.

    Returns:
        pd.DataFrame: DataFrame containing the forecasted volatility series.
    """
    vols = pd.Series()

    for ticker, ret in returns.items():

        # Get realized vols
        variance_series = get_realized_variance(ret, n=realized_vol_n).dropna() 
        y = variance_series

        # Prepare lagged vols as predictors for training
        X = pd.concat([variance_series.shift(lag) for lag in lags], axis=1).dropna()
        X.columns = [f'lag{lag}' for lag in lags]

        # Get current lagged vols for ex-ante vol forecasting
        X_current = pd.concat([variance_series.shift(lag-1) for lag in lags], axis=1).iloc[-1].dropna()
        X_current.index = [f'lag{lag-1}' for lag in lags]
        
        # Prepare indices to ensure alignment between X and y
        indices = y.index.intersection(X.index)
        y = y.loc[indices]
        X = X.loc[indices]    

        # Get rolling data
        X_train = X.tail(rolling_window) # X.iloc[-rolling_window :]
        y_train = y.tail(rolling_window) # y.iloc[-rolling_window :]

        # Estimate the model parameters using OLS regression
        model = sm.OLS(y_train, sm.add_constant(X_train))
        results = model.fit()

        # Initialize an empty DataFrame to store the rolling regression results
        betas_columns = ['alpha'] + [f'beta_{i}' for i in np.arange(1, len(lags)+1)]

        # Update regression params
        betas = pd.Series(results.params.values, index=betas_columns)
        
        # Plot historical OLS vols 
        vol_forecast_ols = np.sqrt(betas['alpha'] + pd.DataFrame(X_train.values * betas.iloc[-len(lags):].values).sum(1))
        vol_forecast_ols.index = X_train.index
        vol_forecast_ols.name = "HAR OLS Vol"

        # Forecast ex-ante vol
        vol_forecast_ex_ante = np.sqrt(betas['alpha'] + np.dot(X_current, betas.iloc[-len(lags):]))
        
        # Control for negative vol estimates (corner-case where OLS forecasts negative vol)
        if np.isnan(vol_forecast_ex_ante):
            return har_vol(returns=ret, lags=lags, realized_vol_n=realized_vol_n, rolling_window=rolling_window+1)
        
        # FIX THIS PART ASAP --- PROBABLY WANT A SEPARATE FUNCTION TO RUN HAR MODEL ON EACH ASSET INSTEAD OF WITH A FOR LOOP
        
        vol_forecast_ex_ante = pd.Series(vol_forecast_ex_ante, name = ticker)

        # Concatonate current vol predictions with other vols
        vols = pd.concat([vols, vol_forecast_ex_ante], axis=1)
    
    return vols


def get_realized_variance(returns: pd.Series, n: int = 21):
    """
    Computes realized varaince using the alternate, non-statistical formula for n day realized variance. 
    Realized variance is computed by summing the squared returns over n days. No time scaling
    is needed with this method.

    variance_n = sum0,n[(x,i)^2]
    volatilty_n = sqrt(variance_n)    
 
    Args:
        returns (pd.Series or pd.DataFrame): time series of daily returns.
        n (int, optional): time frame of realized variance. Defaults to 21 for monthly.

    Returns:
        pd.Series or pd.DataFrame: time series of realized "n" day variance.
    """
    
    resample_n = pd.offsets.BDay(n)
    realized_variance = returns.resample(resample_n).apply(lambda x: np.sum(np.square(x)))
    return realized_variance