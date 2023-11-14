'''
Module for portfolio optimization objective/loss functions.

'''

import pandas as pd
import numpy as np

# ------------------------------------------------------------------------- Objective Functions -------------------------------------------------------------------------
def sharpe_ratio_obj(w: pd.Series, cov: pd.DataFrame, expected_returns: pd.Series) -> float:
    """ 
    Loss function for maximum SR optimization.

    Args:
        w (pd.Series): position weights vector.
        cov (pd.DataFrame): covariance matrix.
        expected_returns (pd.Series): expected returns vector.        

    Returns:
        float: negative SR.
    """

    # Portfolio expected return
    mu = np.dot(np.transpose(w), expected_returns)

    # Portfolio vol
    sigma = np.sqrt(np.dot(np.dot(w.T, cov), w))

    # Portfolio SR
    sharpe_ratio = mu / sigma

    # Negative SR for minimization
    loss = -sharpe_ratio
    
    return loss

def expected_return_obj(w: pd.Series, expected_returns: pd.Series) -> float:
    """ 
    Loss function for maximum expected return optimization.

    Args:
        w (pd.Series): position weights vector.
        expected_returns (pd.Series): expected returns vector.

    Returns:
        float: negative expected return.
    """

    # Portfolio expected return
    mu = np.dot(np.transpose(w), expected_returns)

    # Negative mu for minimization
    loss = -mu 
    
    return loss


def expected_return_turnover_aversion_obj(w1: pd.Series, w0: pd.Series, expected_returns: pd.Series, max_turnover: float, gamma: float = 2) -> float:
    """ 
    Loss function for constrained maximum expected return optimization. 

    Args:
        w (pd.Series): position weights vector.
        expected_returns (pd.Series): expected returns vector.
        gamma (float): turnover aversion term. Defaults to 10000 to create effective hard constraint.

    Returns:
        float: constrained negative expected return.
    """

    # Portfolio expected return
    mu = np.dot(np.transpose(w1), expected_returns)

    # Negative my for minimization
    loss = gamma * np.abs(np.sum(np.abs(w1 - w0)) - max_turnover) - mu
    
    return loss
