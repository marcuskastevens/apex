'''
Module for portfolio optimization functions.

'''

from scipy.optimize import minimize as opt
from . import obj_functions
from scipy.optimize import Bounds
import statsmodels.api as sm
from scipy import stats
import pandas as pd
import numpy as np

# -------------------------------- Optimization Functions --------------------------------

def max_expected_return(expected_returns: pd.DataFrame, 
                        cov: pd.DataFrame, 
                        w0: pd.Series, 
                        first_opt: bool, 
                        gross_exposure = 1, 
                        net_exposure = 0,
                        max_turnover = 0.25, 
                        verbose = False
                        ) -> pd.Series:

    # Match tickers across expected returns and historical returns
    expected_returns = expected_returns.dropna()

    n = expected_returns.shape[0]

    if n == 0:
        print("No views")
        return None

    if n > 0:
        
        # Initial guess is naive 1/n portfolio
        if first_opt:
            initial_guess = np.array([1 / n] * n)
            max_turnover = 1
        else:
            initial_guess = w0

        constraints =  [{"type": "eq", "fun": lambda w: np.sum(w) - net_exposure},  # Dollar neutrality 
                        {"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - gross_exposure},  # GNV 
                        {"type": "ineq", "fun": lambda w: np.sum(np.abs(w - w0))*500 - max_turnover*500}  # Max turnover
                        ]
        
        w = opt(obj_functions.expected_return_obj, 
                initial_guess,
                args=(expected_returns), 
                method='SLSQP',
                constraints=constraints
                )['x']
        
        w = pd.Series(w, index=expected_returns.index)

        turnover = np.sum(np.abs(w - w0))
              
        # Print relevant allocation information
        if verbose:
            print(f'Ex Ante Vol: {np.sqrt(np.dot(np.dot(w, cov), w))}')
            print(f'Gross Exposure (GNV): {np.sum(np.abs(w))}')
            print(f'Net Exposure: {np.sum(w)}')
            print(f'Ex Ante Sharpe Ratio: {-obj_functions.sharpe_ratio_obj(w, expected_returns, cov)}')
            
            print(f'Turnover: {turnover}')    
        
        return w
    
    return None


# def max_expected_return(expected_returns: pd.DataFrame, 
#                         cov: pd.DataFrame, 
#                         w0: pd.Series, 
#                         first_opt: bool, 
#                         gross_exposure = 1, 
#                         net_exposure = 0,
#                         max_turnover = 0.25, 
#                         verbose = False
#                         ) -> pd.Series:

#     # Match tickers across expected returns and historical returns
#     expected_returns = expected_returns.dropna()

#     n = expected_returns.shape[0]

#     if n == 0:
#         print("No views")
#         return None

#     if n > 0:
        
#         # Initial guess is naive 1/n portfolio
#         if first_opt:
#             initial_guess = np.array([1 / n] * n)
#             max_turnover = 1
#         else:
#             initial_guess = w0

#         constraints =  [{"type": "eq", "fun": lambda w: np.sum(w) - net_exposure},  # Dollar neutrality 
#                         {"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - gross_exposure},  # GNV
#                         ]
        
#         w = opt(obj_functions.expected_return_turnover_aversion_obj,
#                 initial_guess,
#                 args=(w0, expected_returns, max_turnover),
#                 method='SLSQP',
#                 constraints=constraints
#                 )['x']
        
#         w = pd.Series(w, index=expected_returns.index)
              
#         # Print relevant allocation information
#         if verbose:
#             print(f'Ex Ante Vol: {np.sqrt(np.dot(np.dot(w, cov), w))}')
#             print(f'Gross Exposure (GNV): {np.sum(np.abs(w))}')
#             print(f'Net Exposure: {np.sum(w)}')
#             print(f'Ex Ante Sharpe Ratio: {-obj_functions.sharpe_ratio_obj(w, expected_returns, cov)}')
        
#         return w
    
#     return None

# -------------------------------- Utils --------------------------------

# Hash map between optimization methods and their respective functions
optimization_algo_map = {"max_expected_return" : max_expected_return}


def update_args(args: dict, returns: pd.DataFrame, w: dict, opt_method: str):

    # Turnover constrained optimization functions
    turnover_constrained_opts = ["max_expected_return"]

    # Get specified optimization algorithm
    optimization_algo = optimization_algo_map[opt_method]

    # If turnover constrained opt
    if opt_method in turnover_constrained_opts:

        # Get previous rebal date's portfolio weights
        w0 = pd.DataFrame(w).T.iloc[-1]

        # Ensure tradable assets are compatable
        columns = w0.index.intersection(returns.columns)
                
        # Update args
        args['w0'] = w0[columns]

    # Get opt function args
    supported_args = list(optimization_algo.__code__.co_varnames[:optimization_algo.__code__.co_argcount])

    # Drop unsupported args
    unsupported_args = [key for key in args.keys() if key not in supported_args]
    if len(unsupported_args) > 0: 
        print(f"{unsupported_args} are not supported args in the {opt_method} optimization function!")
        print(f"Supported args are {supported_args}.")
        for key in unsupported_args:
            del args[key]

    # Get required opt function args (slice args for required args)
    required_args = list(optimization_algo.__code__.co_varnames[:optimization_algo.__code__.co_argcount])

    # Check if all required args are defined
    missing_required_args = [key for key in required_args if key not in args.keys()]
    if len(missing_required_args) > 0:
        raise TypeError(f"{opt_method} optimization function missing required argument(s):\n{missing_required_args}")
    
    return args

