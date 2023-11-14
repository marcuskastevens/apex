'''
Library for walk-forward portfolio optimization.

Current Implementations:
1) Generalized Class for Portfolio Optimization
2) Portfolio Stress Testing Methods (Rebalancing Timing Luck)

'''

from .optimization import opt, cov_functions as risk_models
from .alphas import alpha_utils, alpha_generator
import datetime as dt
import pandas as pd
import numpy as np


# -------------------------------- Optimization Classes --------------------------------
class walk_forward_portfolio_optimization:

    
    def __init__(self, 
                 db_data: dict, 
                 returns: pd.DataFrame, 
                 opt_args: tuple, 
                 cov_args: tuple, 
                 opt_method: str = "Max Sharpe Ratio", 
                 cov_method: str = "ewma_cov",
                 rebal_freq: int = 21, 
                 start_date: int = dt.date(2020, 1, 1)
                 ):
        
        self.db_data = db_data
        self.returns = returns        
        self.opt_method = opt_method
        self.cov_method = cov_method
        self.rebal_freq = rebal_freq
        self.start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        
        self.expected_return_algo = alpha_generator.alpha_generator

        self.optimization_algo = opt.optimization_algo_map[self.opt_method]

        self.cov_algo = risk_models.risk_matrix

        self.opt_args = opt_args
            
        self.cov_args = cov_args

        self.cov_estimates = {}

        self.model_views = {}

        self.expected_returns = {}
            
        self.w, self.rebal_w = self.run()
        
    
    def run(self):
        
        # Empty w matrix for indexing purposes
        empty_w = pd.DataFrame(index=self.returns.index)

        # Dict to hold walk-forward optimized weights 
        w = {}

        for i, date in enumerate(self.returns.index[::self.rebal_freq]):
            
            # Initialize empty portfolio
            if i == 0:
                w[date] = pd.Series(np.zeros(self.returns.shape[1]), index=self.returns.columns)
                continue

            # Initial opt bool
            if i == 1:
                first_opt = True
                        
            # Initial opt bool 
            self.opt_args.update({"first_opt": first_opt})

            # Set to False after first opt
            first_opt = False
            
            # Get expanding historical returns
            tmp_returns = self.returns.loc[:date]

            try: 
                # Estimate covariance matrix 
                cov = self.cov_algo(returns=tmp_returns, method=self.cov_method, **self.cov_args)
                self.opt_args.update({"cov" : cov}) 
                self.cov_estimates.update({date: cov})
            except Exception as e:
                print(f'ERROR - {date}')               
                print(e)
                print(tmp_returns.tail(20))
                print(self.cov_algo(returns=tmp_returns, method=self.cov_method, **self.cov_args))
                return tmp_returns

            # Get alpha model views
            model_views = self.expected_return_algo(db_data=self.db_data, date=date)
            self.model_views.update({date: model_views})
            
            # Get alpha model implied expected returns
            model_expected_returns = alpha_utils.model_asset_level_implied_expected_returns(w=model_views, cov=cov)

            # Equal alpha model weight
            model_w = pd.Series(np.array([1/model_views.shape[0]] * model_views.shape[0]))

            # Estimate stock level expected returns
            expected_returns = alpha_utils.asset_level_implied_weighted_expected_returns(expected_returns=model_expected_returns, w=model_w)
            self.expected_returns.update({date: expected_returns})
            self.opt_args.update({"expected_returns": expected_returns})

            # Update portfolio constituents -- due to data errors that cov_algo handled 
            tmp_returns = tmp_returns[cov.columns]

            # Clean, handle errors, & update args where applicable
            opt.update_args(args=self.opt_args, returns=tmp_returns, w=w, opt_method=self.opt_method)

            # Tradable universe size
            n = tmp_returns.shape[1]

            if n > 1:
                # Get optimal weights
                w[date] = self.optimization_algo(**self.opt_args)

            elif n == 1: 
                # 100% weight in single security -- handle short vs. long later
                w[date] = pd.Series({tmp_returns.columns[0] : 1.00})   

            else:
                # No tradable assets on given date
                continue    

        # Save rebal date weights & forward filled weights
        rebal_w = pd.DataFrame(w).T
        w = pd.concat([rebal_w, empty_w], axis=1).ffill()
            
        return w, rebal_w
