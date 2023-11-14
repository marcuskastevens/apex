'''


'''
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import scipy.stats as stats


class IModel(ABC):
    """
    The builder interface that specifies the methods for creating a formulaic alpha
    """

    
    @abstractmethod
    def __init__(self, X, y):
        raise NotImplementedError

    
    @abstractmethod
    def fit(self):
        raise NotImplementedError
    
    
    @abstractmethod
    def predict(self):
        raise NotImplementedError


class LinearRegression(IModel):
    """
    Follows the builder interface.
    """

    def __init__(self, X, y, fit_intercept: bool = True) -> None:
        
        super(IModel).__init__()

        self.X = X
        self.y = y
        self._preprocess()
        self.fit_intercept = fit_intercept
        self.nobs = np.shape(self.X)[0]
        self.n_features = np.shape(self.X)[1]
        self.model_degrees_of_freedom = self.n_features + 1 
        self.residual_degrees_of_freedom = self.nobs - self.model_degrees_of_freedom 

        # Data integrity assertions
        assert np.shape(self.X)[0] == np.shape(self.y)[0]      
    

    def _preprocess(self) -> None:
        """
        Feature preprocessing.
        """

        # Convert feature to np.ndarray
        if isinstance(self.X, pd.DataFrame):
            
            # Handles univariate or multivariate regression 
            self.X = np.array(X)
        
        elif isinstance(self.X, pd.Series):
            
            # Handles univariate regression
            self.X = np.array(X).reshape(-1, 1)

        elif isinstance(self.X, np.ndarray):

            # Handles univariate or multivariate regression
            if len(np.shape(self.X)) == 1:
                self.X = self.X.reshape(-1, 1)

        else:
            raise NotImplementedError('LinearRegression does not support X values of type {}'.format(type(self.X)))
        
        # Convert endogenous variable to np.ndarray
        if isinstance(self.y, pd.DataFrame):

            # Handles unsupported endogenous variable
            if np.shape(self.y)[1] != 1:
                raise TypeError('y must be of shape (nobs, 1). y is currently  of shape: {}'.format(np.shape(self.y)))

        elif isinstance(self.y, pd.Series):
            
            # Handles pd.Series endogenous variable
            self.y = np.array(self.y).reshape(-1, 1)
        
        elif isinstance(self.y, np.ndarray):

            # Handles np.ndarray endogenous variable
            if len(np.shape(self.y)) == 1:
                self.y = self.y.reshape(-1, 1)
        
        else:
            raise NotImplementedError('LinearRegression does not support y values of type {}'.format(type(self.y)))

        assert np.shape(self.X)[0] == np.shape(self.y)[0], 'X and y must be the same length. X is of shape: {}, y is of shape: {}'.format(np.shape(self.X), np.shape(self.y))
        assert np.shape(self.y)[1] == 1, 'y must be of shape (nobs, 1). y is currently  of shape: {}'.format(np.shape(self.y))
        assert np.all(np.isnan(self.X) == False), 'X cannot contain NaN values'
        assert np.all(np.isnan(self.y) == False), 'y cannot contain NaN values'
        
        
    def _standardize(self) -> None:
        """
        Demean or add intercept to feature.
        """
        
        # Compute mean of feature
        self.X_mean = np.mean(self.X, axis=0)

        # Demean feature
        self.X_demeaned = self.X - self.X_mean        

        # Add intercept if applicable, otherwise use demeaned feature
        if self.fit_intercept:
            self.X = np.concatenate((np.ones(shape=self.nobs).reshape(-1, 1), self.X), axis=1)
        else:
            self.X = self.X_demeaned


    def _beta(self) -> np.ndarray:
        """
        Compute closed-form solution to OLS.

        Returns:
            np.ndarray: _description_
        """
        return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(self.X), self.X)), np.transpose(self.X)), self.y)
    
    
    def _residuals(self) -> np.ndarray:
        """
        Compute closed-form solution to OLS.

        Returns:
            np.ndarray: _description_
        """
        return self.y - self.predict(self.X)


    def _ssd(self) -> float:
        """
        Compute sum of squared deviations.
        
        Returns:
            float
        """
        return np.dot(np.transpose(self.y - np.mean(self.y)), self.y - np.mean(self.y))
    

    def _sse(self) -> float:
        """
        Compute sum of squared residuals/errors.

        Returns:
            float
        """
        return np.dot(np.transpose(self.residuals), self.residuals)
        
    
    def _mse(self) -> float:
        """
        Compute mean-squared error or variance of residuals, scaled by feature's degrees of freedom.

        Returns:
            float: _description_
        """        
        return self.sse / (self.nobs - self.model_degrees_of_freedom)

    
    def _r_squared(self) -> float:
        return 1 - self.sse / self.ssd

    
    def _var_beta(self) -> np.ndarray:
        """
        Compute variance of beta.

        Returns:
            np.ndarray: _description_
        """
        return self.mse / np.sum(np.square(self.X_demeaned), axis=0).reshape(-1, 1)

    
    def _var_intercept(self) -> np.ndarray:
        """
        Compute variance of intercept.

        Returns:
            np.ndarray: _description_
        """
        return np.array(self.mse * (1 / self.nobs + np.sum(np.square(self.X_mean) / np.sum(np.square(self.X_demeaned), axis=0)))).reshape(-1, 1)

    
    def _standard_error(self) -> np.ndarray:
        """
        Compute standard error of beta.

        Args:
            var_beta (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        # Compute standard error of feature beta
        var_beta =  self._var_beta()
        standard_error = np.sqrt(var_beta)

        # Compute standard error of intercept beta
        if self.fit_intercept:
            var_intercept = self._var_intercept()
            standard_error = np.concatenate((np.sqrt(var_intercept), standard_error))
        
        return standard_error

    
    def _t_stat(self) -> np.ndarray:
        """
        Compute t-stat of beta.

        Returns:
            np.ndarray: _description_
        """
        return self.beta / self.standard_error

    
    def _t_stat_p_value(self) -> np.ndarray:
        """
        Compute p-value of t-stat of beta.

        Returns:
            np.ndarray: _description_
        """
        
        p_value = np.empty(shape=np.shape(self.t_stat))
        
        for i, t in enumerate(self.t_stat):
            p_value[i, 0] = 2 * (1 - stats.t.cdf(np.abs(t[0]), df=self.residual_degrees_of_freedom))
        
        return p_value

    
    def _regression_statistics(self) -> None:
        """
        Get relevant statistics for OLS.
        
        Supported statistics:
        1) MSE
        2) Beta standard error
        3) Beta t-stat
        4) Beta p-value
        """
        
        self.ssd = self._ssd()
        self.sse = self._sse()
        self.mse = self._mse()
        self.r_squared = self._r_squared()
        self.standard_error = self._standard_error()
        self.t_stat = self._t_stat()
        self.p_value = self._t_stat_p_value()

    def predict(self, X) -> None:
        if self.beta is not None:
            return np.dot(X, self.beta)
        else:
            raise NotImplementedError('Call LinearRegression.fit() method before calling LinearRegression.predict()!')


    def fit(self) -> None:

        # Preprocess feature
        self._standardize()

        # Compute closed-form OLS beta
        self.beta = self._beta()
        
        # Compute residuals
        self.residuals = self._residuals()
        
        #  Compute summary statistics
        self._regression_statistics()

    def summary(self) -> pd.DataFrame:
        """
        Convert np.ndarray objects into a user-friendly pd.DataFrame.

        Returns:
            pd.DataFrame: A summary table with relevant statistics.
        """
        
        columns = ['Coefficient', 'Standard Error', 'T-Stat', 'P-value']

        if self.fit_intercept:
            index = ['Intercept'] + [f'X{i}' for i in range(1, self.n_features + 1)]
        else:
            index = [f'X{i}' for i in range(1, self.n_features + 1)]

        data = {
            'Coefficient': self.beta.flatten(),
            'Standard Error': self.standard_error.flatten(),
            'T-Stat': self.t_stat.flatten(),
            'P-value': self.p_value.flatten()
        }

        summary_df = pd.DataFrame(data, index=index, columns=columns)
        
        # Additional summary statistics
        additional_stats = {
            'R-squared': self.r_squared[0][0],
            'MSE': self.mse[0][0],
            'SSD': self.ssd[0][0],
            'SSE': self.sse[0][0]
        }

        additional_stats_series = pd.Series(additional_stats, name='Other')
        summary_df = pd.concat([summary_df, additional_stats_series], axis=1).round(4)

        return summary_df

