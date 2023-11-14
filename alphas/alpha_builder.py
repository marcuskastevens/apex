"""
Using a pseudo-builder design pattern to make the creation of alphas more scalable
and efficient
"""
from __future__ import annotations

import os
import sys
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import datetime as dt

sys.path.append(os.getcwd())
sys.path.append(str(Path(os.getcwd()).parent.absolute()))

from tqdm import tqdm
from . import alpha_utils

TRADING_DAYS = 252
DEFAULT_VOL = 0.40
VOL_WINDOW = 20


class IFormulaicAlpha(ABC):
    """
    The builder interface that specifies the methods for creating a formulaic alpha
    """

    @abstractmethod
    # def get_data(self) -> pd.DataFrame:
    #     """
    #     returns the data
    #     """
        
    #     raise NotImplementedError

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """
        runs the strategy
        """
        raise NotImplementedError


class FormulaicAlphaBuilder(IFormulaicAlpha):
    """
    Follows the Builder interface
    """

    def __init__(self,
                 alpha_func: Callable[[pd.DataFrame], pd.DataFrame],
                 date: str,
                 db_data: pd.DataFrame = None,
                 ) -> None:
        
        super(IFormulaicAlpha).__init__()

        self._alpha_func = alpha_func
        self.date = date
        self.db_data = db_data

        self.raw_signal = {}

        self.run()

    def run(self) -> pd.DataFrame:
        """
        Generates vector of position weights according to alpha model.

        Returns:
            pd.Series: 
        """
        # Get raw alpha signal for each instrument in universe
        for _, (ticker, tmp_data) in enumerate(tqdm(self.db_data.items())):

            # Cut data after and on rebal day
            tmp_data = tmp_data.loc[:self.date]
            
            # Update raw alpha signal
            self.raw_signal[ticker] = self._alpha_func(tmp_data).iloc[-1]

            if _ > 8:
                break

        # Convert raw signal to pd.Series
        self.raw_signal = pd.Series(self.raw_signal)

        # Cross-sectional z-score
        self.views = alpha_utils.z_score(self.raw_signal)

        # Convert to weights vector
        self.w = self.views / np.sum(np.abs(self.views))

        # Assert dollar neutrality
        assert -1.0e-6 < np.sum(self.w) < 1.0e-6

        # Assert GNV = 1
        assert 0.999999 < np.sum(np.abs(self.w)) < 1.000001

        return
