"""
Controls the generation of the alpha returns
"""
from __future__ import annotations

import pandas as pd
import datetime as dt

# # Local imports
# sys.path.append(os.getcwd())
# sys.path.append(str(Path(os.getcwd()).parent.absolute()))

from . import alpha_builder, alpha_functions
from ..data import data_utils


DB_PATH = r"C:\Users\marcu\AppData\Roaming\Python\Python311\site-packages\quantlib\database\cache\russell_3000\\"


def alpha_generator(db_data, date):

    # DB_DATA = data_utils.load_cache(DB_PATH + r"russell_3000_cache.pickle")
    # RETURNS_DATA = pd.DataFrame(data_utils.load_cache(DB_PATH + r"adj_close_returns.pickle"))

    # Convert date format
    if isinstance(date, str):
        date = dt.datetime.strptime(date, "%Y-%m-%d").date()
    elif isinstance(date, dt.date):
        pass

    alpha_funcs = [func for func in dir(alpha_functions) if func.startswith("alpha")]
    alpha_models = list(map(lambda alpha_func: alpha_builder.FormulaicAlphaBuilder(alpha_func=getattr(alpha_functions, alpha_func),
                                                                                   date=date,
                                                                                   db_data=db_data,
                                                                                   ),
                                                                                   alpha_funcs
                        )
                    )
    
    w = pd.DataFrame([model.w for model in alpha_models])   
    w.index.name = "alpha_models"

    return w
