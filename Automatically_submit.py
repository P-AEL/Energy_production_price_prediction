import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.smpickle import load_pickle
import comp_utils
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from datetime import datetime, timedelta
import pickle as pkl

def convert_int32_to_int(data):
    if isinstance(data, dict):
        return {k: convert_int32_to_int(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_int32_to_int(item) for item in data]
    elif isinstance(data, np.int32):
        return int(data)
    else:
        return data

def Update(model_strom=None,model_bid=None):
    #create df with times
    rebase_api_client = comp_utils.RebaseAPI(api_key = open("A-Team_key.txt").read())
    submission_data=pd.DataFrame({"datetime":comp_utils.day_ahead_market_times()})
    #get electricity production from model
    if model_strom is not None:
        # code for quantile regression
        pass
    else:
        quantiles = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"]
        for quantile in quantiles:
            submission_data[quantile] = np.random.randint(300,1600,size=len(submission_data))
    #add market_bid
    if model_bid is not None:
        # code for quantile regression
        pass
    else:
        submission_data["market_bid"]=np.random.randint(300,1600,size=len(submission_data))

    submission_data = comp_utils.prep_submission_in_json_format(submission_data)
    submission_data = convert_int32_to_int(submission_data)
    #submit data
    rebase_api_client.submit(submission_data)
    print("Submitted data")

if __name__ == "__main__":
    Update(model_strom=None,model_bid=None)