import pandas as pd
import os
import datetime


def create_submission(prediction, path, name, test_df):
    submission = pd.DataFrame()
    submission['item_id'] = test_df['item_id']
    submission['deal_probability'] = prediction
    
    submission.loc[submission['deal_probability'] < 0, 'deal_probability'] = 0
    submission.loc[submission['deal_probability'] > 1, 'deal_probability'] = 1
    fname = "submission_{}_{}.csv".format(name, str(datetime.datetime.now()).replace(':', '-'))
    submission.to_csv(os.path.join(path, fname), index=False)