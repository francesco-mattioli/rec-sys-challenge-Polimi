import scipy.sparse as sps
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype

# imports for .env usage
import os
from dotenv import load_dotenv
load_dotenv()


class DataReader(object):

    # Convert a dataframe object into a csr object
    def dataframe_to_csr(self,dataframe):
        users = dataframe['UserID'].unique()
        items = dataframe['ItemID'].unique()
        shape = (len(users), len(items))

        # Create indices for users and items
        user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
        item_cat = CategoricalDtype(categories=sorted(items), ordered=True)
        user_index = dataframe["UserID"].astype(user_cat).cat.codes
        item_index = dataframe["ItemID"].astype(item_cat).cat.codes

        # Conversion via COO matrix
        coo = sps.coo_matrix(
            (dataframe["Data"], (user_index.values, item_index.values)), shape=shape)
        csr = coo.tocsr()
        return csr

    
    def load_urm(self):
        interactions_and_impressions = pd.read_csv(filepath_or_buffer=os.getenv('INTERACTIONS_AND_IMPRESSIONS_PATH'),
                                                   sep=',',
                                                   names=[
                                                       'UserID', 'ItemID', 'Impressions', 'Data'],
                                                   header=0,
                                                   dtype={'UserID': np.int32, 'ItemID': np.int32, 'Impressions': np.object0, 'Data': np.int32})
        urm = interactions_and_impressions.drop(['Impressions'], axis=1)
        # removing duplicated (user_id,item_id) pairs
        urm = urm.drop_duplicates(keep='first')
        # removing (user_id,item_id) pairs with data set to 1
        urm = urm[urm.Data != 1]
        # replacing data which is 0 with 1
        urm = urm.replace({'Data': {0: 1}})

        return self.dataframe_to_csr(urm)
       

    def load_target(self):
        df_original = pd.read_csv(filepath_or_buffer=os.getenv('TARGET_PATH'),
                                  sep=',',
                                  header=0,
                                  dtype={'user_id': np.int32})
        df_original.columns = ['user_id']
        user_id_list = df_original['user_id'].values
        user_id_unique = np.unique(user_id_list)
        return user_id_unique
