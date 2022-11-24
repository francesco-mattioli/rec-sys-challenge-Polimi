import scipy.sparse as sps
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
import scipy.sparse as sps

# imports for .env usage
import os
from dotenv import load_dotenv
load_dotenv()


class DataReader(object):

    # Convert a dataframe object into a csr object
    def dataframe_to_csr(self,dataframe):
        users = dataframe['UserID'].unique()
        items = dataframe['ItemID'].unique()
        
        print(">>> number of users (only those who have watched at least movie): ", len(users))
        print(">>> number of items: ", len(items))
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

    
    def load_binary_urm(self):
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
        watchers_urm = urm[urm.Data != 1]
        # replacing data which is 0 with 1
        watchers_urm = watchers_urm.replace({'Data': {0: 1}})
        return self.dataframe_to_csr(watchers_urm)
    '''
    def load_augmented_binary_urm_df_old(self):
        interactions_and_impressions = pd.read_csv(filepath_or_buffer=os.getenv('INTERACTIONS_AND_IMPRESSIONS_PATH'),
                                                   sep=',',
                                                   names=[
                                                       'UserID', 'ItemID', 'Impressions', 'Data'],
                                                   header=0,
                                                   dtype={'UserID': np.int32, 'ItemID': np.int32, 'Impressions': np.object0, 'Data': np.int32})
        interactions = interactions_and_impressions.drop(['Impressions'], axis=1)
        df = interactions.replace({'Data': {0: 1}})
        df = df.drop_duplicates(keep='first')
        ######### Create watchers_urm: the urm having the pairs (user,item) in which a user have watched the paired item at least once
        # remove duplicated (user_id,item_id) pairs
        #df = interactions.drop_duplicates(keep='first')
        # remove (user_id,item_id) pairs with data set to 1
        #df = df[df.Data != 1]
        # replace data which is 0 with 1
        #watchers_urm = df.replace({'Data': {0: 1}})

        ######### Create openers_urm: the urm having the pairs (user,item) in which a user have opened at least 3 times an item page
        # remove rows where data is 0 in order to have only users who have opened some item pages
        #df = interactions[interactions.Data != 0]
        # groupby UserID and ItemID keeping the columns index
        #df=df.groupby(['UserID','ItemID'],as_index=False)
        # count occurrences of pairs (user,item)
        #df=df['Data'].sum()
        # keep only users who have opened an item more than 0 times 
        #openers_urm=df[df.Data>0]
        # replace the number of times a user opened an item page (which is in column 'Data') with '1'
        #openers_urm['Data']=1

        ######### Create the augmented urm: the union of watchers_urm and openers_urm
        # union of watchers and openers, drop the duplicates and sort by the pair (userid,itemid) in order to have a proper formatting
        union_urm = pd.concat([watchers_urm,openers_urm],ignore_index=True).drop_duplicates(keep='first').sort_values(['UserID', 'ItemID'])
        return union_urm
    '''
    
    def load_augmented_binary_urm_df(self):
        interactions_and_impressions = pd.read_csv(filepath_or_buffer=os.getenv('INTERACTIONS_AND_IMPRESSIONS_PATH'),
                                                    sep=',',
                                                    names=[
                                                        'UserID', 'ItemID', 'Impressions', 'Data'],
                                                    header=0,
                                                    dtype={'UserID': np.int32, 'ItemID': np.int32, 'Impressions': np.object0, 'Data': np.int32})
        interactions = interactions_and_impressions.drop(['Impressions'], axis=1)
        df = interactions.replace({'Data': {0: 1}})
        df = df.drop_duplicates(keep='first')
        return df
       
    def load_augmented_binary_urm(self):
        urm = self.load_augmented_binary_urm_df()
        return self.dataframe_to_csr(urm)

    def load_urm(self):
        interactions_and_impressions = pd.read_csv(filepath_or_buffer="Dataset/interactions_and_impressions.csv",
                                            sep=',',
                                            names=[
                                                'UserID', 'ItemID', 'Impressions', 'Data'],
                                            header=0,
                                            dtype={'UserID': np.int32, 'ItemID': np.int32, 'Impressions': np.object0, 'Data': np.int32})
        df = interactions_and_impressions.drop(['Impressions'], axis=1)
        ### for each pair (user,item), count the number interactions with data set to '0'
        # filter out rows with data set to '1'
        df = df[df.Data != 1]
        # groupby UserID and ItemID keeping the columns index
        df=df.groupby(['UserID','ItemID'],as_index=False)
        # count occurrences of pairs (user,item)
        df_number_of_watched_episodes=df['Data'].count()

        data_ICM_length =  pd.read_csv(filepath_or_buffer="Dataset/data_ICM_length.csv",
                                                    sep=',',
                                                    names=[
                                                        'ItemID', 'FeatureID', 'Data'],
                                                    header=0,
                                                    dtype={'item_id': np.int32, 'feature_id': np.int32, 'data': np.int32})
        # drop feature_id column because it is always '0'
        data_ICM_length = data_ICM_length.drop(['FeatureID'], axis=1)
        # calculate average of number of episodes in order to assign it to items without episodes information
        average_number_of_episodes=data_ICM_length['Data'].mean()
        # create personalized urm
        # join df_number_of_watched_episodes with data_ICM_length on ItemID and fill NaN values with the average number of episodes
        df= df_number_of_watched_episodes.merge(data_ICM_length, on='ItemID', how='left').fillna({'Data_y':average_number_of_episodes})
        df = df.rename({'Data_x':'NumWatchedEpisodes','Data_y':'TotNumEpisodes'},axis=1)
        df['A'] = df['NumWatchedEpisodes']/df['TotNumEpisodes']

        # produce urm
        df = df.drop(['NumWatchedEpisodes','TotNumEpisodes'], axis=1)
        urm = df.rename({'A':'Data'},axis=1)
        return self.dataframe_to_csr(urm)



    def load_target(self):
        df_original = pd.read_csv(filepath_or_buffer=os.getenv('TARGET_PATH'),
                                  sep=',',
                                  header=0,
                                  dtype={'user_id': np.int32})
        df_original.columns = ['user_id']
        user_id_list = df_original['user_id'].values
        user_id_unique = np.unique(user_id_list)
        print(">>> number of target users: {}".format(len(user_id_list)))
        return user_id_unique

    
    def load_icm(self):
        data_icm_type =  pd.read_csv(filepath_or_buffer=os.getenv('DATA_ICM_TYPE_PATH'),
                                                    sep=',',
                                                    names=[
                                                        'item_id', 'feature_id', 'data'],
                                                    header=0,
                                                    dtype={'item_id': np.int32, 'feature_id': np.int32, 'data': np.int32})

        return self.dataframe_to_csr(data_icm_type)

    # for S-SLIM
    def load_powerful_binary_urm(self):
        data_icm_type =  pd.read_csv(filepath_or_buffer=os.getenv('DATA_ICM_TYPE_PATH'),
                                                    sep=',',
                                                    names=[
                                                        'item_id', 'feature_id', 'data'],
                                                    header=0,
                                                    dtype={'item_id': np.int32, 'feature_id': np.int32, 'data': np.int32})
        # Swap the columns from (item_id, feature_id, data) to (feature_id, item_id, data)
        swap_list = ["feature_id","item_id","data"]
        f = data_icm_type.reindex(columns=swap_list)
        f = f.rename({'feature_id':'UserID','item_id':'ItemID','data':'Data'},axis=1)

        urm = self.load_augmented_binary_urm_df()

        # urm times alpha
        urm['Data'] = 0.7 * urm['Data']
        # f times (1-aplha)
        f['Data']  = 0.3 * f['Data']
        # Change UserIDs of f matrix in order to make recommender work
        f['UserID'] = 41634 + f['UserID']

        powerful_urm = pd.concat([urm,f],ignore_index=True).sort_values(['UserID', 'ItemID'])

        return self.dataframe_to_csr(powerful_urm)


    ###########################################################################################################################################################
    # JUST STATISTICS
    def print_statistics(self,target):
        interactions_and_impressions = pd.read_csv(filepath_or_buffer=os.getenv('INTERACTIONS_AND_IMPRESSIONS_PATH'),
                                                   sep=',',
                                                   names=[
                                                       'UserID', 'ItemID', 'Impressions', 'Data'],
                                                   header=0,
                                                   dtype={'UserID': np.int32, 'ItemID': np.int32, 'Impressions': np.object0, 'Data': np.int32})
        urm = interactions_and_impressions.drop(['Impressions'], axis=1)
        # removing duplicated (user_id,item_id) pairs
        urm = urm.drop_duplicates(keep='first')
        print(">>> number of users in interactions_and_impressions: {}".format(len(urm['UserID'].unique())))
        print(">>> number of unique users in target that are not in interactions_and_impressions: {}".format(len(np.setdiff1d(target,urm['UserID'].unique()))))
        print(">>> number of unique users in interactions_and_impressions that are not in target: {}".format(len(np.setdiff1d(urm['UserID'].unique(),target))))
        
        # removing (user_id,item_id) pairs with data set to 1
        watchers_urm = urm[urm.Data != 1]
        print('>>> number of unique users in target that are not in "list of users that have watched at least a movie": {}'.format(len(np.setdiff1d(target,watchers_urm['UserID'].unique()))))
        print('>>> number of unique users in "list of users that have watched at least a movie" that are not in target: {}'.format(len(np.setdiff1d(watchers_urm['UserID'].unique(),target))))
        print('>>> number of unique users in interactions_and_impressions that are not in "list of users that have watched at least a movie": {}'.format(len(np.setdiff1d(urm['UserID'].unique(),watchers_urm['UserID'].unique()))))

