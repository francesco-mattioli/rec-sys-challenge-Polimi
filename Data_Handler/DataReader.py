import scipy.sparse as sps
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
import scipy.sparse as sps
from collections import Counter

# imports for .env usage
import os
from dotenv import load_dotenv
load_dotenv()


class DataReader(object):


    def csr_to_dataframe(self,csr):
        coo=csr.tocoo(copy=False)
        df=pd.DataFrame({'UserID': coo.row, 'ItemID': coo.col, 'Data': coo.data})[['UserID', 'ItemID', 'Data']].sort_values(['UserID', 'ItemID']).reset_index(drop=True)
        return df

    def dataframe_to_csr(self, dataframe,row_name,col_name,cell_name):
        """This method converts a dataframe object into a csr

        Args:
            dataframe (dataframe)
            row_name (str): For example, "UserID"
            col_name (str): For example, "ItemID"
            cell_name (str): For example, "Data"
        Returns:
            csr
        """
        rows = dataframe[row_name].unique()
        columns = dataframe[col_name].unique()

        shape = (len(rows), len(columns))

        # Create indices for users and items
        row_cat = CategoricalDtype(categories=sorted(rows), ordered=True)
        col_cat = CategoricalDtype(categories=sorted(columns), ordered=True)
        row_index = dataframe[row_name].astype(row_cat).cat.codes
        col_index = dataframe[col_name].astype(col_cat).cat.codes

        # Conversion via COO matrix
        coo = sps.coo_matrix(
            (dataframe[cell_name], (row_index.values, col_index.values)), shape=shape)
        csr = coo.tocsr()
        return csr

    def load_binary_urm(self):
        """Load urm in which pairs (user,item) are '1' iff user has watched item

        Returns:
            csr: the urm as csr object
        """
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
        return self.dataframe_to_csr(watchers_urm,'UserID','ItemID','Data')

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
        """Load urm in which pairs (user,item) are '1' iff user has either watched item or opened item's details page

        Returns:
            df: urm as dataframe object
        """
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

    def load_augmented_binary_urm_less_items_df(self):
        """Load urm in which pairs (user,item) are '1' iff user has either watched item or opened item's details page
            removing those elements without informations
        Returns:
            df: urm as dataframe object
        """
        interactions_and_impressions = pd.read_csv(filepath_or_buffer=os.getenv('INTERACTIONS_AND_IMPRESSIONS_PATH'),
                                                   sep=',',
                                                   names=[
            'UserID', 'ItemID', 'Impressions', 'Data'],
            header=0,
            dtype={'UserID': np.int32, 'ItemID': np.int32, 'Impressions': np.object0, 'Data': np.int32})
        interactions = interactions_and_impressions.drop(['Impressions'], axis=1)
        interactions = interactions.replace({'Data': {0: 1}})
        interactions = interactions.drop_duplicates(keep='first')
        icm = self.load_augmented_binary_icm_less_items_df()
        diff = np.setdiff1d(interactions['ItemID'].unique(), icm['item_id'].unique())
        df = interactions[interactions.ItemID.isin(diff)==False]
        df.reset_index(drop=True, inplace=True)
        return df


    def load_augmented_binary_icm_less_items_df(self):
        interactions_and_impressions = pd.read_csv(filepath_or_buffer=os.getenv('INTERACTIONS_AND_IMPRESSIONS_PATH'),
                                                   sep=',',
                                                   names=[
            'UserID', 'ItemID', 'Impressions', 'Data'],
            header=0,
            dtype={'UserID': np.int32, 'ItemID': np.int32, 'Impressions': np.object0, 'Data': np.int32})
        interactions = interactions_and_impressions.drop(['Impressions'], axis=1)
        interactions = interactions.replace({'Data': {0: 1}})
        interactions = interactions.drop_duplicates(keep='first')
        icm = self.load_icm_df()
        diff = np.setdiff1d(icm['item_id'].unique(), interactions['ItemID'].unique())
        df = icm[icm.item_id.isin(diff)==False]
        df.reset_index(drop=True, inplace=True)
        return df


    def load_augmented_binary_icm_less_items(self):
        
        data_icm_type = self.load_augmented_binary_icm_less_items_df()
        
        features = data_icm_type['feature_id'].unique()
        items = data_icm_type['item_id'].unique()
        shape = (len(items), len(features))


        # Create indices for users and items
        features_cat = CategoricalDtype(categories=sorted(features), ordered=True)
        item_cat = CategoricalDtype(categories=sorted(items), ordered=True)
        features_index = data_icm_type["feature_id"].astype(features_cat).cat.codes
        item_index = data_icm_type["item_id"].astype(item_cat).cat.codes
        coo = sps.coo_matrix(
            (data_icm_type["data"], (item_index.values, features_index.values)), shape=shape)
        csr = coo.tocsr()
        return csr

    def load_augmented_binary_urm_less_items(self):
        """Load urm in which pairs (user,item) are '1' iff user has either watched item or opened item's details page

        Returns:
            csr: urm as csr object
        """
        urm = self.load_augmented_binary_urm_less_items_df()
        return self.dataframe_to_csr(urm,'UserID','ItemID','Data')

    def load_augmented_binary_urm(self):
        """Load urm in which pairs (user,item) are '1' iff user has either watched item or opened item's details page

        Returns:
            csr: urm as csr object
        """
        urm = self.load_augmented_binary_urm_df()
        return self.dataframe_to_csr(urm,'UserID','ItemID','Data')

    def load_weighted_urm(self):
        """
        Load urm in which pairs (user,item) are non-binary values (0<=data<=1)
        value = number_of_episodes_of_item_watched_by_user / total_number_of_episodes_of_item

        Returns:
            csr: urm as csr object
        """
        interactions_and_impressions = pd.read_csv(filepath_or_buffer=os.getenv('INTERACTIONS_AND_IMPRESSIONS_PATH'),
                                                   sep=',',
                                                   names=[
            'UserID', 'ItemID', 'Impressions', 'Data'],
            header=0,
            dtype={'UserID': np.int32, 'ItemID': np.int32, 'Impressions': np.object0, 'Data': np.int32})
        df = interactions_and_impressions.drop(['Impressions'], axis=1)
        # for each pair (user,item), count the number interactions with data set to '0'
        # filter out rows with data set to '1'
        df = df[df.Data != 1]
        # groupby UserID and ItemID keeping the columns index
        df = df.groupby(['UserID', 'ItemID'], as_index=False)
        # count occurrences of pairs (user,item)
        df_number_of_watched_episodes = df['Data'].count()

        data_ICM_length = pd.read_csv(filepath_or_buffer=os.getenv('ICM_LENGTH_PATH'),
                                      sep=',',
                                      names=[
            'ItemID', 'FeatureID', 'Data'],
            header=0,
            dtype={'item_id': np.int32, 'feature_id': np.int32, 'data': np.int32})
        # drop feature_id column because it is always '0'
        data_ICM_length = data_ICM_length.drop(['FeatureID'], axis=1)
        # calculate average of number of episodes in order to assign it to items without episodes information
        average_number_of_episodes = data_ICM_length['Data'].mean()
        # create personalized urm
        # join df_number_of_watched_episodes with data_ICM_length on ItemID and fill NaN values with the average number of episodes
        df = df_number_of_watched_episodes.merge(data_ICM_length, on='ItemID', how='left').fillna({'Data_y': average_number_of_episodes})
        df = df.rename({'Data_x': 'NumWatchedEpisodes', 'Data_y': 'TotNumEpisodes'}, axis=1)
        df['A'] = df['NumWatchedEpisodes']/df['TotNumEpisodes']

        # produce urm
        df = df.drop(['NumWatchedEpisodes', 'TotNumEpisodes'], axis=1)
        urm = df.rename({'A': 'Data'}, axis=1)
        return self.dataframe_to_csr(urm,'UserID','ItemID','Data')

    def load_target(self):
        """Load target that is the set of users to which recommend items

        Returns:
            numpy.array: array containing unique UserIDs
        """
        df_original = pd.read_csv(filepath_or_buffer=os.getenv('TARGET_PATH'),
                                  sep=',',
                                  header=0,
                                  dtype={'user_id': np.int32})
        df_original.columns = ['user_id']
        user_id_list = df_original['user_id'].values
        user_id_unique = np.unique(user_id_list)
        print(">>> number of target users: {}".format(len(user_id_list)))
        return user_id_unique

    def load_target_df(self):
        """Load target that is the set of users to which recommend items

        Returns:
            dataframe: UserIDs as dataframe object
        """
        target = pd.read_csv(filepath_or_buffer=os.getenv('TARGET_PATH'),
                             sep=',',
                             header=0,
                             dtype={'user_id': np.int32})
        return target

    def load_icm(self):
        """Load icm

        Returns:
            csr: icm as csr object
        """
        data_icm_type = pd.read_csv(filepath_or_buffer=os.getenv('DATA_ICM_TYPE_PATH'),
                                    sep=',',
                                    names=[
            'item_id', 'feature_id', 'data'],
            header=0,
            dtype={'item_id': np.int32, 'feature_id': np.int32, 'data': np.int32})

        features = data_icm_type['feature_id'].unique()
        items = data_icm_type['item_id'].unique()
        shape = (len(items), len(features))


        # Create indices for users and items
        features_cat = CategoricalDtype(categories=sorted(features), ordered=True)
        item_cat = CategoricalDtype(categories=sorted(items), ordered=True)
        features_index = data_icm_type["feature_id"].astype(features_cat).cat.codes
        item_index = data_icm_type["item_id"].astype(item_cat).cat.codes
        coo = sps.coo_matrix(
            (data_icm_type["data"], (item_index.values, features_index.values)), shape=shape)
        csr = coo.tocsr()
        return csr

    def load_icm_df(self):
        """Load icm

        Returns:
            dataframe: icm as dataframe object
        """
        data_icm_type = pd.read_csv(filepath_or_buffer=os.getenv('DATA_ICM_TYPE_PATH'),
                                    sep=',',
                                    names=[
            'item_id', 'feature_id', 'data'],
            header=0,
            dtype={'item_id': np.int32, 'feature_id': np.int32, 'data': np.int32})

        return data_icm_type

    def load_powerful_binary_urm_df(self,mult_param_urm=0.825,mult_param_icm=0.175):
        """
        Load urm by stacking augmented urm and transposed icm.
        This is a smart technique to implement SLIM with side information (or S-SLIM).


        Returns:
            dataframe: urm as dataframe object
        """
        data_icm_type = pd.read_csv(filepath_or_buffer=os.getenv('DATA_ICM_TYPE_PATH'),
                                    sep=',',
                                    names=[
            'item_id', 'feature_id', 'data'],
            header=0,
            dtype={'item_id': np.int32, 'feature_id': np.int32, 'data': np.int32})
        # Swap the columns from (item_id, feature_id, data) to (feature_id, item_id, data)
        swap_list = ["feature_id", "item_id", "data"]
        f = data_icm_type.reindex(columns=swap_list)
        f = f.rename(
            {'feature_id': 'UserID', 'item_id': 'ItemID', 'data': 'Data'}, axis=1)

        urm = self.load_augmented_binary_urm_df()

        # urm times alpha
        urm['Data'] = mult_param_urm * urm['Data']
        # f times (1-aplha)
        f['Data'] = mult_param_icm * f['Data']
        # Change UserIDs of f matrix in order to make recommender work
        f['UserID'] = 41634 + f['UserID']

        powerful_urm = pd.concat(
            [urm, f], ignore_index=True).sort_values(['UserID', 'ItemID'])
        return powerful_urm

    def load_powerful_binary_urm(self,mult_param_urm=0.825,mult_param_icm=0.175):
        """
        Load urm by stacking augmented urm and transposed icm.
        This is a smart technique to implement SLIM with side information (or S-SLIM).


        Returns:
            csr: urm as csr object
        """
        powerful_urm = self.load_powerful_binary_urm_df(mult_param_urm=mult_param_urm,mult_param_icm=mult_param_icm)
        return self.dataframe_to_csr(powerful_urm,'UserID','ItemID','Data')


    def load_powerful_binary_urm_df_given_URM_train_df(self,URM_train_df): #NEW
        data_icm_type = pd.read_csv(filepath_or_buffer=os.getenv('DATA_ICM_TYPE_PATH'),
                                    sep=',',
                                    names=[
            'item_id', 'feature_id', 'data'],
            header=0,
            dtype={'item_id': np.int32, 'feature_id': np.int32, 'data': np.int32})
        # Swap the columns from (item_id, feature_id, data) to (feature_id, item_id, data)
        swap_list = ["feature_id", "item_id", "data"]
        f = data_icm_type.reindex(columns=swap_list)
        f = f.rename(
            {'feature_id': 'UserID', 'item_id': 'ItemID', 'data': 'Data'}, axis=1)

        urm = URM_train_df

        # urm times alpha
        urm['Data'] = 0.825 * urm['Data']
        # f times (1-aplha)
        f['Data'] = 0.175 * f['Data']
        # Change UserIDs of f matrix in order to make recommender work
        f['UserID'] = 41634 + f['UserID']

        powerful_urm = pd.concat(
            [urm, f], ignore_index=True).sort_values(['UserID', 'ItemID'])
        return powerful_urm
    
    def load_powerful_binary_urm_given_URM_train_df(self,URM_train_df): #NEW
        powerful_urm = self.load_powerful_binary_urm_df_given_URM_train_df(URM_train_df)
        return self.dataframe_to_csr(powerful_urm,'UserID','ItemID','Data')


    def get_unique_items_based_on_urm(self, urm):
        """Returns numpy.array of unique items contained in the given urm

        Args:
            urm (dataframe): urm should necessarily be a dataframe object

        Returns:
            numpy.array: _description_
        """
        item_id_list = urm['ItemID'].values
        items = np.unique(item_id_list)
        return items

    def get_impressions_count(self, target, items):
        """
        Return a dictionary of dictionaries. For each UserID there is a dictionary of ItemIDs as keys and a number, corresponding to 
        how many times that ItemID has been presented to the given UserID, as values.

        Args:
            target (int): UserIDs on which count ItemIDs presentations occurences
            items (numpy.array): ItemIDs on which count presentations occurences
        Returns:
            dict: dictionary of dictionaries, for instance { user0:{item0:2, item1:23}, user1:{item2:11, item4:3} }
        """

        df = pd.read_csv(filepath_or_buffer=os.getenv('INTERACTIONS_AND_IMPRESSIONS_PATH'),
                         sep=',',
                         names=[
            'UserID', 'ItemID', 'Impressions', 'Data'],
            header=0,
            dtype={'UserID': np.int32, 'ItemID': np.int32, 'Impressions': np.object0, 'Data': np.int32})
        df = df.drop(['ItemID'], axis=1)
        df = df.drop(['Data'], axis=1)
        df = df.dropna()
        # add a comma at the end of each impression string in order to concat properly then
        df['Impressions'] = df['Impressions'].apply(lambda x: str(x)+',')
        df = df.groupby(['UserID'], as_index=False)
        # to concat impressions of each user
        impressions_per_user = df['Impressions'].apply(sum)

        presentations_per_user={}
        for user in target:
            impressions = impressions_per_user[impressions_per_user['UserID']
                                            == user]['Impressions']
            impressions = impressions.iloc[0].split(",")
            # remove last element which is a '' due to last ','
            impressions = np.delete(impressions, -1)

            counts = Counter(impressions)
            presentations = {}
            for item in items:
                presentations[item] = counts[item]
            presentations_per_user[user]=presentations
        return presentations_per_user


    def stackMatrixes(self, URM_train):
        # Vertical stack so ItemIDs cardinality must coincide.
       
        urm=self.csr_to_dataframe(URM_train)
        f=self.load_icm_df()
        swap_list = ["feature_id", "item_id", "data"]
        f = f.reindex(columns=swap_list)
        f = f.rename({'feature_id': 'UserID', 'item_id': 'ItemID', 'data': 'Data'}, axis=1)

        urm['Data'] = 0.825 * urm['Data']
        # f times (1-aplha)
        f['Data'] = 0.175 * f['Data']
        # Change UserIDs of f matrix in order to make recommender work
        f['UserID'] = 41634 + f['UserID']

        powerful_urm = pd.concat([urm, f], ignore_index=True).sort_values(['UserID', 'ItemID'])
        return self.dataframe_to_csr(powerful_urm,'UserID', 'ItemID','Data')

    
    def print_statistics(self):
        """ Print statistics about dataset """
        target = self.load_target()
        interactions_and_impressions = pd.read_csv(filepath_or_buffer=os.getenv('INTERACTIONS_AND_IMPRESSIONS_PATH'),
                                                   sep=',',
                                                   names=[
                                                       'UserID', 'ItemID', 'Impressions', 'Data'],
                                                   header=0,
                                                   dtype={'UserID': np.int32, 'ItemID': np.int32, 'Impressions': np.object0, 'Data': np.int32})
        urm = interactions_and_impressions.drop(['Impressions'], axis=1)
        # removing duplicated (user_id,item_id) pairs
        urm = urm.drop_duplicates(keep='first')
        print(">>> number of users in interactions_and_impressions: {}".format(
            len(urm['UserID'].unique())))
        print(">>> number of unique users in target that are not in interactions_and_impressions: {}".format(
            len(np.setdiff1d(target, urm['UserID'].unique()))))
        print(">>> number of unique users in interactions_and_impressions that are not in target: {}".format(
            len(np.setdiff1d(urm['UserID'].unique(), target))))

        # removing (user_id,item_id) pairs with data set to 1
        watchers_urm = urm[urm.Data != 1]
        print('>>> number of unique users in target that are not in "list of users that have watched at least a movie": {}'.format(
            len(np.setdiff1d(target, watchers_urm['UserID'].unique()))))
        print('>>> number of unique users in "list of users that have watched at least a movie" that are not in target: {}'.format(
            len(np.setdiff1d(watchers_urm['UserID'].unique(), target))))
        print('>>> number of unique users in interactions_and_impressions that are not in "list of users that have watched at least a movie": {}'.format(
            len(np.setdiff1d(urm['UserID'].unique(), watchers_urm['UserID'].unique()))))




    def paddingICMandURM(self, URM_train):

        """
        Scrivo in italiano per farti capire bene quando lo leggi
        ho preso la differenza tra URM e ICM in modo da avere gli item che mancano all'interno della ICM,
        tramite un for li ho aggiunti al termine del dataframe della ICM assegnando la feature 1 e valore 0 -> ho fatto questo perche se mettevo feature 0, mi avrebbe aggiunto una feature in più e diventavano 6 anzichè 5 DA VEDERE
        per la urm ho fatto lo stesso
        alla fine sia per urm che per icm le ho riordinate e convertite in csr -> hanno tute e due stesso numero di items 27968
        Args:
            target (int): UserIDs on which count ItemIDs presentations occurences
            items (numpy.array): ItemIDs on which count presentations occurences
        Returns:
            dict: dictionary of dictionaries, for instance { user0:{item0:2, item1:23}, user1:{item2:11, item4:3} }
        """
        urm=self.csr_to_dataframe(URM_train)
        icm=self.load_icm_df()
        DiffURM_ICM = np.setdiff1d(urm['ItemID'].unique(), icm['item_id'].unique())
        DiffICM_URM = np.setdiff1d(icm['item_id'].unique(), urm['ItemID'].unique())
        print(DiffURM_ICM.size)
        for id in DiffURM_ICM:
            icm.loc[len(icm.index)] = [id, 1, 0]
        sorted_icm = icm.sort_values('item_id').reset_index(drop= True)
        for id in DiffICM_URM:
            urm.loc[len(urm.index)] = [1, id, 0]
        sorted_urm = urm.sort_values('UserID').reset_index(drop= True)
        URM = self.dataframe_to_csr(sorted_urm, 'UserID', 'ItemID', 'Data')
        ICM = self.dataframe_to_csr(sorted_icm, 'item_id', 'feature_id', 'data')
        return URM, ICM