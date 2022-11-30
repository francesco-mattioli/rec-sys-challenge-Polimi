from Data_Handler.DataReader import DataReader
import numpy as np


class Impressions(object):

    def __init__(self,target,all_items):
        dataReader=DataReader()
        self.presentations_per_user=dataReader.get_impressions_count(target,all_items)



    def update_ranking(self,user_id,recommended_items):
        """Change the order of recommendations in order to comply with impressions information.
        Return an array containing sorted given recommended_items based on number of occurences in impressions list.
        For example, if item 'x' has occured 2 times in impressions lists of user user_id while item 'y' has occured 3 times, 
        then 'x' will be ranked higher than 'y'.


        Args:
            user_id (int): UserID
            items (numpy.array): ItemIDs of corresponding urm
            recommended_items (list): array of recommendations i.e. ouput of a recommendation model

        Returns:
            list: sorted array of recommendations
        """
        presentations = self.presentations_per_user[user_id]
        sorted_presentations = sorted(presentations.items(), key=lambda x:x[1], reverse=True) # discending order
        sorted_presentations = dict(sorted_presentations)

        new_ranking= sorted(recommended_items, key=lambda x:sorted_presentations[x], reverse=False) # ascending order
        return new_ranking