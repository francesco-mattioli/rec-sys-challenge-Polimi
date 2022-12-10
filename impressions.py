from Data_Handler.DataReader import DataReader
import numpy as np


class Impressions(object):

    def __init__(self,target,all_items):
        """_summary_

        Args:
            target (_type_): _description_
            all_items (_type_): ItemIDs of corresponding urm
        """
        #dataReader=DataReader()
        #self.presentations_per_user=dataReader.get_impressions_count(target,all_items)


    '''
    def update_ranking(self,user_id,recommended_items):
        """Change the order of recommendations in order to comply with impressions information.
        Return an array containing sorted given recommended_items based on number of occurences in impressions list.
        For example, if item 'x' has occured 2 times in impressions lists of user user_id while item 'y' has occured 3 times, 
        then 'x' will be ranked higher than 'y'.


        Args:
            user_id (int): UserID
            recommended_items (list): array of recommendations i.e. ouput of a recommendation model

        Returns:
            list: sorted array of recommendations
        """
        presentations = self.presentations_per_user[user_id]

        presentations_items_ids=presentations.copy().keys()

        for key in presentations_items_ids:
            if(key not in recommended_items):
                del presentations[key]
        
        sorted_presentations = sorted(presentations.items(), key=lambda x:x[1], reverse=True) # discending order
        sorted_presentations = dict(sorted_presentations)

        new_ranking= sorted(recommended_items, key=lambda x:sorted_presentations[x], reverse=False) # ascending order
        return new_ranking
    '''


    def update_ranking(self,user_id,recommended_items,dataReader):
        """Change the order of recommendations in order to comply with impressions information.
        Return an array containing sorted given recommended_items based on number of occurences in impressions list.
        For example, if item 'x' has occured 2 times in impressions lists of user user_id while item 'y' has occured 3 times, 
        then 'x' will be ranked higher than 'y'.


        Args:
            user_id (int): UserID
            recommended_items (list): array of recommendations i.e. ouput of a recommendation model

        Returns:
            list: sorted array of recommendations
        """
        presentations = dataReader.get_impressions_count_given_user(recommended_items, user_id)
        if(presentations):
            presentations_items_ids=presentations.copy().keys()

            for key in presentations_items_ids:
                if(key not in recommended_items):
                    del presentations[key]
            
            sorted_presentations = sorted(presentations.items(), key=lambda x:x[1], reverse=True) # discending order
            sorted_presentations = dict(sorted_presentations)

            new_ranking= sorted(recommended_items, key=lambda x:sorted_presentations[x], reverse=False) # ascending order
            return new_ranking
        else:
            return recommended_items