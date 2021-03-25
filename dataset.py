from torch.utils.data import Dataset
from dbConnection import Mongo
import pickle as pkl
import torch
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import jieba.posseg as pseg
import operator
import string
from functools import reduce
USER_PATH = 'Data/user_info.csv'
ACTIVITY_PATH = 'Data/activity.csv'
RANK_PATH = 'Data/rank.csv'
TEXTMODEL_PATH = 'Params/word2vec/word2vec_from_weixin/word2vec/word2vec_wx'
# load a word-to-vector model
text_model = Word2Vec.load(TEXTMODEL_PATH)
def get_one_hot(label, N):
    '''
    convert label data to one-hot vector
    :param label: label data with int data type
    :param N: the number of label types
    :return: one-hot vector
    '''
    size = list(label.size())
    label = label.view(-1)
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)
    size.append(N)
    return ones.view(*size)
def constellation2int(constellation):
    '''
    convert constellation string to an int data type
    :param constellation: a string descrinbing constellation
    :return: an int data type which represents the constellation
    '''
    if constellation in "白羊座":
        return 0
    if constellation in "金牛座":
        return 1
    if constellation in "双子座":
        return 2
    if constellation in "巨蟹座":
        return 3
    if constellation in "狮子座":
        return 4
    if constellation in "处女座":
        return 5
    if constellation in "天秤座":
        return 6
    if constellation in "天蝎座":
        return 7
    if constellation in "射手座":
        return 8
    if constellation in "魔羯座":
        return 9
    if constellation in '摩羯座':
        return 9
    if constellation in "水瓶座":
        return 10
    if constellation in "双鱼座":
        return 11
    else:
        print("12")
def text2vec(text, model):
    '''
    convert a text to a vector
    :param text: a text string
    :param model: a trained text-to-vector model
    :return:
    '''
    text_jieba = pseg.cut(text)
    vector = []
    for word in text_jieba:
        w = str(word.word)
        try:
            v = model[w]
            vector.append(v.tolist())
        except:
            pass
    # print(vector)
    return np.mean(np.array(vector), axis=0)
def read_csv(PATH):
    '''
    read CSV file data
    :param PATH: the csv file path
    :return: data read from file
    '''
    with open(PATH, encoding="utf8", errors='ignore') as f:
        data = f.read().splitlines()
    results = []
    for row in data:
        results.append(row.split('\t'))
    results = results[1:]
    results = np.array(results)
    return results
def read_muldata(original_data):
    '''
    read string data and convert it into a list
    :param original_data: data with string type
    :return: data with list type
    '''
    data = original_data.split(',')
    # data = np.array([i.split(',') for i in original_data])
    # print(data)
    data_int = []
    for i in data:
        data_int.append(int(i))
    return data_int
def read_objectData(original_data):
    '''
    convert an objectId list to an int list
    :param original_data: data with object type
    :return: data with list type
    '''
    data_int = []
    for i in original_data:
        data_int.append(int(int(str(i), 16)/100000000000000000000000))
    data_int = reduce(operator.add, data_int)
    return [data_int]
def searchDataset(item, dataset):
    '''
    judge if data is in a dataset
    :param item: data item
    :param dataset: target dataset
    :return: judging result
    '''
    for one in dataset:
        if item == one[0]:
            return True, one
    return False, one
def dataProcess():
    '''
    collect data from mongo db and then clear data and convert them to lists
    :return: data with list type
    '''
    # read target data from a mongo database
    mongo = Mongo()
    myquery = {"data_complete": 1}
    user_raw = mongo.queryLimited('user_info', myquery)
    activity_raw = mongo.query('activity')
    activity_collect_raw = mongo.query('activity_collect')
    activity_part_raw = mongo.query('activity_part')
    part_rank = [5]
    collect_rank = [3]
    click_rank = [1]
    users = []
    activities = []
    ranks = []
    users_rank = []
    activities_rank = []
    # process raw user data into data with int or int list types
    for one in user_raw:
        row = []
        user_id = int(str(one['_id']), 16)
        row.append(user_id)
        sex = one['sex']
        row.append(sex)
        birthyear = int(one['birthday'].year)
        row.append(birthyear)
        constellation = constellation2int(str(one['constellation']))
        row.append(constellation)
        character_tag = read_objectData(one['extra']['character_tag'])
        row.append(character_tag)
        user_tag = read_objectData(one['extra']['user_tag'])
        row.append(user_tag)
        theme_tag = read_objectData(one['extra']['activity_theme_tag'])
        row.append(theme_tag)
        # row = reduce(operator.add, row)
        users.append(row)
    # process raw activity data into data with int or int list types
    for one in activity_raw:
        row = []
        party_id = int(str(one['_id']), 16)
        row.append(party_id)
        title = one['title']
        row.append(title)
        activity_type = int(one['activity_type'])
        row.append(activity_type)
        activityThemeTag = read_objectData(one['activityThemeTag'])
        row.append(activityThemeTag)
        # activityTag = read_objectData(one['activityTag'])
        # row.append(activityTag)
        people_max = one['recruit_num_limit']['max']
        row.append(people_max)
        people_min = one['recruit_num_limit']['min']
        row.append(people_min)
        sex_limit_type = int(one['sex_limit']['type'])
        row.append(sex_limit_type)
        longitude = one['longitude']
        row.append(longitude)
        latitude = one['latitude']
        row.append(latitude)
        pre_amt = one['pre_amt']
        row.append(pre_amt)
        # ave_price = one['ave_price']
        # row.append(ave_price)
        # total_amt = one['total_amt']
        # row.append(total_amt)
        price_type = one['price_type']
        row.append(price_type)
        # row = reduce(operator.add, row)
        activities.append(row)
    # process raw activity_part data into int or int list types
    for one in activity_part_raw:
        user_id = int(str(one['user']), 16)
        party_id = int(str(one['activity']), 16)
        userinusers, user = searchDataset(user_id, users)
        partyinparties, party = searchDataset(party_id, activities)
        if userinusers & partyinparties:
            users_rank.append(user)
            activities_rank.append(party)
            ranks.append(part_rank)
    # process raw activity_collect data into data with int or int list type
    for one in activity_collect_raw:
        user_id = int(str(one['user']), 16)
        party_id = int(str(one['activity']), 16)
        userinusers, user = searchDataset(user_id, users)
        partyinparties, party = searchDataset(party_id, activities)
        if userinusers & partyinparties:
            users_rank.append(user)
            activities_rank.append(party)
            ranks.append(collect_rank)
    return np.array(users_rank), np.array(activities_rank), np.array(ranks)
class PartyDataset():
    '''
    class PartyDataset represents a dataset with user, activity and relation data which can be used as a training dataset
    '''
    def __init__(self):
        users, activities, ranks = dataProcess()
        df = np.concatenate((users,activities,ranks), axis=1)
        self.dataFrame = pd.DataFrame(df)
        '''
        users = read_csv(USER_PATH)[:5000]
        parties = read_csv(ACTIVITY_PATH)[:5000]
        rank = read_csv(RANK_PATH)[:5000]
        df = np.concatenate((users, parties, rank), axis=1)
        self.dataFrame = pd.DataFrame(df)
        '''
    def __len__(self):
    # get the length of the dataset
        return len(self.dataFrame)

    def __getitem__(self, idx):
    # get an item data from the dataset according to the item index
        # user data
        uid = self.dataFrame.iloc[idx, 0]
        gender = self.dataFrame.iloc[idx, 1]
        birthyear = self.dataFrame.iloc[idx, 2]
        constellation = self.dataFrame.iloc[idx, 3]
        character_tag = self.dataFrame.iloc[idx, 4]
        user_tag = self.dataFrame.iloc[idx, 5]
        theme_tag = self.dataFrame.iloc[idx, 6]

        # party data
        pid = self.dataFrame.iloc[idx, 7]
        title = text2vec(self.dataFrame.iloc[idx, 8], text_model)
        activity_type = self.dataFrame.iloc[idx, 9]
        activity_theme_tag = self.dataFrame.iloc[idx, 10]
        # activity_tag = read_muldata(self.dataFrame.iloc[idx, 11])
        people_max = self.dataFrame.iloc[idx, 11]
        people_min = self.dataFrame.iloc[idx, 12]
        sex_limit_type = self.dataFrame.iloc[idx, 13]
        longitude = self.dataFrame.iloc[idx, 14]
        latitude = self.dataFrame.iloc[idx, 15]
        pre_amt = self.dataFrame.iloc[idx, 16]
        # ave_price = self.dataFrame.iloc[idx, 18]
        # total_amt = self.dataFrame.iloc[idx, 19]
        price_type = self.dataFrame.iloc[idx, 17]

        # target
        rank = torch.FloatTensor([self.dataFrame.iloc[idx, 18]])

        user_inputs = {
            'uid': torch.LongTensor([int(uid/100000000000000000000000000000)]).view(1, -1),
            'gender': torch.LongTensor([int(gender)]).view(1, -1),
            'birthyear': torch.LongTensor([int(birthyear)]).view(1, -1),
            'constellation': get_one_hot(torch.LongTensor([constellation]), 12).view(1, -1),
            'character_tag': torch.LongTensor(character_tag).view(1, -1),
            'user_tag': torch.LongTensor(user_tag).view(1, -1),
            'theme_tag': torch.LongTensor(theme_tag).view(1, -1)
        }
        party_inputs = {
            'pid': torch.LongTensor([int(pid/100000000000000000000000000000)]).view(1, -1),
            'title': torch.FloatTensor(title).view(1, 1, -1),
            'activity_type': torch.LongTensor(activity_type).view(1, -1),
            'activity_theme_tag': torch.LongTensor(activity_theme_tag).view(1, -1),
            # 'activity_tag': torch.LongTensor(activity_tag).view(1, -1),
            'people_max': torch.LongTensor([int(people_max)]).view(1, -1),
            'people_min': torch.LongTensor([int(people_min)]).view(1, -1),
            'sex_limit_type': torch.LongTensor([int(sex_limit_type)]).view(1, -1),
            'longitude': torch.LongTensor([int(longitude)]).view(1, -1),
            'latitude': torch.LongTensor([int(latitude)]).view(1, -1),
            'pre_amt': torch.LongTensor([int(pre_amt)]).view(1, -1),
            # 'ave_price': torch.LongTensor([int(ave_price)]).view(1, -1),
            # 'total_amt': torch.LongTensor([int(total_amt)]).view(1, -1),
            'price_type': torch.LongTensor([int(price_type)]).view(1, -1)
        }
        sample = {
            'user_inputs': user_inputs,
            'party_inputs': party_inputs,
            'rank': rank
        }
        return sample

if __name__ == '__main__':
    print(get_one_hot(torch.LongTensor([2]),12))

class MovieRankDataset(Dataset):
    def __init__(self, pkl_file, drop_dup=False):
        df = pkl.load(open(pkl_file,'rb'))
        if drop_dup == True:
            df_user = df.drop_duplicates(['user_id'])
            df_movie = df.drop_duplicates(['movie_id'])
            self.dataFrame = pd.concat((df_user, df_movie), axis=0)
        else:
            self.dataFrame = df

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, idx):
        # user data
        uid = self.dataFrame.iloc[idx,0]
        gender = self.dataFrame.iloc[idx,3]
        age = self.dataFrame.iloc[idx,4]
        job = self.dataFrame.iloc[idx,5]

        # movie data
        mid = self.dataFrame.iloc[idx,1]
        mtype=self.dataFrame.iloc[idx,7]
        mtext=self.dataFrame.iloc[idx,6]

        # target
        rank = torch.FloatTensor([self.dataFrame.iloc[idx,2]])
        user_inputs = {
            'uid': torch.LongTensor([uid]).view(1,-1),
            'gender': torch.LongTensor([gender]).view(1,-1),
            'age': torch.LongTensor([age]).view(1,-1),
            'job': torch.LongTensor([job]).view(1,-1)
        }

        movie_inputs = {
            'mid': torch.LongTensor([mid]).view(1,-1),
            'mtype': torch.LongTensor(mtype),
            'mtext': torch.LongTensor(mtext)
        }

        sample = {
            'user_inputs': user_inputs,
            'movie_inputs': movie_inputs,
            'target': rank
        }
        return sample