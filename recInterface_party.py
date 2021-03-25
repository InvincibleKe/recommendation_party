# Recommendation Interface

import torch
from torch.utils.data import DataLoader
from dataset import PartyDataset
from model_CNN import rec_model
import numpy as np
import pickle as pkl
from dbConnection import Mongo
HEAD_PATH = '/Users/kejianrong/英迈/作局/recommendation_party/'
FEATURE_DATA_PATH = 'Params/feature_data.pkl'
DATA_PATH = 'Params/user_party_dict.pkl'
PATH = 'Params/model_params.txt'
# load the current trained model
model = rec_model()
model.load_state_dict(torch.load(HEAD_PATH + PATH, map_location=torch.device('cpu')))
def readPkl(file_path):
    '''
    read pkl file until all the data in the file has been loaded.
    :param file_path: the pkl file path
    :return: the data in the pkl file
    '''
    data = {}
    f = open(file_path, 'rb')
    with open(file_path, 'rb') as f:
        while True:
            try:
                data.update(f)
            except:
                break
    return data
def savePartyAndUserFeature(model):
    '''
    Save party and User feature and data into files
    :param model: the curret trained model
    :return: no return but pkl files stored
    '''
    batch_size = 256
    datasets = PartyDataset()
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=False, num_workers=4)

    # format: {id(int) : feature(numpy array)}
    user_feature_dict = {}
    party_feature_dict = {}
    parties = {}
    users = {}
    with torch.no_grad():
        for i_batch, sample_batch in enumerate(dataloader):
            user_inputs = sample_batch['user_inputs']
            party_inputs = sample_batch['party_inputs']

            # B x 1 x 200 = 256 x 1 x 200
            _, feature_user, feature_party = model(user_inputs, party_inputs)

            # B x 1 x 200 = 256 x 1 x 200
            feature_user = feature_user.cpu().numpy()
            feature_party = feature_party.cpu().numpy()

            for i in range(user_inputs['uid'].shape[0]):
                uid = user_inputs['uid'][i]   # uid
                gender = user_inputs['gender'][i]
                birthyear = user_inputs['birthyear'][i]
                constellation = user_inputs['constellation'][i]
                character_tag = user_inputs['character_tag'][i]
                user_tag = user_inputs['user_tag'][i]
                theme_tag =user_inputs['theme_tag'][i]

                pid = party_inputs['pid'][i]   # mid
                title = party_inputs['title'][i]
                activity_type = party_inputs['activity_type'][i]
                activity_theme_tag = party_inputs['activity_theme_tag'][i]
                # activity_tag = party_inputs['activity_tag'][i]
                people_max = party_inputs['people_max'][i]
                people_min = party_inputs['people_min'][i]
                sex_limit_type = party_inputs['sex_limit_type'][i]
                longitude = party_inputs['longitude'][i]
                latitude = party_inputs['latitude'][i]
                pre_amt = party_inputs['pre_amt'][i]
                # ave_price = party_inputs['ave_price'][i]
                # total_amt = party_inputs['total_amt'][i]
                price_type = party_inputs['price_type'][i]

                if uid.item() not in users.keys():
                    users[uid.item()]={'uid':uid,'gender':gender,'constellation':constellation, 'birthyear':birthyear,
                                       'character_tag':character_tag, 'user_tag':user_tag, 'theme_tag':theme_tag}
                if pid.item() not in parties.keys():
                    parties[pid.item()]={'pid':pid,'title':title, 'activity_type':activity_type, 'activity_theme_tag':activity_theme_tag, 'activity_tag':activity_tag,
                                         'people_max':people_max, 'people_min':people_min, 'sex_limit_type':sex_limit_type, 'longitude':longitude,
                                         'latitude':latitude, 'pre_amt':pre_amt, 'price_type':price_type}

                if uid.item() not in user_feature_dict.keys():
                    user_feature_dict[uid.item()] = feature_user[i]
                if pid.item() not in party_feature_dict.keys():
                    party_feature_dict[pid.item()] = feature_party[i]

            print('Solved: {} samples'.format((i_batch+1)*batch_size))
    feature_data = {'feature_user': user_feature_dict, 'feature_party':party_feature_dict}
    dict_user_party={'user': users, 'party':parties}
    pkl.dump(feature_data, open(HEAD_PATH + FEATURE_DATA_PATH,'wb'))
    pkl.dump(dict_user_party, open(HEAD_PATH + DATA_PATH,'wb'))
    print(len(dict_user_party['user']))
    print(len(feature_data['feature_party']))

def getKNNitem(itemID,itemName='party',K=10):
    '''
    Use KNN at feature data to get K neighbors
    Args:
        itemID: target item's id
        itemName: 'party' or 'user'
        K: K-neighbors
    return:
        a list of item ids of which close to itemID
    '''
    assert K>=1, 'Expect K bigger than 0 but get K<1'
    # get cosine similarity between vec1 and vec2
    def getCosineSimilarity(vec1, vec2):
        cosine_sim = float(vec1.dot(vec2.T).item()) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return cosine_sim

    feature_data = pkl.load(open(HEAD_PATH + FEATURE_DATA_PATH, 'rb'))
    feature_items = feature_data['feature_'+itemName]
    if itemID not in feature_items.keys():
        savePartyAndUserFeature(model)
        feature_data = pkl.load(open(HEAD_PATH + FEATURE_DATA_PATH, 'rb'))
        feature_items = feature_data['feature_' + itemName]
    assert itemID in feature_items.keys(), 'Expect item ID exists in dataset, but get None.'
    feature_current = feature_items[itemID]
    id_sim = [(item_id, getCosineSimilarity(feature_current,vec2)) for item_id, vec2 in feature_items.items()]
    id_sim = sorted(id_sim, key=lambda x:x[1], reverse=True)
    return [id_sim[i][0] for i in range(K+1)][1:]

def getUserMostLike(uid, K=10):
    '''
    Get user(uid) mostly like party
    feature_user * feature_party
    Args:
        model: net model
        uid: target user's id
    return:
        the biggest rank party id
    '''
    feature_data = pkl.load(open(HEAD_PATH + FEATURE_DATA_PATH, 'rb'))
    user_party_dict = pkl.load(open(HEAD_PATH + DATA_PATH,'rb'))
    '''
    feature_data = readPkl(HEAD_PATH + FEATURE_DATA_PATH)
    user_party_dict = readPkl(HEAD_PATH + DATA_PATH)
    '''
    # simple version which updates the whole data but may spend more time to run
    if uid not in user_party_dict['user']:
        savePartyAndUserFeature(model)
    '''
    # update version which updates the incremental data and may spend less time to run
    if uid not in user_party_dict['user']:
        oldFeature_dict = readPkl(HEAD_PATH + FEATURE_DATA_PATH)
        mongo = Mongo()
        myquery = {'_id': uid}
        new_data = mongo.queryLimited('user_info', myquery)
        new_feature = oldFeature_dict.update(new_data)
        pkl.dump(new_feature, open(HEAD_PATH+FEATURE_DATA_PATH, 'wb'))
        feature_data = readPkl(HEAD_PATH + FEATURE_DATA_PATH)
        user_party_dict = readPkl(HEAD_PATH + DATA_PATH)
    '''
    assert uid in user_party_dict['user'], \
        'Expect user whose id is uid exists, but get None'
    feature_user = feature_data['feature_user'][uid]
    party_dict = user_party_dict['party']
    pid_rank = {}
    for pid in party_dict.keys():
        feature_party=feature_data['feature_party'][pid]
        rank = np.dot(feature_user, feature_party.T)
        if pid not in pid_rank:
            pid_rank[pid]=rank.item()

    pid_rank = [(pid, rank) for pid, rank in pid_rank.items()]
    pids = [pid[0] for pid in sorted(pid_rank, key=lambda x: x[1], reverse=True)]
    return pids[0:K]
def getPartyCanRecommend(pid, K=10):
    '''
    get users who may like the party with the pid
    :param pid: party id
    :param K: the certain number of users
    :return: user ids
    '''
    feature_data = pkl.load(open(HEAD_PATH + FEATURE_DATA_PATH, 'rb'))
    user_party_dict = pkl.load(open(HEAD_PATH + DATA_PATH, 'rb'))
    '''
    feature_data = readPkl(HEAD_PATH + FEATURE_DATA_PATH)
    user_party_dict = readPkl(HEAD_PATH + DATA_PATH)
    '''
    # simple version which updates the whole data but may spend more time to run
    if pid not in user_party_dict['party']:
        savePartyAndUserFeature(model)
    '''
    # update version which updates the incremental data and may spend less time to run
    if pid not in user_party_dict['party']:
        oldFeature_dict = readPkl(HEAD_PATH + FEATURE_DATA_PATH)
        mongo = Mongo()
        myquery = {"_id": pid}
        new_data = mongo.queryLimited('activity', myquery)
        new_feature = oldFeature_dict.update(new_data)
        pkl.dump(new_feature, open(HEAD_PATH + FEATURE_DATA_PATH, 'wb'))
        feature_data = readPkl(HEAD_PATH + FEATURE_DATA_PATH)
        user_party_dict = readPkl(HEAD_PATH + DATA_PATH)
    '''
    assert pid in user_party_dict['party'], \
        'Expect party whose id is pid exists, but get None'
    feature_party = feature_data['feature_party'][pid]
    user_dict = user_party_dict['user']
    uid_rank = {}
    for uid in user_dict.keys():
        feature_user = feature_data['feature_user'][uid]
        rank = np.dot(feature_party, feature_user.T)
        if uid not in uid_rank:
            uid_rank[uid] = rank.item()
    uid_rank = [(uid, rank) for uid, rank in uid_rank.items()]
    uids = [uid[0] for uid in sorted(uid_rank, key=lambda x:x[1], reverse=True)]
    return uids[0:K]

def userRecentSearch(uid, K=10):

    return 0
if __name__=='__main__':
    b = {}
    a = {3: 'aaa', 4: 'aaaa'}
    b.update(a)
    print(a)