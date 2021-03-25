import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
user_max_dict={
    'uid': 1000000000,  # 20000 users
    'gender': 3,
    'constellation': 13,
    'birthyear': 3000,
    'character_tag': 100,
    'user_tag': 100,
    'theme_tag': 100,
    'country': 233,
    'province': 34,
    'city': 333,
    'address': 100,  # 100 words
    'price': 10000,
    'hobby': 50,  # 50 words
    'diet_prefer': 50,
    'memo': 1,
    'store': 1000  # each user can collect max 5 stores
}

party_max_dict={
    'pid':1000000000,  # 1000 parties
    'title':50,  # 50 words
    'activity_type': 10,
    'activity_theme_tag': 1000,
    'activity_tag': 1000,
    'people_max': 1000,
    'people_min':1000,
    'sex_limit_type': 10,
    'longitude': 360,
    'latitude': 360,
    'pre_amt': 100000,
    'total_amt': 100000,
    'ave_price': 1000000,
    'price_type': 10,
    'owner':100000,  # each party can only have 1 owner
    'planner':100000,  # each party can only have 1 planner
    'memo': 100,  # 100 words
    'adress': 100,  # 100 words
    'max_price': 10000,
    'min_price': 10000,
    'project_tagid': 1000,
    'participant_num': 100,
    'participants': 20000
    #'store': 10
}

convParams={
    'kernel_sizes': [1,1,1,1]
}
class rec_model(nn.Module):
    def __init__(self, user_max_dict=user_max_dict, party_max_dict=party_max_dict, convParams=convParams, embed_dim=32, fc_size=200):
        '''
        Args:
            user_max_dict: the max value of each user attribute. {'uid': xx, 'gender': xx, ...}
            user_embeds: size of embedding_layers.
            party_max_dict: {'pid':xx, 'title':50, ...}
            embed_dim: embedding layer sizes. normally 32 or 16
            fc_sizes: fully connect layer sizes. normally 200
        '''
        super(rec_model, self).__init__()

        # --------------------------------- user channel ----------------------------------------------------------------
        # user embeddings
        self.embedding_uid = nn.Embedding(user_max_dict['uid'], embed_dim)
        self.embedding_gender = nn.Embedding(user_max_dict['gender'], embed_dim // 2)
        self.embedding_birthyear = nn.Embedding(user_max_dict['birthyear'], embed_dim // 2)
        self.embedding_constellation = nn.Embedding(user_max_dict['constellation'], embed_dim // 2)
        self.embedding_character_tag = nn.EmbeddingBag(user_max_dict['character_tag'], embed_dim, mode='sum')
        self.embedding_user_tag = nn.EmbeddingBag(user_max_dict['user_tag'], embed_dim, mode='sum')
        self.embedding_theme_tag = nn.EmbeddingBag(user_max_dict['theme_tag'], embed_dim, mode='sum')
        '''
        # region embedding
        self.embedding_country = nn.Embedding(user_max_dict['country'], embed_dim)
        self.embedding_province = nn.Embedding(user_max_dict['province'], embed_dim // 2)
        self.embedding_city = nn.Embedding(user_max_dict['city'], embed_dim)
        '''

        # input word vector matrix
        # load text_CNN params
        kernel_sizes = convParams['kernel_sizes']
        # 8 kernel, stride=1, padding=0, kernel_sizes=[2x32, 3x32, 4x32, 5x32]
        self.Convs_text = [nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(k, embed_dim)),
            nn.ReLU()
        ).to(device) for k in kernel_sizes]
        # real-time address embedding tbc..

        # user embedding to fc: the first dense layer
        self.fc_uid = nn.Linear(embed_dim, embed_dim)
        self.fc_gender = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_birthyear = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_constellation = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_character_tag = nn.Linear(256, embed_dim)
        self.fc_user_tag = nn.Linear(256, embed_dim)
        self.fc_theme_tag = nn.Linear(256, embed_dim)
        '''
        # region embedding to fc and concat
        self.fc_country = nn.Linear(embed_dim, embed_dim)
        self.fc_province = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_city = nn.Linear(embed_dim, embed_dim)
        self.fc_region = nn.Linear(3 * embed_dim, embed_dim)
        '''

        # user channel concat
        self.fc_user_combine = nn.Linear(7 * embed_dim, fc_size)

        # --------------------------------- party channel -----------------------------------------------------------------
        # party embeddings
        self.embedding_pid = nn.Embedding(party_max_dict['pid'], embed_dim)  # normally 32
        self.embedding_title = nn.Embedding(party_max_dict['title'], embed_dim)
        self.embedding_activity_type = nn.Embedding(party_max_dict['activity_type'], embed_dim // 2)
        self.embedding_activity_theme_tag = nn.EmbeddingBag(party_max_dict['activity_theme_tag'], embed_dim, mode='sum')
        # self.embedding_activity_tag = nn.EmbeddingBag(party_max_dict['activity_tag'], embed_dim, mode='sum')
        self.embedding_people_max = nn.Embedding(party_max_dict['people_max'], embed_dim // 2)
        self.embedding_people_min = nn.Embedding(party_max_dict['people_min'], embed_dim // 2)
        self.embedding_sex_limit_type = nn.EmbeddingBag(party_max_dict['sex_limit_type'], embed_dim // 2)
        self.embedding_longitude = nn.EmbeddingBag(party_max_dict['longitude'], embed_dim // 2)
        self.embedding_latitude = nn.Embedding(party_max_dict['latitude'], embed_dim // 2)
        self.embedding_pre_amt = nn.Embedding(party_max_dict['pre_amt'], embed_dim // 2)
        # self.embedding_ave_price = nn.Embedding(party_max_dict['ave_price'], embed_dim // 2)
        # self.embedding_total_amt = nn.Embedding(party_max_dict['total_amt'], embed_dim // 2)
        self.embedding_price_type = nn.Embedding(party_max_dict['price_type'], embed_dim // 2)
        # self.embedding_store = nn.EmbeddingBag(party_max_dict['store'], embed_dim, mode='sum')

        self.fc_pid = nn.Linear(embed_dim, embed_dim)
        self.fc_title = nn.Linear(embed_dim, embed_dim)
        self.fc_activity_type = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_activity_theme_tag = nn.Linear(embed_dim, embed_dim)
        # self.fc_activity_tag = nn.Linear(embed_dim, embed_dim)
        self.fc_people_max = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_people_min = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_sex_limit_type = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_longitude = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_latitude = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_pre_amt = nn.Linear(embed_dim // 2, embed_dim)
        # self.fc_ave_price = nn.Linear(embed_dim // 2, embed_dim)
        # self.fc_total_amt = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_price_type = nn.Linear(embed_dim // 2, embed_dim)
        # self.fc_store = nn.Linear(embed_dim, embed_dim)

        # party channel concat
        self.fc_party_combine = nn.Linear(11 * embed_dim, fc_size)  # tanh

        # BatchNorm layer
        self.BN = nn.BatchNorm2d(1)

    def forward(self, user_input, party_input):
        # pack train_data
        uid = user_input['uid']
        print(uid)
        gender = user_input['gender']
        birthyear = user_input['birthyear']
        constellation = user_input['constellation']
        character_tag = user_input['character_tag']
        user_tag = user_input['user_tag']
        theme_tag = user_input['theme_tag']
        # print(self.fc_umemo(self.fc_umemo(umemo)).size())

        pid = party_input['pid']
        title = party_input['title']
        activity_type = party_input['activity_type']
        activity_theme_tag = party_input['activity_theme_tag']
        # activity_tag = party_input['activity_tag']
        people_max = party_input['people_max']
        people_min = party_input['people_min']
        sex_limit_type = party_input['sex_limit_type']
        longitude = party_input['longitude']
        latitude = party_input['latitude']
        pre_amt = party_input['pre_amt']
        # ave_price = party_input['ave_price']
        # total_amt = party_input['total_amt']
        price_type = party_input['price_type']

        if torch.cuda.is_available():
            uid, gender, birthyear, constellation, character_tag, user_tag, theme_tag, \
            pid, title, activity_type, activity_theme_tag, activity_tag, people_max, people_min, sex_limit_type, longitude, \
            latitude, pre_amt, ave_price, total_amt, price_type = \
            uid.to(device), gender.to(device), birthyear.to(device), constellation.to(device), character_tag.to(device),\
            user_tag.to(device), theme_tag.to(device), \
            pid.to(device), title.to(device), activity_type.to(device), activity_theme_tag.to(device), \
            people_max.to(device), people_min.to(device), sex_limit_type.to(device), longitude.to(device), \
            latitude.to(device), pre_amt.to(device), price_type.to(device)

        # user channel
        print(uid)
        feature_uid = self.BN(F.relu(self.fc_uid(self.embedding_uid(uid))))
        feature_gender = self.BN(F.relu(self.fc_gender(self.embedding_gender(gender))))
        feature_birthyear = self.BN(F.relu(self.fc_level(self.embedding_birthyear(birthyear))))
        feature_constellation = self.BN(F.relu(self.fc_constellation(self.embedding_constellation(constellation))))
        feature_character_tag = self.BN(F.relu(self.fc_character_tag(self.embedding_character_tag(character_tag)).view(-1, 1, 1, 32)))
        feature_user_tag = self.BN(F.relu(self.fc_user_tag(self.embedding_user_tag(user_tag)).view(-1, 1, 1, 32)))
        feature_theme_tag = self.BN(F.relu(self.fc_theme_tag(self.embedding_theme_tag(theme_tag)).view(-1, 1, 1, 32)))
        '''
        umemo_tensor = []
        for conv in self.Convs_text:
            umemo_tensor.append(conv(feature_umemo.view(-1,1,1,32)).view(-1,1, 8))
        feature_umemo = F.dropout(torch.cat(umemo_tensor,2), p=0.5)
        '''
        feature_user = torch.tanh(self.fc_user_combine(
            torch.cat([feature_uid.view(-1, 1, 32), feature_gender.view(-1, 1, 32), feature_birthyear.view(-1, 1, 32),
                       feature_constellation.view(-1, 1, 32),
                       feature_character_tag.view(-1, 1, 32), feature_user_tag.view(-1, 1, 32), feature_theme_tag.view(-1, 1, 32)], 2)
        )).view(-1, 1, 200)

        # party channel
        feature_pid = self.BN(F.relu(self.fc_pid(self.embedding_pid(pid))))
        feature_title = self.BN(F.relu(self.fc_title(title)))
        feature_activity_type = self.BN(F.relu(self.fc_activity_type(self.embedding_activity_type(activity_type))))
        feature_activity_theme_tag = self.BN(F.relu(self.fc_activity_theme_tag(self.embedding_activity_theme_tag(activity_theme_tag)).view(-1, 1, 1, 32)))
        # feature_activity_tag = self.BN(F.relu(self.fc_activity_tag(self.embedding_activity_tag(activity_tag)).view(-1, 1, 1, 32)))
        feature_people_max = self.BN(
            F.relu(self.fc_people_max(self.embedding_people_max(people_max))))
        feature_people_min = self.BN(
            F.relu(self.fc_people_min(self.embedding_people_min(people_min))))
        feature_sex_limit_type = self.BN(
            F.relu(self.fc_sex_limit_type(self.embedding_sex_limit_type(sex_limit_type))))
        feature_longitude = self.BN(F.relu(self.fc_longitude(self.embedding_longitude(longitude))))
        feature_latitude = self.BN(F.relu(self.fc_latitude(self.embedding_latitude(latitude))))
        feature_pre_amt = self.BN(F.relu(self.fc_pre_amt(self.embedding_pre_amt(pre_amt))))
        # feature_ave_price = self.BN(F.relu(self.fc_ave_price(self.embedding_ave_price(ave_price))))
        # feature_total_amt = self.BN(F.relu(self.fc_total_amt(self.embedding_total_amt(total_amt))))
        feature_price_type = self.BN(F.relu(self.fc_price_type(self.embedding_price_type(price_type))))
        '''
        pmemo_tensor = []
        for conv in self.Convs_text:
            pmemo_tensor.append(conv(feature_pmemo.view(-1,1,16,32)).view(-1,1, 8))
        feature_pmemo = F.dropout(torch.cat(pmemo_tensor,2), p=0.5)
        '''
        # feature_party B x 1 x 200
        feature_party = torch.tanh(self.fc_party_combine(
            torch.cat([feature_pid.view(-1, 1, 32), feature_title.view(-1, 1, 32), feature_activity_type.view(-1, 1, 32),
                       feature_activity_theme_tag.view(-1, 1, 32), feature_people_max.view(-1, 1, 32),
                       feature_people_min.view(-1, 1, 32), feature_sex_limit_type.view(-1, 1, 32), feature_longitude.view(-1, 1, 32),
                       feature_latitude.view(-1, 1, 32), feature_pre_amt.view(-1, 1, 32), feature_price_type.view(-1, 1, 32)], 2)
        ))
        output = torch.sum(feature_user * feature_party, 2)  # B x rank
        return output, feature_user, feature_party