import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class rec_model(nn.Module):

    def __init__(self, user_max_dict, party_max_dict, convParams, embed_dim=32, fc_size=200):
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
        self.embedding_level = nn.Embedding(user_max_dict['level'], embed_dim // 2)
        self.embedding_constellation = nn.Embedding(user_max_dict['constellation'], embed_dim // 2)
        self.embedding_birthyear = nn.Embedding(user_max_dict['birthyear'], embed_dim)
        self.embedding_price = nn.Embedding(user_max_dict['price'], embed_dim)

        # region embedding
        self.embedding_country = nn.Embedding(user_max_dict['country'], embed_dim)
        self.embedding_province = nn.Embedding(user_max_dict['province'], embed_dim // 2)
        self.embedding_city = nn.Embedding(user_max_dict['city'], embed_dim)
        # real-time address embedding tbc..

        # user embedding to fc: the first dense layer
        self.fc_uid = nn.Linear(embed_dim, embed_dim)
        self.fc_gender = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_level = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_constellation = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_birthyear = nn.Linear(embed_dim, embed_dim)
        self.fc_price = nn.Linear(embed_dim, embed_dim)
        # region embedding to fc and concat
        self.fc_country = nn.Linear(embed_dim, embed_dim)
        self.fc_province = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_city = nn.Linear(embed_dim, embed_dim)
        self.fc_region = nn.Linear(3 * embed_dim, embed_dim)

        # user channel concat
        self.fc_user_combine = nn.Linear(7 * embed_dim, fc_size)

        # --------------------------------- party channel -----------------------------------------------------------------
        # party embeddings
        self.embedding_pid = nn.Embedding(party_max_dict['pid'], embed_dim)  # normally 32
        self.embedding_owner = nn.Embedding(party_max_dict['owner'], embed_dim)
        self.embedding_planner = nn.Embedding(party_max_dict['planner'], embed_dim)
        self.embedding_maxprice = nn.Embedding(party_max_dict['max_price'], embed_dim)
        self.embedding_minprice = nn.Embedding(party_max_dict['min_price'], embed_dim)
        self.embedding_participant_num = nn.Embedding(party_max_dict['participant_num'], embed_dim // 2)
        self.embedding_participants = nn.EmbeddingBag(party_max_dict['participants'], embed_dim, mode='sum')
        self.embedding_project_tagid = nn.EmbeddingBag(party_max_dict['project_tagid'], embed_dim, mode='sum')
        # self.embedding_store = nn.EmbeddingBag(party_max_dict['store'], embed_dim, mode='sum')

        self.fc_pid = nn.Linear(embed_dim, embed_dim)
        self.fc_owner = nn.Linear(embed_dim, embed_dim)
        self.fc_planner = nn.Linear(embed_dim, embed_dim)
        self.fc_maxprice = nn.Linear(embed_dim, embed_dim)
        self.fc_minprice = nn.Linear(embed_dim, embed_dim)
        self.fc_participant_num = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_participants = nn.Linear(embed_dim, embed_dim)
        self.fc_project_tagid = nn.Linear(embed_dim, embed_dim)
        # self.fc_store = nn.Linear(embed_dim, embed_dim)

        # party channel concat
        self.fc_party_combine = nn.Linear(8 * embed_dim, fc_size)  # tanh

        # BatchNorm layer
        self.BN_uid = nn.BatchNorm2d(1)
        self.BN_gender = nn.BatchNorm2d(1)
        self.BN_level = nn.BatchNorm2d(1)
        self.BN_constellation = nn.BatchNorm2d(1)
        self.BN_birthyear = nn.BatchNorm2d(1)
        self.BN_region = nn.BatchNorm2d(1)
        self.BN_price = nn.BatchNorm2d(1)

        self.BN_pid = nn.BatchNorm2d(1)
        self.BN_owner = nn.BatchNorm2d(1)
        self.BN_planner = nn.BatchNorm2d(1)
        self.BN_maxprice = nn.BatchNorm2d(1)
        self.BN_minprice = nn.BatchNorm2d(1)
        self.BN_participant_num = nn.BatchNorm2d(1)
        self.BN_participants = nn.BatchNorm2d(1)
        self.BN_project_tagid = nn.BatchNorm2d(1)

    def forward(self, user_input, party_input):
        # pack train_data
        uid = user_input['uid']
        gender = user_input['gender']
        level = user_input['level']
        constellation = user_input['constellation']
        birthyear = user_input['birthyear']
        country = user_input['country']
        province = user_input['province']
        city = user_input['city']
        price = user_input['price']

        pid = party_input['pid']
        owner = party_input['owner']
        planner = party_input['planner']
        maxprice = party_input['maxprice']
        minprice = party_input['minprice']
        participant_num = party_input['participant_num']
        participants = party_input['participants']
        project_tagid = party_input['project_tagid']

        if torch.cuda.is_available():
            uid, gender, level, constellation, birthyear, country, province, city, price,\
                pid, owner, planner, maxprice, minprice, participant_num, participants, title, pmemo = \
            uid.to(device), gender.to(device), level.to(device), constellation.to(device), birthyear.to(device), country.to(device), province.to(device),
            city.to(device), price.to(device),
            pid.to(device), owner.to(device), maxprice.to(device), minprice.to(device), participant_num.to(device), participants.to(device),
            project_tagid.to(device)

        # user channel
        feature_uid = self.BN_uid(F.relu(self.fc_uid(self.embedding_uid(uid))))
        feature_gender = self.BN_gender(F.relu(self.fc_gender(self.embedding_gender(gender))))
        feature_level =  self.BN_level(F.relu(self.fc_age(self.embedding_level(level))))
        feature_constellation = self.BN_constellation(F.relu(self.fc_constellation(self.embedding_constellation(constellation))))
        feature_birthyear = self.BN_birthyear(F.relu(self.fc_birthyear(self.embedding_birthyear(birthyear))))
        feature_region = self.BN_region(F.relu(self.fc_region(torch.cat([self.fc_country(self.embedding_country(country)),
                                                                         self.fc_province(self.embedding_province(province)),
                                                                         self.fc_city(self.embedding_city(city))], 3))))
        feature_price = self.BN_price(F.relu(self.fc_price(self.embedding_price(price))))

        feature_user = F.tanh(self.fc_user_combine(
            torch.cat([feature_uid.view(-1,1,32), feature_gender.view(-1,1,32), feature_level.view(-1,1,32), feature_constellation.view(-1,1,32),
                       feature_birthyear.view(-1,1,32), feature_region.view(-1,1,32), feature_price.view(-1,1,32)], 2)
        )).view(-1,1,200)

        # party channel
        feature_pid = self.BN_pid(F.relu(self.fc_pid(self.embedding_pid(pid))))
        feature_owner = self.BN_owner(F.relu(self.fc_owner(self.embedding_owner(owner))))
        feature_planner = self.BN_planner(F.relu(self.fc_planner(self.embedding_planner(planner))))
        feature_maxprice = self.BN_maxprice(F.relu(self.fc_maxprice(self.embedding_maxprice(maxprice))))
        feature_minprice = self.BN_minprice(F.relu(self.fc_minprice(self.embedding_minprice(minprice))))
        feature_participant_num = self.BN_participant_num(F.relu(self.fc_participant_num(self.embedding_participant_num(participant_num))))
        feature_participants = self.BN_participants(F.relu(self.fc_participants(self.embedding_participants(participants)).view(-1,1,1,32)))
        feature_project_tagid = self.BN_project_tagid(F.relu(self.fc_project_tagid(self.embedding_project_tagid(project_tagid)).view(-1, 1, 1, 32)))

        # feature_party B x 1 x 200
        feature_party = F.tanh(self.fc_party_combine(
            torch.cat([feature_pid.view(-1,1,32), feature_owner.view(-1,1,32), feature_planner.view(-1,1,32), feature_maxprice.view(-1,1,32),
                       feature_minprice.view(-1,1,32), feature_participant_num.view(-1,1,32), feature_participants.view(-1,1,32),
                       feature_project_tagid.view(-1,1,32)], 2)
        ))

        output = torch.sum(feature_user * feature_party, 2)  # B x rank
        return output, feature_user, feature_party