from model_CNN import rec_model
from dataset import PartyDataset
from torch.utils.data import DataLoader
from recInterface_party import savePartyAndUserFeature
import torch
import torch.optim as optim
import torch.nn as nn
import os
from tensorboardX import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --------------- hyper-parameters------------------
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
PATH = 'Params/model_params.txt'

def train(model,num_epochs=100, lr=0.0001, batch_size=256):
    '''
    train the target model
    :param model: initial model with neural networks
    :param num_epochs:
    :param lr: learning rate
    :param batch_size:
    :return: no return but model parameters and data features stored
    '''
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    sample = PartyDataset()
    dataloader = DataLoader(sample, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        loss_all = 0
        for i_batch, sample_batch in enumerate(dataloader):
            user_inputs = sample_batch['user_inputs']
            party_inputs = sample_batch['party_inputs']
            target = sample_batch['rank'].to(device)
            model.zero_grad()
            tag_rank, _, _ = model(user_inputs, party_inputs)

            loss = loss_function(tag_rank, target)
            if i_batch % 20 == 0:
                # writer.add_scalar('data/loss', loss, i_batch*20)
                print(loss)

            loss_all += loss
            loss.backward()
            optimizer.step()
        print('Epoch {}:\t loss:{}'.format(epoch, loss_all))
    # writer.export_scalars_to_json("./test.json")
    # writer.close()
    torch.save(model.state_dict(), PATH)
    savePartyAndUserFeature(model=model)

if __name__=='__main__':
    model = rec_model(user_max_dict=user_max_dict, party_max_dict=party_max_dict, convParams=convParams)
    model = model.to(device)

    # training hyper parameters
    num_epochs = 50
    lr = 0.0001
    batch_size = 256

    # train model
    train(model=model, num_epochs=num_epochs, batch_size=batch_size)

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    torch.save(model.state_dict(), PATH)

    # get user and activity feature
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    from recInterface import savePartyAndUserFeature
    savePartyAndUserFeature(model=model)

    # test recsys
    from recInterface_party import getKNNitem, getUserMostLike
    print(getKNNitem(itemID=100, K=10))
    print(getUserMostLike(uid=100))

def getUser_dict():
    return user_max_dict
def getParty_dict():
    return party_max_dict
def getConv():
    return convParams