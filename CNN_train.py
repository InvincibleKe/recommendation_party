from model_CNN import rec_model
from dataset import MovieRankDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import os
import csv
from tensorboardX import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --------------- hyper-parameters------------------
user_max_dict={
    'uid': 1000,  # 1000 users
    'gender': 2,
    'level': 3,
    'constellation': 12,
    'birthyear': 100,
    'country': 233,
    'province': 34,
    'city': 333,
    'address': 100,  # 100 words
    'price': 10000,
    'hobby': 50,  # 50 words
    'diet_prefer': 50,
    'memo': 100,
    'store': 1000  # each user can collect max 5 stores
}

party_max_dict={
    'pid':1000,  # 1000 parties
    'title':50,  # 50 words
    'owner':1000,  # each party can only have 1 owner
    'planner':1000,  # each party can only have 1 planner
    'memo': 100,  # 100 words
    'adress': 100,  # 100 words
    'max_price': 10000,
    'min_price': 10000,
    'project_tagid': 1000,
    'participant_num': 100,
    'participants': 100
    #'store': 10
}

convParams={
    'kernel_sizes':[2,3,4,5]
}

model = rec_model(user_max_dict=user_max_dict, movie_max_dict=movie_max_dict, convParams=convParams)
model = model.to(device)

# training hyper parameters
num_epochs = 100
lr = 0.0001
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
batch_size = 256
# model save files
PATH = 'Params/CNN_model_params.txt'
if not os.path.exists(PATH):
    os.makedirs(PATH)

USER_PATH = 'Data/user_info.csv'
ACTIVITY_PATH = 'Data/activity.csv'

def train():
    datasets = MovieRankDataset(pkl_file='data.p')
    print(datasets[0])
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)

    writer = SummaryWriter()
    for epoch in range(num_epochs):
        loss_all = 0
        for i_batch, sample_batch in enumerate(dataloader):

            user_inputs = sample_batch['user_inputs']
            movie_inputs = sample_batch['movie_inputs']
            target = sample_batch['target'].to(device)

            model.zero_grad()
            tag_rank, _, _ = model(user_inputs, movie_inputs)

            loss = loss_function(tag_rank, target)
            if i_batch % 20 == 0:
                writer.add_scalar('data/loss', loss, i_batch*20)
                print(loss)

            loss_all += loss
            loss.backward()
            optimizer.step()
        print('Epoch {}:\t loss:{}'.format(epoch,loss_all))
    writer.export_scalars_to_json("./test.json")
    writer.close()


if __name__=='__main__':
    print(device)

    # train model
    train()
    torch.save(model.state_dict(), PATH)

    # get user and movie feature
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    from recInterface import saveMovieAndUserFeature
    saveMovieAndUserFeature(model=model)

    # test recsys
    from recInterface import getKNNitem, getUserMostLike
    print(getKNNitem(itemID=100,K=10))
    print(getUserMostLike(uid=100))