
from model import rec_model
import torch
import gensim as gensim
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import gensim.models.keyedvectors as word2vec
from gensim.models.word2vec import LineSentence
import re
import gensim.downloader as api
import jieba
import jieba.posseg as pseg
uid=torch.tensor(
    [1,2,3,4]
).long

user_max_dict={
    'uid':10,
    'gender':2,
    'age':10,
    'job':10
}

movie_max_dict={
    'mid':10,
    'mtype':20,
    'mword':20
}

convParams={
    'kernel_sizes':[2,3,4,5]
}
def simulatorData():
    uid = mid = age = job = torch.tensor(
        [1, 2, 3, 4]
    ).view(4, -1).long()

    gender = torch.tensor(
        [1, 0, 1, 0]
    ).view(4, -1).long()

    user_inputs = {
        'uid': uid,
        'gender': gender,
        'age': age,
        'job': job
    }

    mtype = torch.tensor(
        [[1] * 18, [2] * 18, [3] * 18, [4] * 18]
    ).long()

    mword = torch.tensor(
        [[1] * 15, [2] * 15, [3] * 15, [4] * 15]
    )

    movie_inputs = {
        'mid': mid,
        'mtype': mtype,
        'mtext': mword
    }
    return (user_inputs, movie_inputs)
def tensorboardTest():
    from tensorboardX import SummaryWriter

    writer = SummaryWriter()
    x = torch.FloatTensor([100])
    y = torch.FloatTensor([500])

    for epoch in range(100):
        x /= 1.5
        y /= 1.5
        loss = y-x
        print(loss)
        writer.add_histogram('zz/x', x, epoch)
        writer.add_histogram('zz.y', y, epoch)
        writer.add_scalar('data/x', x, epoch)
        writer.add_scalar('data/y', y, epoch)
        writer.add_scalar('data/loss', loss, epoch)
        writer.add_scalars('data/scalar_group', {'x':x, 'y':y,'loss':loss},epoch)
        writer.add_text('zz/text', 'zz:this is epoch '+ str(epoch), epoch)

    writer.export_scalars_to_json("./test.json")
    writer.close()
def text2vec(text,model):
    text_jieba = pseg.cut(text)
    vector = []
    for word in text_jieba:
        w = str(word.word)
        print(w)
        try:
            v = model[w]
            vector.append(v.tolist())
        except:
            pass
    return vector
if __name__=='__main__':
    # model = rec_model(user_max_dict=user_max_dict, movie_max_dict=movie_max_dict, convParams=convParams)
    # user_inputs, movie_inputs = simulatorData()
    # model(user_inputs, movie_inputs)
    # tensorboardTest()
    #a = np.zeros((10000,200))
    #pkl.dump(a, open('test.p','wb'))
    text = '爱晶可真贤惠。'
    MODEL_PATH = 'Params/word2vec/word2vec_from_weixin/word2vec/word2vec_wx'
    model = gensim.models.Word2Vec.load(MODEL_PATH)
    words_vec = np.array(text2vec(text,model))
    print(words_vec.shape)
    # word_vectors = api.load(MODEL_PATH)
    # model = gensim.models.Word2Vec.load(MODEL_PATH)
    #text_model = word2vec.KeyedVectors.load_word2vec_format(MODEL_PATH)
    #text_model = Word2Vec.load(MODEL_PATH)
    #print(text_model['非常']