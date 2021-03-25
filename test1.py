import gensim as gensim
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
from model_CNN import rec_model
from recInterface_party import getKNNitem,getUserMostLike
import torch
from train import getUser_dict, getParty_dict, getConv
if __name__=='__main__':
    MODEL_PATH = 'Params/word2vec_c'
    CNN_PATH = 'Params/CNN_model_params .txt'
    # word_vectors = api.load(MODEL_PATH)
    # model = gensim.models.Word2Vec.load(MODEL_PATH)
    model = rec_model(user_max_dict=getUser_dict(), party_max_dict=getParty_dict(), convParams=getConv())
    model.load_state_dict(torch.load(CNN_PATH))
    text_model = word2vec.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)
    text_model = Word2Vec.load(MODEL_PATH)
    print(text_model['烧烤'])
    print(getKNNitem(itemID=1000, itemName='user', K=10))
    print(getUserMostLike(uid=1000))
    print(getKNNitem(itemID=100008, itemName='party', K=10))