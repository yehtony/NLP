from model import biLSTM
from data import data_loader
import torch
import utils
import pickle

dl = data_loader()
test_data = dl.make_data_from_pkl('./dataset/test.pkl')['data']
vocab = dl.make_data_from_pkl('vocab.pkl')
vocab_rev = dl.make_data_from_pkl('vocab_rev.pkl')
with open('./saved_models/model_loss0.5736.model', 'rb') as model_file:
    nn_model = pickle.load(model_file).to(utils.device)

test_data = utils.sort_by_lengths(test_data, utils.tag_list)[0]
test_data_tensor, lengths = utils.tensorize(test_data, vocab, tag=False)
test_data_tensor = test_data_tensor.to(utils.device)
scores = nn_model(test_data_tensor, lengths)

data = []
result = []

# def get_key (dict, value):
#     return [k for k, v in dict.items() if v == value]

with open('./test-submit2.txt', 'w') as result_file:
    for i in range(scores.shape[0]):
        data.append([])
        result.append([])
        for j in range(lengths[i]):
            data[i].append(test_data[i][j])
            result[i].append(utils.tag_list_rev[torch.max(scores[i, j], 0)[1]])
            result_file.write('{}\t{}\n'.format(test_data[i][j], utils.tag_list_rev[torch.max(scores[i, j], 0)[1]]))
        result_file.write('\n')
