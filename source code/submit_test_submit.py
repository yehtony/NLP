from data import data_loader
import torch
import utils
import pickle

dl = data_loader()
test_data = dl.make_data_from_pkl('./pickle/test-submit.pkl')['data']
vocab = dl.make_data_from_pkl('./pickle/vocab.pkl')
vocab_dec = dl.make_data_from_pkl('./pickle/vocab_dec.pkl')
with open('./model/loss0.7312.model', 'rb') as model_file:
    nn_model = pickle.load(model_file).to(utils.device)


test_data2 = utils.sort_by_lengths_sub(test_data)[0]
test_data_tensor, lengths = utils.tensorize_sub(
    test_data, test_data2, vocab, tag=False)
test_data_tensor = test_data_tensor.to(utils.device)
scores = nn_model(test_data_tensor, lengths)

data = []
tag = []
result = []

with open('./submit/test-submit.txt', 'w', encoding="utf-8") as result_file:
    for i in range(scores.shape[0]):
        data.append([])
        result.append([])
        for j in range(lengths[i]):
            data[i].append(test_data[i][j])
            result[i].append(utils.tag_list_dec[torch.max(scores[i, j], 0)[1]])
            result_file.write('{}\t{}\n'.format(
                test_data[i][j], utils.tag_list_dec[torch.max(scores[i, j], 0)[1]]))
        result_file.write('\n')