from data import data_loader
import torch
import utils
import pickle

dl = data_loader()
test_data = dl.make_data_from_pkl('./pickle/dev_eva.pkl')['data']
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
data2 = {'Word': [], 'Tag': []}

with open('./dataset/dev.txt', encoding='utf8') as file:
    for line in file.readlines():
        if line != '\n':
            word = line[:-1].split('\t')
            data2['Tag'].append(word[-1])

with open('./submit/dev_eva.txt', 'w', encoding="utf-8") as result_file:
    n = 0
    for i in range(scores.shape[0]):
        data.append([])
        tag.append([])
        result.append([])
        for j in range(lengths[i]):
            data[i].append(test_data[i][j])
            tag[i].append(data2['Tag'][n])
            result[i].append(utils.tag_list_dec[torch.max(scores[i, j], 0)[1]])
            result_file.write('{}\t{}\t{}\n'.format(
                test_data[i][j], data2['Tag'][n], utils.tag_list_dec[torch.max(scores[i, j], 0)[1]]))
            n += 1
        result_file.write('\n')