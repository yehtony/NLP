# from model import biLSTM
# from data import data_loader
# import torch
# import utils
# import pickle

# dl = data_loader()
# test_data = dl.make_data_from_pkl('./dataset/test-submit.pkl')['data']
# vocab = dl.make_data_from_pkl('vocab.pkl')
# vocab_rev = dl.make_data_from_pkl('vocab_rev.pkl')
# with open('./saved_models/model_loss0.4622.model', 'rb') as model_file:
#     nn_model = pickle.load(model_file).to(utils.device)

# test_data = utils.sort_by_lengths_sub(test_data)[0]
# test_data_tensor, lengths = utils.tensorize_sub(test_data, vocab, tag=False)
# test_data_tensor = test_data_tensor.to(utils.device)
# scores = nn_model(test_data_tensor, lengths)

# data = []
# result = []

# # def get_key (dict, value):
# #     return [k for k, v in dict.items() if v == value]

# with open('./test-submit.txt', 'w') as result_file:
#     for i in range(scores.shape[0]):
#         data.append([])
#         result.append([])
#         for j in range(lengths[i]):
#             data[i].append(test_data[i][j])
#             result[i].append(utils.tag_list_rev[torch.max(scores[i, j], 0)[1]])
#             result_file.write('{}\t{}\n'.format(test_data[i][j], utils.tag_list_rev[torch.max(scores[i, j], 0)[1]]))
#         result_file.write('\n')

from cgi import test
from model import biLSTM
from data import data_loader
import torch
import utils
import pickle
from torch.utils.data import Dataset, DataLoader

dl = data_loader()
test_data = dl.make_data_from_pkl('./dataset/test-submit.pkl')['data']
vocab = dl.make_data_from_pkl('vocab.pkl')
vocab_rev = dl.make_data_from_pkl('vocab_rev.pkl')
# with open('./saved_models/model_loss0.5736.model', 'rb') as model_file:
with open('./saved_models/model_loss0.7519.model', 'rb') as model_file:
    nn_model = pickle.load(model_file).to(utils.device)


# class testset(Dataset):
#     def __init__(self):
#         self.data=test_data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# def get_batch(sample):
    #  batch_sents, batch_tags, _ = utils.sort_by_lengths(word_lists[ind:ind + utils.batch_size],
    #                                                        tag_lists[ind:ind + utils.batch_size])
    #     batch_sents, lengths = utils.tensorize(batch_sents, vocab, tag=False)
    #     batch_tags, _ = utils.tensorize(batch_tags, utils.tag_list, tag=True)
    #     batch_sents = batch_sents.to(utils.device)
    #     batch_tags = batch_tags.to(utils.device)

    #     scores = nn_model(batch_sents, lengths)

#     return

# ts =  testset()
# dl =  DataLoader(ts, batch_size=16, collate_fn=get_batch, shuffle=False)

# for index, content in enumerate(dl):
#     print(content)
#     scores = nn_model(content)
#     result = torch.nn.argmax(scores, dim=-1)
#     with open:


#####################原版######################
# test_data = utils.sort_by_lengths(test_data, utils.tag_list)[0]
# test_data_tensor, lengths = utils.tensorize(test_data, vocab, tag=False)
# test_data_tensor = test_data_tensor.to(utils.device)
# scores = nn_model(test_data_tensor, lengths)

# data = []
# result = []

# # def get_key (dict, value):
# #     return [k for k, v in dict.items() if v == value]

# with open('./dataset/1/test_submit.txt', 'w') as result_file:
#     for i in range(scores.shape[0]):
#         data.append([])
#         result.append([])
#         for j in range(lengths[i]):
#             data[i].append(test_data[i][j])
#             result[i].append(utils.tag_list_rev[torch.max(scores[i, j], 0)[1]])
#             result_file.write('{}\t{}\n'.format(test_data[i][j], utils.tag_list_rev[torch.max(scores[i, j], 0)[1]]))
#         result_file.write('\n')

#####################原版######################









#####################改######################
test_data = utils.sort_by_lengths_sub(test_data)[0]
test_data_tensor, lengths = utils.tensorize(test_data, vocab, tag=False)
test_data_tensor = test_data_tensor.to(utils.device)
scores = nn_model(test_data_tensor, lengths)

data = []
result = []

# def get_key (dict, value):
#     return [k for k, v in dict.items() if v == value]

with open('./test-submit.txt', 'w' ,encoding="utf-8") as result_file:
    for i in range(scores.shape[0]):
        data.append([])
        result.append([])
        for j in range(lengths[i]):
            data[i].append(test_data[i][j])
            result[i].append(utils.tag_list_rev[torch.max(scores[i, j], 0)[1]])
            result_file.write('{}\t{}\n'.format(test_data[i][j], utils.tag_list_rev[torch.max(scores[i, j], 0)[1]]))
        result_file.write('\n')

#####################改######################