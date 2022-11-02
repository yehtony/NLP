import utils
import torch
import pickle

class data_loader:
    def __init__(self):
        pass

    def make_data_from_pkl(self, file_dir):
        data_file = open(file_dir, 'rb')
        data = pickle.load(data_file)
        data_file.close()
        return data

    # def load_data(self):
    #     with open(utils.train_dir_0) as file:
    #         data = []
    #         labels = []
    #
    #         sentence = []
    #         label = []
    #         for line in file.readlines():
    #             if '\t' in line:
    #                 word = line[:-1].split('\t')
    #                 sentence.append(word[0])
    #                 label.append(word[1])
    #             else:
    #                 data.append(sentence)
    #                 labels.append(label)
    #                 sentence = []
    #                 label = []
    #
    #     return data, labels

    # def load(self):
    #     with open(utils.train_dir_1) as file:
    #         data = []
    #         labels = []
    #
    #         sentence = []
    #         label = []
    #
    #         counter = 0
    #         for line in file.readlines():
    #             if line is not '\n':
    #                 word = line[:-1].split()
    #                 sentence.append(word[0])
    #                 label.append(word[-1])
    #                 if word[-1] not in utils.extra_tag:
    #                     utils.extra_tag[word[-1]] = counter
    #                     counter += 1
    #             else:
    #                 if len(sentence) > 5:
    #                     data.append(sentence)
    #                     labels.append(label)
    #                 sentence = []
    #                 label = []
    #     print(utils.extra_tag)
    #     return data, labels
