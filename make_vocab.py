import os
import pickle

data_file = './dataset/train.txt'
split_char = ' '

if 'vocab.pkl' in os.listdir():
    with open('vocab.pkl', 'rb') as file_read:
        vocab = pickle.load(file_read)
else:
    vocab = {}


if '<unk>' not in vocab: vocab['<unk>'] = len(vocab)
if '<pad>' not in vocab: vocab['<pad>'] = len(vocab)

with open(data_file) as file:
    for line in file.readlines():
        word = line[:-1].split(split_char)
        if (word[0].lower() not in vocab) and ('@' not in word[0]) and ('/' not in word[0]) and (len(word[0]) > 0):
            vocab[word[0].lower()] = len(vocab)
    file_write = open('vocab.pkl', 'wb')
    pickle.dump(vocab, file_write)
    file_write.close()
