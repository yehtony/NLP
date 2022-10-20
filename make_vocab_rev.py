from data import data_loader
import utils
import pickle

dl = data_loader()


vocab = dl.make_data_from_pkl('vocab.pkl')
vocab_rev = [' ']*len(vocab)
for each in vocab.keys():
    vocab_rev[vocab[each]] = each

with open('./vocab_rev.pkl', 'wb') as rev_file:
    pickle.dump(vocab_rev, rev_file)