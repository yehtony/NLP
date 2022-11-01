import utils
from data import data_loader
from model import biLSTM  # , Model_CRF
import pickle
from torch import optim
import torch
import os

# dataloader = data.data_loader()
# dataloader.load()


dl = data_loader()
vocab = dl.make_data_from_pkl('./vocab.pkl')
data = dl.make_data_from_pkl('./dataset/train.pkl')
dev = dl.make_data_from_pkl('./dataset/dev.pkl')

if utils.create_new_model:
    nn_model = biLSTM(vocab_size=len(vocab), emb_dim=utils.emb_size,
                      hidden_size=utils.hidden_size, out_size=len(utils.tag_list)).to(utils.device)
else:
    with open('./saved_models/'+utils.get_file_list('./saved_models/')[-1], 'rb') as model_file:
        nn_model = pickle.load(model_file).to(utils.device)

word_lists, tag_lists, _ = utils.sort_by_lengths(data['data'], data['label'])
dev_word_lists, dev_tag_lists, _ = utils.sort_by_lengths(
    dev['data'], dev['label'])
nn_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# optimizer = optim.Adam(nn_model.parameters(), lr=utils.lr)
loss_f = nn_model.cal_loss

for ep in range(1, utils.epoch + 1):
    batch_counter = 0
    losses = 0.
    for ind in range(0, len(word_lists), utils.batch_size):
        nn_model.train()
        batch_sents, batch_tags, _ = utils.sort_by_lengths(word_lists[ind:ind + utils.batch_size],
                                                           tag_lists[ind:ind + utils.batch_size])
        batch_sents, lengths = utils.tensorize(batch_sents, vocab, tag=False)
        batch_tags, _ = utils.tensorize(batch_tags, utils.tag_list, tag=True)
        batch_sents = batch_sents.to(utils.device)
        batch_tags = batch_tags.to(utils.device)

        scores = nn_model(batch_sents, lengths)

        optimizer.zero_grad()
        loss = loss_f(scores, batch_tags, utils.tag_list).to(utils.device)
        loss.backward()
        optimizer.step()

        batch_counter += 1
        losses += loss.item()

        if batch_counter % utils.print_step == 0:
            total_batches = (len(word_lists) // utils.batch_size + 1)
            print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                ep, batch_counter, total_batches,
                100. * batch_counter / total_batches,
                losses / utils.print_step
            ))
            losses = 0

    val_loss = nn_model.validate(
        dev_word_lists, dev_tag_lists, vocab, utils.tag_list)
    print("Epoch {}, Val Loss:{:.4f}".format(ep, val_loss))
