from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch
import utils
import pickle
import copy


class biLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, out_size):
        super(biLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.bilstm = nn.LSTM(emb_dim, hidden_size, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(2 * hidden_size, out_size)
        self._best_val_loss = 10000.

    def forward(self, x, length):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, length, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        scores = self.dense(rnn_out)
        return scores

    def validate(self, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.bilstm.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_word_lists), utils.batch_size):
                val_step += 1
                batch_sents = dev_word_lists[ind:ind + utils.batch_size]
                batch_tags = dev_tag_lists[ind:ind + utils.batch_size]

                tensorized_sents, lengths = utils.tensorize(batch_sents, word2id, tag=False)
                tensorized_sents = tensorized_sents.to(utils.device)

                targets, lengths = utils.tensorize(batch_tags, tag2id, tag=True)
                targets = targets.to(utils.device)

                scores = self(tensorized_sents, lengths)

                loss = self.cal_loss(scores, targets, tag2id).to(utils.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            # if val_loss < self._best_val_loss:
            print("Saving...")
            with open('./saved_models/model_loss{:.4f}.model'.format(val_loss), 'wb') as model_file:
                pickle.dump(self, model_file)
                # self._best_val_loss = val_loss

            return val_loss

    def cal_loss(self, logits, targets, tag2id):
        PAD = tag2id.get('<pad>')

        mask = (targets != PAD)
        targets = targets[mask]
        out_size = logits.size(2)
        logits = logits.masked_select(
            mask.unsqueeze(2).expand(-1, -1, out_size)
        ).contiguous().view(-1, out_size)

        loss = F.cross_entropy(logits, targets)

        return loss

