import torch
import torch.nn as nn
import torch.nn.functional as F
import random




class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 hidden_size,
                 num_layer,
                 p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layer, dropout=p)

    def forward(self, x):
        # Embedding our sentence (shape (seq_len , N) (where N is batch size)
        embed = self.dropout(self.embedding(x))
        # Output shape (seq_len , N , embed_size)
        output, (hidden, cell) = self.rnn(embed)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 hidden_size,
                 ouput_size,
                 num_layer,
                 p):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layer, dropout=p)
        self.fc = nn.Linear(hidden_size, ouput_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embed = self.dropout(self.embedding(x))
        # output_shape : (1 , N , embedding_size)

        outputs, (hidden, cell) = self.rnn(embed, (hidden, cell))
        # output_shape : (1 , N , hidden_size)

        predictions = self.fc(outputs)
        # output_shape:(1, N , len_of_lan)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


# defining our sequence to sequence model

class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 target_vocab):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab = target_vocab

    def forward(self, src, target, teacher_force_ratio=0.5):
        target_len = target.shape[0]
        batch_size = src.shape[1]

        target_vocab_len = self.target_vocab

        outputs = torch.zeros(size=(target_len, batch_size, target_vocab_len))

        hidden, cell = self.encoder(src)

        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output

            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
