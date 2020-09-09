import torch
import torch.nn as nn
from transformer1.enocder import Encoder
from transformer1.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self ,
                 input_vocab,
                 output_vocab,
                 embedding_dim,
                 num_head,
                 src_pad_idx,
                 trg_pad_idx,
                 num_layers,
                 ):
        super(Transformer , self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder = Encoder(input_vocab
                               ,embedding_dim
                               ,num_head
                               ,num_layers)
        self.decoder = Decoder(output_vocab
                               ,embedding_dim
                               ,num_head
                               ,num_layers)

    def make_src_mask(self , src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #shape==> ( N , 1 , 1 , src_len)
        return src_mask

    def make_trg_mask(self , trg):
        N ,trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len , trg_len))).expand(
            N ,1 , trg_len , trg_len
            )
        return trg_mask

    def forward(self , src , trg):
        src_mask = self.make_src_mask(src)
        print(src_mask)
        trg_mask = self.make_trg_mask(trg)
        print(trg_mask)
        enc_src = self.encoder(src , src_mask)

        out = self.decoder(trg , enc_src , src_mask , trg_mask)

        return out

if __name__ == "__main__":

    inp = torch.randint(0,200 , size = (64 , 16))
    out = torch.randint(0 , 200 , size = (64 , 12))

    src_pad_idx , trg_pad_idx = 0 , 0
    input_vocab = 200
    trg_vocab = 200

    model = Transformer(input_vocab,
                        trg_vocab,
                        512,
                        16,
                        src_pad_idx,
                        trg_pad_idx,
                        6)

    out = model(inp , out)

    print(out.shape)