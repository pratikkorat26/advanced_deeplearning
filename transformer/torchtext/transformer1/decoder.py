import torch
import torch.nn as nn
from transformer1.layers import PostitionalandWordEncoding , DecoderBlock

class Decoder(nn.Module):
    def __init__(self , output_vocab ,embedding_dim , num_head ,  num_layers):
        super(Decoder , self).__init__()
        self.embdding_dim = embedding_dim
        self.num_head = num_head
        self.output_vocab = output_vocab
        self.num_layers = num_layers

        self.postion_embedding = PostitionalandWordEncoding(self.embdding_dim , self.output_vocab)

        self.feed_forward = nn.ModuleList(
            [
                DecoderBlock(self.embdding_dim, self.num_head) for _ in range(self.num_layers)
            ]
        )
        self.fc_out = nn.Linear(self.embdding_dim , self.output_vocab)

    def forward(self , x , enc_out , src_mask , trg_mask):
        print("in decoder")
        embedding = self.postion_embedding(x)
        #shape ==> (N , trg_len , embedding_size)

        for layers in self.feed_forward:
            out = layers(embedding , enc_out ,enc_out ,src_mask , trg_mask)

        out = self.fc_out(out)

        return out

if __name__ == "__main__":
    x = torch.randint(0, 100, size=(64, 16))
    enc_out = torch.rand(size = (64 , 12 , 512))

    layer = Decoder(512 , 16 , 200 , 6)

    out = layer(x , enc_out , None , None)

    print(out.shape)


