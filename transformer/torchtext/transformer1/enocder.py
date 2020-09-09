import torch
import torch.nn as nn

from transformer1.layers import PostitionalandWordEncoding, TransformerBlock


class Encoder(nn.Module):
    def __init__(self , input_vocab , embedding_dim , num_head , num_layers):
        super(Encoder , self).__init__()
        self.input_vocab = input_vocab
        self.embedding_dim = embedding_dim
        self.num_head = num_head
        self.num_layers = num_layers

        self.posiotionadnword = PostitionalandWordEncoding(self.embedding_dim , self.input_vocab)
        self.transformerblock = nn.ModuleList(

            [TransformerBlock(self.embedding_dim
                              , self.num_head)
            for _ in range(self.num_layers)]
        )

    def forward(self , x , mask):

        #shape ==> (N , seq_len)
        out = self.posiotionadnword(x)
        #shape ==> (N , seq_len , embedding_dim)
        for layers in self.transformerblock:
            out = layers(out , out , out ,mask)


        return out

if __name__ == "__main__":
    x = torch.randint(0, 100, size=(64, 16))

    encoder = Encoder(200 , 512 ,16 , 6)

    out = encoder(x , None)

    print(out.shape)