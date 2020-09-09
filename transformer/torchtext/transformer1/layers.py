import torch
import torch.nn as nn

import time
class MultiheadAttention(nn.Module):
    def __init__(self , num_head , embedding_dim):
        super(MultiheadAttention , self).__init__()
        self.num_head = num_head
        self.embedding_dim = embedding_dim
        self.head_dim = int(embedding_dim / num_head)

        self.query = nn.Linear(self.head_dim , self.head_dim )
        self.key = nn.Linear(self.head_dim , self.head_dim)
        self.value = nn.Linear(self.head_dim , self.head_dim)
        self.fc = nn.Linear(embedding_dim , embedding_dim)

    def forward(self , query , key ,value , mask):
        print("In multihead")
        N = query.shape[0]


        query_len , key_len , value_len = query.shape[1] , key.shape[1] , value.shape[1]

        #input_shape ==> ( N , seq_len , embedding_dim)
        value = value.reshape(N , value_len ,self.num_head, self.head_dim)
        query = query.reshape(N, query_len, self.num_head,self.head_dim)
        key = key.reshape(N, key_len, self.num_head, self.head_dim)

        query = self.query(query) # query ==> (N , query_len , head_dim)
        value = self.value(value)   # query ==> (N , value_len , head_dim)
        key = self.key(key) # query ==> (N , key_len , head_dim)


        energy = torch.einsum('nqhd , nkhd -> nhqk' , [query , key])

        if mask is not None:
            energy = energy.masked_fill(mask == 0 , float(-1e20))

        attention = torch.softmax(energy/(self.embedding_dim **(1/2)) , dim = 3)

        out = torch.einsum(('nhqk , nvhd -> nqhd') , [attention , value]).reshape(shape = (N , query_len , self.embedding_dim))

        out = self.fc(out)

        return out


class PostitionalandWordEncoding(nn.Module):
    def __init__(self , embedding_dim , input_vocab):
        super(PostitionalandWordEncoding , self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_vocab , embedding_dim)
        self.embedding1 = nn.Embedding(input_vocab, embedding_dim)
        self.dropout = nn.Dropout(p = 0.1)

    def forward(self, x):
        print("In Embedding")
        N , length = x.shape

        rang = torch.arange(start = 0 , end = length , step = 1).expand(size = (N , length))

        rang = self.dropout(self.embedding(rang) + self.embedding1(x)) #shape ==> ( N , seq_len , embedding_dim)

        return rang

class TransformerBlock(nn.Module):
    def __init__(self , embedding_dim ,num_head):
        super(TransformerBlock , self).__init__()
        self.embedding_dim = embedding_dim
        self.num_head = num_head

        self.attention = MultiheadAttention(self.num_head , self.embedding_dim)
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.layernorm2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim , embedding_dim*2),
            nn.ReLU(),
            nn.Linear(embedding_dim*2 , embedding_dim)
        )

        self.dropout = nn.Dropout(p = 0.1)


    def forward(self ,query , key ,value, mask):
        print("In block")

        attention = self.attention(query , key , value , mask)

        x = self.dropout(self.layernorm1(query + attention))
        x = self.feed_forward(x)
        out = self.dropout(x)

        return out

class DecoderBlock(nn.Module):
    def __init__(self , embedding_size , head):
        super(DecoderBlock , self).__init__()
        self.embedding_size = embedding_size
        self.num_head = head

        self.layernorm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p = 0.2)
        self.attention = MultiheadAttention(self.num_head,self.embedding_size )
        self.transformerblock = TransformerBlock(self.embedding_size , self.num_head)

    def forward(self , x , value , key , src_mask , trg_mask):
        attention = self.attention(x , x , x ,trg_mask)
        query = self.dropout(self.layernorm(attention + x))

        out = self.transformerblock(query , key , value , src_mask)

        return out

if __name__ == "__main__":
    start = time.time()
    x = torch.randint(0 ,100 , size = (1000 , 16))
    new = torch.rand(size = (1000 , 12 , 512))
    layer = PostitionalandWordEncoding(512 , 200)
    multi = TransformerBlock(512 , 16)


    out = layer(x)
    out = multi(out ,out , out , None)
    print(out.shape)
    decoder = DecoderBlock(512, 16)
    out = decoder(new,out ,out , None , None)
    end = time.time()

    print(out.shape , f"time taken by {end-start}")