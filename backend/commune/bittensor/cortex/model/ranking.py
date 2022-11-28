import torch
from torch import nn
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

class RankingModel(nn.Module):


    def __init__(self, num_endpoints: int, load_optimizer=True):

        nn.Module.__init__(self)
        self.num_endpoints = num_endpoints

        self.transformer = SentenceTransformer("sentence-transformers/all-distilroberta-v1")

        sentence_dim = self.transformer.get_sentence_embedding_dimension()
        self.ff1 = torch.nn.Linear(sentence_dim, sentence_dim).to('cpu')
        self.act1 = torch.nn.ReLU()



        self.embeddings = torch.nn.Embedding(
            num_embeddings=num_endpoints,
            embedding_dim=sentence_dim,
        ).to('cpu')

        if load_optimizer:
            self.load_optimizer()

        

    def forward(self, sequences, ids):

        assert len(sequences) == ids.shape[0]

        seq_embeddings = torch.tensor(self.transformer.encode(sequences)).to(self.device)
        seq_embeddings = self.ff1(seq_embeddings)
        seq_embeddings = self.act1(seq_embeddings)
        seq_embeddings = F.normalize(seq_embeddings, p=2, dim=1)
        seq_embeddings = seq_embeddings.reshape(seq_embeddings.shape[0], seq_embeddings.shape[1], 1)

        end_embeddings = self.embeddings(ids)
        end_embeddings = F.normalize(end_embeddings, p=2, dim=1)

        similarities = torch.bmm(end_embeddings, seq_embeddings)

        return torch.squeeze(similarities)

    @property
    def device(self):
        return next(self.parameters()).device
    def get_all_sims(self, sequence):

        seq_embeddings = torch.tensor(self.transformer.encode(sequence)).to(self.device)
        seq_embeddings = self.ff1(seq_embeddings)
        seq_embeddings = self.act1(seq_embeddings)
        seq_embeddings = F.normalize(seq_embeddings, p=2, dim=1)

        # (num_receptors, dim)
        endpoint_embeddings = self.embeddings(torch.arange(0, self.num_endpoints).to(self.device))
        endpoint_embeddings = F.normalize(endpoint_embeddings, p=2, dim=1)

        # (batch_size, num_endpoints)
        sims = torch.matmul(seq_embeddings, endpoint_embeddings.T)
        sims = (sims + 1) / 2  # bound from (0, 1)

        return sims


    def load_optimizer(self, **kwargs):
        # optimizer_kwargs.update(kwargs)
        # optimizer_kwargs.update(self.config.get('optimizer', {}))
        # optim_class = self.import_object(optimizer_kwargs['path'])
        # self.optimizer = optim_class(self.model.parameters(),**optimizer_kwargs['params'])
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.00032, **kwargs)
