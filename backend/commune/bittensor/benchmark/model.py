import torch
from sentence_transformers import SentenceTransformer

class RankingLoss(nn.Module):
    def __init__(self):
        super(RankingLoss, self).__init__()

    def forward(self, x, y):
        print(self)
        loss = torch.mean((x - y) ** 2)
        return loss
class RankingModel(nn.Module):
    def __init__(self, num_endpoints: int):

        super().__init__()
        self.num_endpoints = num_endpoints

        self.transformer = SentenceTransformer(
            "sentence-transformers/all-distilroberta-v1"
        )

        # TODO match embedding dim to transformer
        self.embeddings = torch.nn.Embedding(
            num_embeddings=num_endpoints,
            embedding_dim=self.transformer.get_sentence_embedding_dimension(),
        )

    def forward(self, sequence):

        seq_embeddings = torch.tensor(self.transformer.encode(sequence))

        # (num_receptors, dim)
        endpoint_embeddings = self.embeddings(torch.arange(0, self.num_endpoints))
        endpoint_embeddings = torch.nn.functional.normalize(endpoint_embeddings, p=2, dim=1)

        # (batch_size, num_endpoints)
        sims = torch.matmul(seq_embeddings, endpoint_embeddings.T)
        sims = (sims + 1) / 2  # bound from (0, 1)

        return sims

