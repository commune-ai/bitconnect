from collections import defaultdict
from random import seed, shuffle
from scipy.stats import kendalltau
import bittensor
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
import torchsort

from model import RankingModel


# https://github.com/huggingface/transformers/blob/v4.21.3/src/transformers/models/gptj/modeling_gptj.py#L847


bittensor.logging(debug=False)
data = bittensor.dataset(block_size=128)
tokenizer = data.tokenizer
x = 64
graph = bittensor.metagraph().sync()
wallet = bittensor.wallet(name="const", hotkey="Tiberius")
seed(0)
shuffle(endpoints)

model = RankingModel(len(endpoints))
optimizer = torch.optim.Adam(model.parameters(), lr=0.00032)
synapses = [bittensor.synapse.TextCausalLM()]

perf_df = pd.DataFrame()
for step in range(1000):
    print("getting next batch of data")
    inputs = next(data)
    str_inputs = [tokenizer.decode(s) for s in inputs]

    print(f"Querying endpoints")
    rpool = bittensor.receptor_pool(
        wallet=wallet, max_worker_threads=64, max_active_receptors=len(endpoints)
    )
    results = rpool.forward(
        endpoints, synapses=synapses, inputs=[inputs] * len(endpoints), timeout=20
    )
    del rpool
    tensors = []
    for tensor in results[0]:
        tensors.append(tensor[0])

    codes = []
    codes_count = defaultdict(int)
    for code in results[1]:
        code = code[0]
        codes.append(code)
        codes_count[code] += 1
    for code in sorted(set(codes)):
        print(f"{code}: {codes_count[code]}")
    print()

    print("Calculating losses for each endpoint")
    all_losses = []
    for _, logits in tqdm(enumerate(tensors)):
        all_losses.append(causal_lm_loss(inputs, logits))

    all_losses_tensor = torch.vstack(all_losses).T  # (batch_size, num_endpoints)
    ideal_rankings = torch.argsort(torch.argsort(all_losses_tensor, axis=1, descending=False), axis=1)

    all_sims = model.get_all_sims(str_inputs)
    model_rankings = torch.argsort(torch.argsort(all_sims, axis=1, descending=True), axis=1)

    x1 = [[] for _ in range(all_losses_tensor.shape[0])]
    x2 = [[] for _ in range(all_losses_tensor.shape[0])]
    ys = [[] for _ in range(all_losses_tensor.shape[0])]
    for batch in range(all_losses_tensor.shape[0]):
        for idx in range(all_losses_tensor.shape[1]):
            for idx2 in range(all_losses_tensor.shape[1]):
                # TODO: Contrastive sampling improvements
                # while len(x1[batch]) != 10:
                # idx2 = randint(0, all_losses_tensor.shape[1] - 1)
                if idx == idx2:
                    continue
                d = all_losses_tensor[batch][idx] - all_losses_tensor[batch][idx2]
                t = (
                    1.0
                    if all_losses_tensor[batch][idx] < all_losses_tensor[batch][idx2]
                    else 0.0
                    if all_losses_tensor[batch][idx] > all_losses_tensor[batch][idx2]
                    else 0.5
                )
                x1[batch].append(idx)
                x2[batch].append(idx2)
                ys[batch].append(t)

    x1, x2, ys = torch.tensor(x1), torch.tensor(x2), torch.tensor(ys)
    print(f"Batch size: {x1.shape}")
    print("Model forward")
    s1 = model(str_inputs, x1)
    s2 = model(str_inputs, x2)
    print("model backwards")

    sorted_losses_tensor = all_losses_tensor.clone()
    sorted_by_model_rank = all_losses_tensor.clone()

    ideal_idx = torch.argsort(all_losses_tensor, axis=1, descending=False)
    model_idx = torch.argsort(all_sims, axis=1, descending=True)

    for i in range(sorted_by_model_rank.shape[0]):
        sorted_losses_tensor[i, :] = sorted_losses_tensor[i, ideal_idx[i]]
        sorted_by_model_rank[i, :] = sorted_by_model_rank[i, model_idx[i]]

    topk = 10
    ideal_softrank = torchsort.soft_rank(all_losses_tensor, regularization_strength=1e-6)
    model_softrank = torchsort.soft_rank(-all_sims, regularization_strength=1e-6)

    tau_softrank, _p = kendalltau(model_softrank.detach().numpy(), ideal_softrank.detach().numpy())
    tau_losses, _p = kendalltau(sorted_losses_tensor.detach().numpy(), sorted_by_model_rank.detach().numpy())

    tau_rank_topk, _p = kendalltau(model_softrank[:, :topk].detach().numpy(), ideal_softrank[:, :topk].detach().numpy())
    tau_loss_topk, _p = kendalltau(sorted_losses_tensor[:, :topk].detach().numpy(), sorted_by_model_rank[:, :topk].detach().numpy())

    loss = ranknet_loss(s1, s2, ys)
    ndcg = metrics.ndcg_score(1 / (ideal_rankings + 1), 1 / (model_rankings + 1))
    print(f"step: {step} | loss={loss.item():.5f} | {ndcg=:.3f} | {tau_rank_topk=:.3f}")
    _df = pd.DataFrame.from_dict(
        {"step": [step], "loss": [loss.item()], "ndcg": [ndcg], "tau_softrank": [tau_softrank],
         "tau_losses": [tau_losses], "tau_softrank_topk": [tau_rank_topk],
         "tau_loss_topk": [tau_loss_topk]}
    )
    perf_df = pd.concat((perf_df, _df), ignore_index=True)
    perf_df.to_csv(f"model_perf_{x}.csv")

    for inp in (0, -1):
        print(tokenizer.decode(inputs[inp]))

        for idx in range(20):
            i = (ideal_rankings[inp] == idx).nonzero()[0][0].item()
            modeli = (model_rankings[inp] == idx).nonzero()[0][0].item()

            lossm = all_losses_tensor[inp][modeli].item()
            rank = model_rankings[inp][i].item() + 1
            hkm = endpoints[modeli].hotkey
            namem = hkm[:10]
            sim = all_sims[inp][modeli].item()

            hk = endpoints[i].hotkey
            name = hk[:10]
            model_loss = all_losses_tensor[inp][i].item()
            loss_rank = ideal_rankings[inp][i].item() + 1

            print(
                f"{loss_rank:02} | {name:<30} | {model_loss:06.3f}      | {loss_rank:03} | {namem:<30} | {lossm:06.3f} | {sim}"
            )
        print(f"lr |name{' '*27} | loss        | mr | model_choice |")




    model.train()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
