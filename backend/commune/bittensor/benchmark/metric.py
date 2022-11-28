

def causal_lm_loss(labels, logits):

    batch_size = logits.shape[0]
    loss_fct = CrossEntropyLoss()

    losses = []
    for batch in range(batch_size):
        shift_logits = logits[batch, :-1, :].contiguous()
        shift_labels = labels[batch, 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, 50258), shift_labels.view(-1))
        losses.append(loss)
    return torch.tensor(losses)


def ranknet_loss(s1, s2, t):
    o = torch.sigmoid(s1 - s2)
    loss = (-t * o + F.softplus(o)).mean()
    return loss