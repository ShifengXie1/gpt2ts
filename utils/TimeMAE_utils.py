import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class CE:
    def __init__(self, model):
        self.model = model
        self.ce = nn.CrossEntropyLoss()
        self.ce_pretrain = nn.CrossEntropyLoss(ignore_index=0)

    def compute(self, batch):
        seqs, labels = batch
        outputs = self.model(seqs)
        labels = labels.view(-1).long()
        return self.ce(outputs, labels)


class Align:
    def __init__(self):
        self.mse = nn.MSELoss(reduction="mean")
        self.ce = nn.CrossEntropyLoss()

    def compute(self, rep_mask, rep_mask_prediction):
        return self.mse(rep_mask, rep_mask_prediction)


class Reconstruct:
    def __init__(self):
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.2)

    def compute(self, token_prediction_prob, tokens):
        hits = torch.sum(torch.argmax(token_prediction_prob, dim=-1) == tokens)
        ndcg10 = recalls_and_ndcgs_for_ks(
            token_prediction_prob.view(-1, token_prediction_prob.shape[-1]),
            tokens.reshape(-1, 1),
            10,
        )
        reconstruct_loss = self.ce(
            token_prediction_prob.view(-1, token_prediction_prob.shape[-1]),
            tokens.view(-1),
        )
        return reconstruct_loss, hits, ndcg10


def recalls_and_ndcgs_for_ks(scores, answers, k):
    answers = answers.tolist()
    labels = torch.zeros_like(scores).to(scores.device)
    for i in range(len(answers)):
        labels[i][answers[i]] = 1
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels_float.gather(1, cut)
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits * weights.to(hits.device)).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
    return (dcg / idcg).mean().cpu().item()


def get_rep_with_label(model, dataloader):
    reps = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            seq, label = batch
            seq = seq.to(model.device)
            labels += label.cpu().numpy().tolist()
            rep = model(seq)
            reps += rep.cpu().numpy().tolist()
    return reps, labels


def fit_lr(features, y):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=3407,
            max_iter=1000000,
            multi_class="ovr",
        ),
    )
    pipe.fit(features, y)
    return pipe
