from typing import Tuple

import torch
import torch.nn as nn
from umbc.models.layers.bigset import OT, T
from umbc.models.layers.sse import SlotSetEncoder
from umbc.models.layers.set_xformer import SAB, PMA
from transformers import BertConfig, BertForSequenceClassification, BertModel


class Encoder(nn.Module):
    def __init__(self, embedding=None, num_layers=2):
        super().__init__()
        if embedding is None:
            bert = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased")
            embedding = bert.get_input_embeddings()
        padding_idx = embedding.padding_idx
        h = embedding.weight.size(1)
        self.hidden_dim = h
        self.embedding = nn.Embedding.from_pretrained(embedding.weight,
                                                      padding_idx=padding_idx)

        layers = [nn.LayerNorm(h)]
        for _ in range(num_layers):
            layers.append(nn.Linear(h, h))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        return self.layers(emb)

    def get_cls_emb(self, n):
        device = next(self.parameters()).device
        cls_id = torch.tensor([101]).to(device)
        cls_emb = self.embedding(cls_id).repeat(n, 1)
        return cls_emb


class UMBC(nn.Module):
    def __init__(self,
                 num_layers: int = 2,
                 position: bool = False,
                 num_classes: int = 227,
                 d_dim: int = 768,        # hidden dim for encoder
                 h_dim: int = 128,        # slot dim
                 d_hat: int = 768,        # dim after q,k,v
                 K: int = 256,
                 heads: int = 1,
                 ln_slots: bool = False,
                 ln_after: bool = True,
                 slot_type: str = "random",
                 slot_drop: float = 0.0,
                 attn_act: str = "sigmoid-slot",
                 slot_residual: bool = False):
        super().__init__()
        self.pooler = SlotSetEncoder(
            K=K, h=h_dim, d=d_dim, d_hat=d_hat, slot_type=slot_type,
            ln_slots=ln_slots, heads=heads, slot_drop=slot_drop,
            attn_act=attn_act, slot_residual=slot_residual, ln_after=ln_after
        )
        self.name = f"text/position_{position}/" + \
            self.pooler.name + f"_heads_{heads}_K_{K}_{attn_act}"

        self.hidden_dim = d_dim
        self.position = position

        if position:
            position_embedding_type = "absolute"
        else:
            position_embedding_type = None
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.update({
            "num_labels": num_classes,
            "problem_type": "multi_label_classification",
            "position_embedding_type": position_embedding_type
        })
        self.decoder = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", config=config)
        embed = self.decoder.get_input_embeddings()
        self.encoder = Encoder(embed, num_layers)

    def forward(
        self,
        input_ids: T,
        input_mask: T,
        labels: OT = None,
    ) -> Tuple[torch.Tensor]:
        h = self.encoder(input_ids)
        rep = self.pooler(h, mask=input_mask)
        b = rep.size(0)

        cls_emb = self.encoder.get_cls_emb(b).to(input_ids.device)
        rep = torch.cat([cls_emb.unsqueeze(1), rep], dim=1)
        outputs = self.decoder(inputs_embeds=rep, labels=labels)

        return outputs

    def forward_mbc(
        self,
        input_ids: T,
        input_mask: T,
        split_size: int,
        labels: OT = None,
    ) -> Tuple[torch.Tensor]:
        self.pooler.pre_forward_mbc()  # sample slots

        c_ids = torch.split(input_ids, split_size, dim=1)
        c_input_mask = torch.split(input_mask, split_size, dim=1)

        num_chunks = len(c_ids)
        x_t, c_t = None, None
        for idx in range(num_chunks):
            mask = c_input_mask[idx]
            if idx == 0:
                c_h = self.encoder(c_ids[idx])
                x_t, c_t = self.pooler.forward_mbc(
                    c_h, X_prev=x_t, c_prev=c_t,
                    grad=True, mask=mask
                )
            else:
                with torch.no_grad():
                    c_h = self.encoder(c_ids[idx])
                x_t, c_t = self.pooler.forward_mbc(
                    c_h, X_prev=x_t, c_prev=c_t,
                    grad=False, mask=mask
                )
        rep = self.pooler.post_forward_mbc(x_t, c=c_t)
        b = input_ids.size(0)

        cls_emb = self.encoder.get_cls_emb(b).to(input_ids.device)
        rep = torch.cat([cls_emb.unsqueeze(1), rep], dim=1)

        outputs = self.decoder(inputs_embeds=rep, labels=labels)

        return outputs


class BoW(nn.Module):
    def __init__(self,
                 embedding: T = None,
                 num_layers: int = 2,
                 num_labels: int = 4271) -> None:
        super().__init__()
        self.name = "bow_classifier"
        self.encoder = Encoder(embedding, num_layers)
        h = self.encoder.hidden_dim
        self.decoder = nn.Sequential(nn.Linear(h, h),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(h, h),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(h, num_labels))

    def forward(self,
                input_ids: T = None,
                input_mask: T = None,
                labels: T = None,
                ) -> Tuple[T]:
        rep = self.encoder(input_ids)
        rep = rep.masked_fill(input_mask.unsqueeze(-1) == 0, 0.0)

        # avg pooling
        lengths = torch.sum(input_mask, dim=1, keepdim=True).float()
        pooled = torch.sum(rep, dim=1) / lengths

        logits = self.decoder(pooled)
        criterion = nn.BCEWithLogitsLoss()
        if labels is not None:
            loss = criterion(logits, labels.float())
            return loss, logits

        else:
            return logits,

    def forward_mbc(
        self,
        input_ids: T,
        input_mask: T,
        split_size: int,
        labels: OT = None,
    ) -> Tuple[torch.Tensor]:
        c_ids = torch.split(input_ids, split_size, dim=1)
        c_input_mask = torch.split(input_mask, split_size, dim=1)
        num_chunks = len(c_ids)
        rep = 0
        for idx in range(num_chunks):
            c_id = c_ids[idx]
            mask = c_input_mask[idx]
            if idx == 0:
                h = self.encoder(c_id)
                h = h.masked_fill(mask.unsqueeze(-1) == 0, 0.0)
                h = torch.sum(h, 1)
            else:
                with torch.no_grad():
                    h = self.encoder(c_id)
                    h = h.masked_fill(mask.unsqueeze(-1) == 0, 0.0)
                    h = torch.sum(h, 1)
            rep = rep + h
        lengths = torch.sum(input_mask, dim=1, keepdim=True).float()
        pooled = rep / lengths
        logits = self.decoder(pooled)
        criterion = nn.BCEWithLogitsLoss()
        if labels is not None:
            loss = criterion(logits, labels.float())
            return loss, logits

        else:
            return logits,


class MBC(nn.Module):
    def __init__(self,
                 num_layers: int = 2,
                 position: bool = False,
                 num_classes: int = 227,
                 d_dim: int = 768,        # hidden dim for encoder
                 h_dim: int = 128,        # slot dim
                 d_hat: int = 768,        # dim after q,k,v
                 K: int = 1,
                 heads: int = 1,
                 ln_slots: bool = True,
                 ln_after: bool = True,
                 slot_type: str = "random",
                 slot_drop: float = 0.0,
                 attn_act: str = "sigmoid-slot",
                 add_st: bool = False):
        super().__init__()
        self.pooler = SlotSetEncoder(
            K=K, h=h_dim, d=d_dim, d_hat=d_hat, slot_type=slot_type,
            ln_slots=ln_slots, heads=heads, slot_drop=slot_drop,
            attn_act=attn_act, slot_residual=False, ln_after=ln_after
        )
        if add_st:
            assert K > 1
            self.set_trans = nn.Sequential(
                SAB(dim_in=d_hat, dim_out=d_hat, num_heads=heads, ln=True),
                SAB(dim_in=d_hat, dim_out=d_hat, num_heads=heads, ln=True),
                PMA(dim=d_hat, num_heads=heads, num_seeds=1, ln=True))
        else:
            self.set_trans = None
        prefix = "SSE_SET" if add_st else "SSE"
        self.name = f"{prefix}_" + \
            self.pooler.name + f"_heads_{heads}_K_{K}_{attn_act}"

        self.hidden_dim = d_dim
        self.position = position

        bert = BertModel.from_pretrained("bert-base-uncased")
        embed = bert.get_input_embeddings()
        self.encoder = Encoder(embed, num_layers)

        self.decoder = nn.Sequential(nn.Linear(d_hat, d_hat),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(d_hat, d_hat),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(d_hat, num_classes))

    def forward(
        self,
        input_ids: T,
        input_mask: T,
        labels: OT = None,
    ) -> Tuple[torch.Tensor]:
        h = self.encoder(input_ids)
        rep = self.pooler(h, mask=input_mask)

        if self.set_trans is not None:
            rep = self.set_trans(rep).squeeze(1)
        else:
            rep = rep.squeeze(1)
        logits = self.decoder(rep)

        criterion = nn.BCEWithLogitsLoss()
        if labels is not None:
            loss = criterion(logits, labels.float())
            return loss, logits

        else:
            return logits,

    def forward_mbc(
        self,
        input_ids: T,
        input_mask: T,
        split_size: int,
        labels: OT = None,
    ) -> Tuple[torch.Tensor]:
        self.pooler.pre_forward_mbc()  # sample slots

        c_ids = torch.split(input_ids, split_size, dim=1)
        c_input_mask = torch.split(input_mask, split_size, dim=1)

        num_chunks = len(c_ids)
        x_t, c_t = None, None
        for idx in range(num_chunks):
            mask = c_input_mask[idx]
            if idx == 0:
                c_h = self.encoder(c_ids[idx])
                x_t, c_t = self.pooler.forward_mbc(
                    c_h, X_prev=x_t, c_prev=c_t,
                    grad=True, mask=mask
                )
            else:
                with torch.no_grad():
                    c_h = self.encoder(c_ids[idx])
                x_t, c_t = self.pooler.forward_mbc(
                    c_h, X_prev=x_t, c_prev=c_t,
                    grad=False, mask=mask
                )
        rep = self.pooler.post_forward_mbc(x_t, c=c_t)
        if self.set_trans is not None:
            rep = self.set_trans(rep).squeeze(1)
        else:
            rep = rep.squeeze(1)
        logits = self.decoder(rep)

        criterion = nn.BCEWithLogitsLoss()
        if labels is not None:
            loss = criterion(logits, labels.float())
            return loss, logits

        else:
            return logits,
