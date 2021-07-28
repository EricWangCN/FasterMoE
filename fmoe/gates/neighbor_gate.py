from .naive_gate import NaiveGate

import torch
import torch.nn.functional as F


class NeighborGate(NaiveGate):
    def __init__(self, d_model, n_expert, world_size, rank):
        super().__init__(d_model, n_expert, world_size, top_k=2)
        self.mask = []
        valid_workers = set([(rank - 1 + world_size) % world_size, rank, (rank + 1) % world_size])
        for i in range(n_expert * world_size):
            if (i // n_expert) not in valid_workers:
                self.mask.append(i)

    def forward(self, inp):
        gate = self.gate(inp)
        gate[:, self.mask] = gate.min() - 1
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        gate_score = F.softmax(gate_top_k_val, dim=-1)
        return gate_top_k_idx, gate_top_k_val


def gen_neighbor_gate(rank):
    def _gen(d_model, n_expert, world_size, top_k=2):
        assert top_k == 2
        return NeighborGate(d_model, n_expert, world_size, rank)
