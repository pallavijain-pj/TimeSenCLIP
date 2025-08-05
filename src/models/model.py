import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional

from .pooling import AttentionPoolPerImage, AttentionPoolPerDimension


def concat_all_gather(tensor):
    """Gathers tensors from all processes and concatenates them."""
    world_size = torch.distributed.get_world_size()
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


class CrossViewModel(nn.Module):
    """
    CrossViewModel Contrastive Learning Model

    Args:
        embed_dim (int): Embedding dimension.
        pooling (str): Pooling strategy ('avgpool', 'attpool_perimage', 'attpool_perdim').
        device (int): CUDA device index.
        queue_size (int): Size of the negative sample queue.
        queue_data (Iterable): Dataset or dataloader to initialize the queue.
        tsvit_model (nn.Module): Backbone model for temporal-spectral vision.
        pool_out (str): Output strategy for attention pooling ('sum' or others).
    """
    def __init__(
        self,
        embed_dim: int = 512,
        pooling: str = 'avgpool',
        device: int = 0,
        queue_size: int = 2048,
        queue_data: Optional[torch.utils.data.DataLoader] = None,
        tsvit_model: Optional[nn.Module] = None,
        pool_out: str = 'sum',
    ):
        super().__init__()

        self.pooling = pooling
        self.pool_out = pool_out
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

        print(f"[INFO] Using pooling method: {pooling}")

        self.TS_ViT = tsvit_model
        self.pooling_layer = self._init_pooling_layer(embed_dim)

        self.K = queue_size
        self.register_buffer("queue", F.normalize(torch.randn(embed_dim, self.K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.queue_data = queue_data

    def _init_pooling_layer(self, embed_dim):
        if self.pooling == 'attpool_perimage':
            return AttentionPoolPerImage(embed_dim, out=self.pool_out).to(self.device)
        elif self.pooling == 'attpool_perdim':
            return AttentionPoolPerDimension(embed_dim).to(self.device)
        else:
            return nn.AdaptiveAvgPool1d(1)  # For avgpool

    @torch.no_grad()
    def fill_queue(self):
        """
        Pre-fills the memory queue with embeddings from the provided dataset.
        """
        print("[INFO] Filling queue with initial embeddings...")
        for i, batch in tqdm(enumerate(self.queue_data), desc="Queue Init"):
            ground_embeddings = self._apply_pooling(batch)
            self._dequeue_and_enqueue(ground_embeddings)
            if i >= self.K:
                break
        print("[INFO] Queue initialized.")

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """
        Enqueue new embeddings and dequeue the oldest to maintain queue size.
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        assert self.K % batch_size == 0, f"Queue size ({self.K}) must be divisible by batch size ({batch_size})"

        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def _apply_pooling(self, y_emb: torch.Tensor) -> torch.Tensor:
        """
        Applies the selected pooling strategy on the input embeddings.
        """
        if self.pooling == 'avgpool':
            return self.pooling_layer(y_emb.permute(0, 2, 1)).squeeze(-1)
        else:
            return self.pooling_layer(y_emb).squeeze(-1)

    def forward(self, data):
        """
        Forward pass for contrastive training.

        Args:
            data (Tuple[torch.Tensor, torch.Tensor]): Tuple of (ground_embeddings, ts_images)

        Returns:
            logits (Tensor): Similarity scores (N, 1 + K)
            labels (Tensor): Ground truth labels (N,)
        """
        with torch.cuda.amp.autocast():
            ground_emb, ts_img = data  # y_emb, ts_img

            # Project and normalize embeddings
            ground_emb = self._apply_pooling(ground_emb)
            ts_emb = self.TS_ViT(ts_img)

            ground_emb = F.normalize(ground_emb, p=2, dim=-1)
            ts_emb = F.normalize(ts_emb, p=2, dim=-1)

            # Positive logits (Nx1)
            l_pos = torch.einsum("nc,nc->n", [ts_emb, ground_emb]).unsqueeze(-1)

            # Negative logits (NxK)
            l_neg = torch.einsum("nc,ck->nk", [ts_emb, self.queue.clone().detach()])

            # Combine logits
            logits = torch.cat([l_pos, l_neg], dim=1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)

            # Update the queue
            with torch.no_grad():
                self._dequeue_and_enqueue(ground_emb)

        return logits, labels
