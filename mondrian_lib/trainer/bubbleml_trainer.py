import torch
import torch.nn.functional as F
import einops
import lightning as L

from mondrian_lib.metrics import Metrics

class BubbleMLModule(L.LightningModule):
    def __init__(self, model, total_iters, loss_func=F.mse_loss):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.embedding = torch.nn.Embedding(
                num_embeddings=502,
                embedding_dim=32 * 32,
                padding_idx=0)
        self.loss_func = loss_func
        self.total_iters = total_iters
        self.metrics = Metrics(self.log)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=self.total_iters)
        return [optimizer], [scheduler]

    def forward(self, x, nuc_indices):
        r"""
        Args:
            x: preprocessed input (temperature, velx, velx, etc).
               This is laid out [B x C x H x W]
            nuc_indices: the nucleation sites are floats in the range [-2.5, 2.5]
                         in order to input them to the model, the CALLER should map this range
                         (approximately) to [1, 501]. This preserves two decimals of
                         precision, which should be sufficient.
                         Index 0 is used as a "padding" index, so it will not contribute.
                         This is [B x L]. Where L is the length of the padded sequence. 
        """
        assert nuc_indices.max() < 502
        assert nuc_indices.min() >= 0
        assert x.size(0) == nuc_indices.size(0)
        assert x.dim() == 4
        assert nuc_indices.dim() == 2

        # [B x padded_num_sites x embedding_dim]
        nuc_embeddings = self.embedding(nuc_indices)

        # hopefully, summing the embeddings preserves enough info.
        # If the embedding dim is large enough resolution, it should...
        # [B, 32, 32]
        nuc_vec_embedding = torch.unflatten(nuc_embeddings.sum(1), dim=1, size=(32, 32))
        # [B, 1, H, W]
        nuc = F.interpolate(nuc_vec_embedding.unsqueeze(1),
                            size=(x.size(2), x.size(3)),
                            mode='bilinear')

        # [B x (C_in + embedding_dim) x H x W]
        input = torch.cat((x, nuc), dim=1)

        return self.model(input)

    def training_step(self, batch, batch_idx):
        x, nuc_indices, y = batch
        pred = self.forward(x, nuc_indices)
        loss = self.metrics.log(pred, y, 'Train')
        return loss

    def validation_step(self, batch, batch_idx):
        x, nuc_indices, y = batch
        pred = self.forward(x, nuc_indices)
        loss = self.metrics.log(pred, y, 'Val')
        return loss

    def test_step(self, batch, batch_idx):
        x, nuc_indices, y = batch
        pred = self.forward(x, nuc_indices)
        loss = self.metrics.log(pred, y, 'Test')
        return loss
