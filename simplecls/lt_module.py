import pytorch_lightning as L

from simplecls.torch_utils import compute_accuracy


class LightningModelWrapper(L.LightningModule):
    def __init__(self, model, optimizer, loss, scheduler):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = self.loss(output, target.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        pred_cats = self(inputs)
        top1 = compute_accuracy(pred_cats, target, reduce_mean=True)
        val_loss = self.loss(pred_cats, target).item()
        self.log_dict({'val_loss': val_loss, 'val_acc': top1})

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler }