import torch
from dataclasses import dataclass
from tqdm import tqdm

from .torch_utils import put_on_device


@dataclass
class Quantizer:
    model: torch.nn.Module
    val_loader: object
    cfg: dict
    max_batches: int
    writer: object = None
    device: str = 'cuda'
    debug: bool = False
    half_precision: bool = False

    @torch.no_grad()
    def run(self):
        '''Runs quantization on the model using the validation dataset.'''

        self.model.eval()
        self.model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        model_fp32_fused = self.model
        #model_fp32_fused = torch.ao.quantization.fuse_modules(self.model, [['conv', 'bn', 'relu']])
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)
        loop = tqdm(enumerate(self.val_loader), desc='Quantizing model')

        for batch_idx, (imgs, gt_cats) in loop:
            if batch_idx >= self.max_batches:
                break
            imgs, gt_cats = put_on_device([imgs, gt_cats], self.device)
            model_fp32_prepared(imgs)

        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
        print("Quantization completed.")

        return model_int8