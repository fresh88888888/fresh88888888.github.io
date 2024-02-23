---
title: PyTorch Lightning
date: 2024-02-23 08:34:32
tags:
  - AI
categories:
  - 人工智能
---

{% asset_img Lightning_AI.png %}

`PyTorch Lightning` 是专业人工智能研究人员和机器学习工程师的深度学习框架。是一个`batteries included`的深度学习框架，适合需要最大灵活性同时大规模增强性能的专业人工智能研究人员和机器学习工程师。
<!-- more -->

##### 1. PyTorch Lightning安装

```bash
$ pip install lightning
```
##### 2. 定义一个LightningModule

```python
import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)
```
##### 3. 定义一个数据集

```python
# setup data
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)
```

##### 4. 训练模型

```python
# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
```
执行训练：
{% asset_img lightning_train_model.png %}

##### 5.使用训练好的模型

```python
# load checkpoint
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

# choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

# embed 4 fake images!
fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)
```
输出结果为：
```bash
(hello-D1UArRDQ-py3.11) umbrella:hello zcj$ poetry run python Lightning_module_demo.py 
⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡ 
Predictions (4 image embeddings) :
 tensor([[-0.5218, -0.0958,  0.4148],
        [-0.6634, -0.0083,  0.5347],
        [-0.6266,  0.0502,  0.4794],
        [-0.6974, -0.0774,  0.5666]], grad_fn=<AddmmBackward0>) 
 ⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡
```

##### 6.训练可视化

`If you have tensorboard installed, you can use it for visualizing experiments. Run this on your commandline and open your browser to http://localhost:6006/`

```bash
$ tensorboard --logdir .
```
##### 7.增压训练

使用`Trainer`参数启用高级训练功能。这些是最先进的技术，可以自动集成到您的训练循环中，而无需更改您的代码。

```python
# train on 4 GPUs
trainer = L.Trainer(
    devices=4,
    accelerator="gpu",
 )

# train 1TB+ parameter models with Deepspeed/fsdp
trainer = L.Trainer(
    devices=4,
    accelerator="gpu",
    strategy="deepspeed_stage_2",
    precision=16
 )

# 20+ helpful flags for rapid idea iteration
trainer = L.Trainer(
    max_epochs=10,
    min_epochs=5,
    overfit_batches=1
 )

# access the latest state of the art techniques
trainer = L.Trainer(callbacks=[StochasticWeightAveraging(...)])
```
`Lightning`的核心指导原则是始终提供最大的灵活性，而不隐藏任何`PyTorch`。根据项目的复杂性，`Lightning`提供`5`种额外的灵活性。