import torch as t
from trainer import Trainer
import sys
import torchvision as tv
import model

epoch = int(sys.argv[1])
#TODO: Enter your model here
# model = t.nn.ResNet(
#     # Conv2D(in channels, out channels, filter size, stride)
#     t.nn.Conv2D(3, 64, 7, 2),
#     t.nn.BatchNorm(),
#     t.nn.ReLU(),
#     # MaxPool(pool size, stride)
#     t.nn.MaxPool(3, 2),
#     # ResBlock(in channels, out channels, stride)
#     t.nn.ResBlock(64, 64, 1),
#     t.nn.ResBlock(64, 128, 2),
#     t.nn.ResBlock(128, 256, 2),
#     t.nn.ResBlock(256, 512, 2),
#     t.nn.GlobalAvgPool(),
#     t.nn.Flatten(),
#     # FC(in features, out features)
#     t.nn.FC(512, 2),
#     t.nn.Sigmoid(),
# )

model = model.ResNet()

crit = t.nn.BCELoss()
trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
