import torch
import numpy
import torchvision
from torchvision import transforms, datasets

# train =datasets.MNIST("",train=True,download=True,transform=transforms.Compose([transforms.ToTensor]))
# test = datasets.MNIST("",train=False,download=True,transform=transforms.Compose([transforms.ToTensor]))
#
# trainset = torch.utils.data.Dataloader(train,batch_size=10,shuffle=True)
# testset = torch.utils.data.Dataloader(test,batch_size=10,shuffle=True)

print(numpy.__version__)
print(torch.cuda.is_available())
torch.version.cuda