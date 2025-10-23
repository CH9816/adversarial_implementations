import torchvision
from torchvision.models import vgg16, VGG16_Weights

import torch
from torch import nn
from torch.nn.functional import softmax, interpolate

import json
from warnings import warn as RaiseWarning


cpu, gpu = torch.device("cpu"), torch.device("cuda")



# vgg that prints out top and bottom k classes
# and automatically resizes and batches
class VGG16_easy(nn.Module):
    def __init__(self, vgg16_ : vgg16, returnK = 10, path = "", device = cpu):

        super().__init__()

        self.model = vgg16_
        self.model.eval()

        self.path = path
        self.K = returnK

        self.inSize = 224
        
        #with open(self.path + r"/imagenet_labels.json", "r") as f:
        with open("imagenet_labels.json", "r") as f:
            self.classJson = json.loads(f.read())

        self.device = device

    def forward(self, x : torch.Tensor, raw_return = False):

        x = x.to(self.device)

        # if input is not batched, resize so that it is the right
        # shape for the model
        if len(list(x.shape)) < 4:
            x = x.unsqueeze(0)

        # interpolate if image size is wrong
        if size := list(x.shape)[2] != self.inSize:
            x = interpolate(x, self.inSize)

            if size < self.inSize:
                RaiseWarning("image size passed to vgg16 < 224, interpolating upwards may break model")

        out = self.model(x)#.detach()

        if raw_return:
            
            return softmax(out, dim = 1)

        out = out.detach()

        out_soft = softmax(out, dim = 1)

        top_k_indices = torch.topk(out, self.K, dim=1)[1][0]
        least_k_indices = torch.topk(out, self.K, dim=1, largest=False)[1][0]

        outString = ''.join(
            
            # top K
            [f"top {self.K} predictions;\n\n"] + [
            
                f'- ({i}) {self.classJson[str(i.item())]}, {round(float(out_soft[0][i] * 100), 2)}%\n'
                    for i in top_k_indices
             
            ] + ["\n"] #+

            # bottom K
            #[f"bottom {self.K} predictions;\n\n"] + [
            #    f'- ({i}) {self.classJson[str(i.item())]}, {round(float(out_soft[0][i] * 100), 4)}%\n'
            #        for i in least_k_indices
            #]
            
            
            )


        return outString




def get_vgg16(pretrained = True, device = cpu):

    model = vgg16(weights = VGG16_Weights.IMAGENET1K_V1) \
                if pretrained else vgg16()
                
    easy_model = VGG16_easy(model, 20, device=device)

    return easy_model.to(device)