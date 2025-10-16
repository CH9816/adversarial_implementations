import torch
from torch import Tensor, nn

from torchvision import transforms



cpu, gpu = torch.device("cpu"), torch.device("cuda")



def undo_data_norm(
        x : torch.Tensor,
      
        # imagenet
        mu = [0.485, 0.456, 0.406], 
        sigma = [0.229, 0.224, 0.225]
    ):

    device = x.device

    mu    = torch.tensor(mu)[None, :, None, None].to(device)
    sigma = torch.tensor(sigma)[None, :, None, None].to(device)

    return x * sigma + mu








def fsgm_attack(
        model : nn.Module,
        datapoint : torch.Tensor, groundTruth = None,
        lossf = torch.nn.functional.mse_loss,
        eps = .001,
        mu_sigma = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        device = cpu
    ):

    # get model and datapoints in the right state
    model.eval()
    model = model.to(device)
    datapoint = datapoint.to(device)
    # zero previous gradients
    model.zero_grad()
    datapoint.requires_grad = True
    # if ground truth not known, 
    # assume (maximised) model output is ground truth


    out = model(datapoint, raw_return = True)
    if groundTruth is None:
        #groundTruth = torch.ones_like(out) * 0  
        groundTruth = torch.zeros_like(out)
        groundTruth[0][torch.argmax(out)] = 1 #torch.max(out)
        

    # get gradient of datapoint w.r.t. loss
    loss = lossf(out, groundTruth.to(device))
    loss.backward()
    gradient = datapoint.grad.data


    # adversarial = original + eps * sign(grad)
    adv = undo_data_norm(datapoint, *mu_sigma) + eps * gradient.sign()


    # re normalise and clamp data so that model interacts with it correctly
    adv = transforms.Normalize(*mu_sigma)(adv)
    adv = torch.clamp(adv, 0, 1)

    return adv


