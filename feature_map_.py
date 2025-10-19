import torch
from torch import Tensor, nn

from torchvision import transforms
from torch_common import undo_data_norm



cpu, gpu = torch.device("cpu"), torch.device("cuda")




def zero_less_than(x : Tensor, val):
    x.apply_(lambda e : 0 if abs(e) < val else e)


def get_gradient_image_fm(
        model : nn.Module,
        datapoint : torch.Tensor, desired : Tensor,
        lossf = torch.nn.functional.mse_loss,
        device = torch.device("cuda")
    ):
    model.eval()
    model = model.to(device)
    datapoint = datapoint.to(device)
    # zero previous gradients
    model.zero_grad()

    x = datapoint#.clone()
    x.requires_grad = True


    grad = []
    def get_hook():
        def hook(_, inp, __):

            #with open("fm1.pt", "wb") as f:
            #    torch.save(inp[0], f)
            #quit()



            l = lossf(inp[0], desired if desired is not None else 0 * inp[0])
            l.backward()
            grad.append(x.grad.data)
        return hook

    k = 1
    hook = list(model.model.modules())[-k].register_forward_hook(get_hook())
    
    model(x)

    hook.remove()

    return grad[-1].to(device)


def feature_map_attack(
        model : nn.Module,
        datapoint : torch.Tensor, groundTruth = None,
        lossf = torch.nn.functional.mse_loss,
        eps = .01,
        # data normalisation parameters
        # default for imagenet
        mu_sigma = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        returnNoiseAdded = False,
        device = cpu

    ) -> torch.Tensor:

    datapoint = datapoint.to(device)


    #desired = torch.load("fm.pt")
    desired = None

    gradient = get_gradient_image_fm(model, datapoint, desired, lossf, device)


    # adversarial = original + eps * sign(grad)
    noise = eps * -gradient
    zero_less_than(noise, noise.abs().mean())

    adv = undo_data_norm(datapoint, *mu_sigma) + noise

    # re normalise and clamp data so that model interacts with it correctly
    adv = transforms.Normalize(*mu_sigma)(adv)
    adv = torch.clamp(adv, 0, 1)

    #noise = transforms.Normalize(*mu_sigma)(noise)
    #noise = torch.clamp(noise, 0, 1)

    #hook.remove()

    return adv if not returnNoiseAdded else (adv, gradient.sign())



    













def feature_map_attack_repeated(
        model : nn.Module,
        datapoint : torch.Tensor, groundTruth = None,
        lossf = torch.nn.functional.mse_loss,
        eps = .01,
        # data normalisation parameters
        # default for imagenet
        mu_sigma = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        returnNoiseAdded = False, Ns = 10,
        device = cpu

    ) -> torch.Tensor:

    datapoint = datapoint.to(device)


    desired = torch.load("pufferfish_fm_k1.pt")


    for n in range(Ns):
        print(n)
        gradient = get_gradient_image_fm(model, datapoint.clone(), desired, lossf, device)

        noise = eps * -gradient
        zero_less_than(noise, noise.abs().mean())

        datapoint = undo_data_norm(datapoint, *mu_sigma) if n == 0 else datapoint + noise

    # re normalise and clamp data so that model interacts with it correctly
    adv = transforms.Normalize(*mu_sigma)(datapoint)
    adv = torch.clamp(adv, 0, 1)

    #noise = transforms.Normalize(*mu_sigma)(noise)
    #noise = torch.clamp(noise, 0, 1)

    #hook.remove()

    return adv if not returnNoiseAdded else (adv, gradient.sign())




























