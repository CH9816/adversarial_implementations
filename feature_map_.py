from math import log10
import torch
from torch import Tensor, nn

from torchvision import transforms
from torch_common import showtensor, undo_data_norm, topil

from time import sleep

cpu, gpu = torch.device("cpu"), torch.device("cuda")




def zero_less_than(x : Tensor, val):
    device = x.device
    x.cpu().apply_(lambda e : 0 if abs(e) < val else e).to(device)



EPSILON_DIV = 1e-6
STOP_K_BEFORE = 15


def feature_map_loss_rec(output : Tensor, target : Tensor):
    rec_abs_diff = 1. / (output - target + EPSILON_DIV).abs()
    return rec_abs_diff.mean()


def feature_map_loss_rec_exp_exaggerated(output : Tensor, target : Tensor, scale = 2):
    rec_abs_diff = 1. / (output - target + EPSILON_DIV).abs()
    exp_val = torch.exp(scale * rec_abs_diff) - 1.
    return exp_val.mean()




def get_gradient_image_fm_away_original(
        model : nn.Module,
        datapoint : Tensor, original : Tensor,
        div_lossf = feature_map_loss_rec_exp_exaggerated,
        device = torch.device("cuda")

    ):
    
    model = model.eval().to(device)
    datapoint, original = datapoint.to(device), original.to(device)
    model.zero_grad()
    x = datapoint
    x.requires_grad = True

    grad = []
    def get_hook():
        def hook(_, inp, __):
            l = div_lossf(inp[0], original.clone())
            l.backward()
            grad.append(x.grad.data)
        return hook

    k = STOP_K_BEFORE
    hook = list(model.model.modules())[-k].register_forward_hook(get_hook())
    model(x)
    hook.remove()
    return grad[-1].to(device)






def get_gradient_image_fm_towards_desired(
        model : nn.Module,
        datapoint : torch.Tensor, desired : Tensor,
        lossf = torch.nn.functional.l1_loss,
        device = torch.device("cuda")
    ):
    model.eval()
    model = model.to(device)
    datapoint = datapoint.to(device)
    desired = desired.to(device)# if desired is not None else None
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

            out = inp[0]
            #print(out.shape, desired.shape); quit()

            l = lossf(out, desired)
            l.backward()
            grad.append(x.grad.data)
        return hook

    k = STOP_K_BEFORE
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
        device = torch.device("cuda")

    ) -> torch.Tensor:

    datapoint = datapoint.to(device)


    desired = torch.load("fm_airship_100.pt")
    #desired = None

    gradient = get_gradient_image_fm_towards_desired(model, datapoint, desired, lossf, device)


    # adversarial = original + eps * sign(grad)
    noise = eps * -gradient
    #zero_less_than(noise, noise.abs().mean())

    adv = undo_data_norm(datapoint, *mu_sigma) + noise

    # re normalise and clamp data so that model interacts with it correctly
    adv = transforms.Normalize(*mu_sigma)(adv)
    adv = torch.clamp(adv, 0, 1)

    #noise = transforms.Normalize(*mu_sigma)(noise)
    #noise = torch.clamp(noise, 0, 1)

    #hook.remove()

    return adv if not returnNoiseAdded else (adv, gradient.sign())



    





def get_X_given_ans(model : nn.Module, ans, Xshape, lr=1e-3, minloss_pow10 = 20, device = gpu):

    X = torch.randn(Xshape)
    
    desired_out = torch.zeros_like(model(X, raw_return = True))
    desired_out[0][ans] = 1
    #print(desired_out); quit()

    loss = 1

    while loss > 10 ** -minloss_pow10:
        X = X.clone().detach()
        X.requires_grad = True
        out = model(X, raw_return = True)
        loss = torch.nn.functional.mse_loss(out, desired_out)
        loss.backward()
        gradient = X.grad.data.clone()
        X = X + -lr * gradient.sign()
        #print(log10(loss.item()))

    #showtensor(X)

    return X #get_feature_map(model, X)
















def get_feature_map(
        model : nn.Module,
        datapoint : Tensor
    ):
    
    original_feature_map = []
    k=STOP_K_BEFORE
    def get_hook():
        def hook(_, inp, __): original_feature_map.append(inp[0].clone().detach())
        return hook
    hook = list(model.model.modules())[-k].register_forward_hook(get_hook())
    model(datapoint)
    hook.remove()
    return original_feature_map[-1]




from lpips import LPIPS

perceptual_lossf = LPIPS(net='alex')



def get_gradient_magnitude_mask(model : nn.Module, datapoint : Tensor, desired_model_out : Tensor) -> Tensor:
    
    X = datapoint.clone().detach().requires_grad_(True)
    desired_model_out = desired_model_out.clone().detach()
    
    out = model(X, raw_return = True)
    #loss = torch.nn.functional.mse_loss(out, desired_model_out)
    loss = high_power_loss(out, desired_model_out)

    #print(out.shape, desired_model_out.shape); quit()

    loss.backward()

    grad = X.grad.data

    #grad[grad > .1] = 0
    

    mask = grad.abs() / grad.abs().max()

    #showtensor(mask); quit()

    #threshold = .1

    #mask[mask < threshold] = 0
    #mask[mask > .5] = 1

    #return torch.ones_like(mask)
    return mask


    #k=STOP_K_BEFORE
    #Xs = []

    #def get_hook():
    #    def hook(module, inp, outp): 
    #        inp[0] = inp[0].detach().requires_grad_(True)
    #        Xs.append(inp[0])
    #    return hook
    #hook = list(model.modules())[-k].register_forward_hook(get_hook())
    
    #X = datapoint.clone().detach().requires_grad_(False)

    #out = model(X)

    #loss = torch.nn.functional.mse_loss(out, desired_model_out)
    #loss.backward()


    #grad = X.grad.data

    #grad = grad.abs() / grad.max()

    
    #hook.remove()


    #return grad




def elem_multip__keeping_mag(first : Tensor, second : Tensor):

    mag = first.abs().mean().sqrt()
    out = torch.mul(first, second)
    mag_after = out.abs().mean().sqrt()

    eps = 1e-10
    
    out_scaled = out * (mag / (mag_after + eps))



    return out_scaled
    #return torch.mul(first, second)


def high_power_loss(inp : Tensor, target : Tensor) -> Tensor:
    return torch.nn.functional.mse_loss(inp, target).pow(2)


def feature_map_attack_repeated(
        model : nn.Module,
        datapoint : torch.Tensor, 

        #target = None,
        target = "target.pt",

        lossf = torch.nn.functional.mse_loss,
        #lossf = perceptual_lossf,

        eps = .01,
        # data normalisation parameters
        # default for imagenet
        mu_sigma = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        returnNoiseAdded = False, Ns = None,
        try_correct = True,
        device = gpu,
        data = None

    ) -> torch.Tensor:

    perceptual_lossf_ = perceptual_lossf.to(device)

    model = model.to(device).eval()
    out = model(datapoint, raw_return = True)
    original_out = out.clone()
    class_ = torch.argmax(out) 
    class_out = class_.clone()
    model_out = out.clone()

    if Ns == -1: Ns = 1_000_000_000

    datapoint = undo_data_norm(datapoint.to(device), *mu_sigma).to(device)
    datapoint_pre_adv = datapoint.clone()


    
    #desired = None

    original_feature_map = get_feature_map(model, datapoint)
    away_feature_map = original_feature_map.clone()
    
    desired_out = None

    if target is not None:
        desired_img = torch.load(target)
        out = model(desired_img, raw_return = True)
        desired_out = out.clone()
        class_target = torch.argmax(out)
        desired_fm = get_feature_map(model, desired_img)

    else:
        desired_out = (1. - original_out)

    n = 0
    noiseSum = torch.zeros_like(datapoint)
    current_datapoint_weight = .7 if target is None else .97
    #current_datapoint_weight = 1
    n_fully_forward = 5
    top_k_to_ignore_away = 0
    stop_when_target_in_top = 1

    noise_mag = 1e-7

    #noise_add = torch.randn_like(datapoint)
    noise_add = get_X_given_ans(model, class_target.clone().detach(), datapoint.shape, 1e-2, 30, device).to(device).sign() \
                if target is not None else torch.randn_like(datapoint)
    
    prev_class_score = 0

    check_every = 20

    grad_pre_perc = None

    feature_map_dict = {}

    #print(list(torch.topk(model_out[0], stop_when_target_in_top))); quit()

    # either iterate a set amount of times (Ns), or till the network is fooled / target class is achieved
    while                                       \
            (n < Ns)                            \
                if Ns is not None else          \
                                                \
            (class_ == class_out 
                if target is None else 
        
             #class_out != class_target):
             class_target not in list(torch.topk(model_out[0], stop_when_target_in_top)[1])):



        model.zero_grad()






        if target is None:

            gradient_away = get_gradient_image_fm_away_original(model, 
                                                                datapoint.clone(), original_feature_map.clone(), 
                                                                feature_map_loss_rec, 
                                                                device)

            gradient = torch.clamp(gradient_away, -1, 1)


        else:


                


            gradient_towards = get_gradient_image_fm_towards_desired(model, datapoint.clone().detach(), desired_fm, lossf, device)

            out = model(datapoint, raw_return = True)


            if class_target in list(torch.topk(out, top_k_to_ignore_away)[1][0]):
                
                gradient = gradient_towards

            else:
                # away_feature_map = get_feature_map(model, datapoint)

                if n <= n_fully_forward:
                    gradient_away = get_gradient_image_fm_away_original(model, datapoint.clone().detach(), original_feature_map,
                                                                        feature_map_loss_rec, device)
                
                    multip = min(1, n / n_fully_forward)

                    gradient_away = gradient_away.sign() * gradient_towards.mean()
                    #gradient_away = gradient_away / gradient_away.max() * gradient_towards.max()

                    gradient = gradient_away * (1-multip) + gradient_towards * multip


                else:
                    gradient = gradient_towards
        

            max_class = torch.argmax(out)

            if max_class != class_target and out[0][max_class] > .5:

                if class_target not in feature_map_dict:
                    X_ans = get_X_given_ans(model, class_target, datapoint.shape, 1e-2, 10, device)
                    feature_map_dict[class_target] = get_feature_map(model, X_ans.clone().detach())
                    print("new")


                X_fm = feature_map_dict[class_target]

                gradient_away_specific = get_gradient_image_fm_away_original(model, datapoint.clone().detach(), X_fm, device = device)

                gradient_away_specific = gradient_away_specific.sign() * gradient_towards.mean()

                #gradient = gradient * .8 + gradient_away_specific * .2
                gradient = gradient_away_specific

        


        #mean = noise.abs().mean()
        #maxx = noise.abs().max()
        #zero_less_than(noise, mean)

        if n == 0 or n % check_every == 0 and target is not None:
            datapoint_pre_update = datapoint.clone().detach()



        # get grad of datapoint towards perceptual loss of original

        



        gradient_scaled = eps * gradient * -1.

        #mag_mask = None
        mag_mask = get_gradient_magnitude_mask(model, datapoint, desired_out if desired_out is not None else (1. - original_out))
        gradient_scaled = elem_multip__keeping_mag(gradient_scaled, mag_mask) if mag_mask is not None else gradient_scaled



        X_ = datapoint_pre_adv.clone().detach().requires_grad_(True).to(datapoint.device)

        datapoint = datapoint + gradient_scaled

        if try_correct:

            #datapoint_after = datapoint.clone().detach().requires_grad_(True).to(datapoint.device)
            ploss = perceptual_lossf_(datapoint, X_).to(datapoint.device)
            ploss.backward(retain_graph = True)
            #ploss.backward()
            grad_pre_perc = X_.grad.data
            #grad_pre_perc = grad_pre_perc / grad_pre_perc.abs().max()

            #print(grad_towards_original); quit()

            #grad_towards_original_perceptual = torch.mul(grad_towards_original_perceptual, (1. - mag_mask))
            grad_pre_perc = elem_multip__keeping_mag(grad_pre_perc, (1. - mag_mask)) if mag_mask is not None else grad_pre_perc

            # 1e-5 -> 7
            perc_noise = grad_pre_perc * eps * -1 * 1e-6
            #noise_ = perc_noise + noise_add * noise_mag
            noise_ = perc_noise # + noise_add * noise_mag

            datapoint = datapoint + noise_

            #datapoint = datapoint + grad_towards_original * eps * -1.
            datapoint = datapoint * current_datapoint_weight + datapoint_pre_adv * (1. - current_datapoint_weight)






        #datapoint = datapoint + noise_add * noise_mag
        #datapoint = datapoint + noise_add


        noiseSum += gradient_scaled.clone()




        n += 1

        datapoint_normalised = torch.clamp(
            transforms.Normalize(*mu_sigma)(datapoint.clone())
            , 0, 1    
        )
        class_out = torch.argmax(model_out := model(datapoint_normalised, raw_return = True))

        #print(class_, class_out)

        #print(model_out.shape); quit()

        if n == 0 or n % check_every == 0 and target is not None:
            current_score = model_out[0][class_target]
            
            if prev_class_score > current_score:
                # iteration was pointless
                datapoint = datapoint_pre_update.clone().detach()
                #print("backtrack")

            prev_class_score = model_out[0][class_target]







        #print(n)
        if data is not None and (n < 20 or n % 10 == 0):
            #data = [n, datapoint_normalised]
            data[0] = n
            data[1] = datapoint_normalised
            data[2] = mag_mask
            sleep(.01)

            if not data[3]:
                break;




        #if abs(eps) <= 10e-4 and n > 10:
        if n > 1000 and False:
            break











    adv = datapoint

    original = datapoint_pre_adv
    diff = adv - original
    final = adv.clone()
    last = adv.clone()

    datapoint_normalised = torch.clamp(
        transforms.Normalize(*mu_sigma)(final.clone())
        , 0, 1    
    )

    model_out = model(datapoint_normalised, raw_return = True)




    



    i = 0
    i_inc = 1e-3
    index = 0
    while i_inc > 1e-20:
        last = final.clone()

        #print(desired_out.shape); quit()
        mask = get_gradient_magnitude_mask(model, last, desired_out)
        #mask_ = torch.ones_like(mask)

        mask[mask < i] = 0
        mask[mask >= i] = 1

        #final = original + (1-i) * diff
        #m = (1-i) * mask_ + (i) * mask
        m = mask
        final = original + torch.mul(diff, m)


        datapoint_normalised = torch.clamp(
            transforms.Normalize(*mu_sigma)(final.clone())
            , 0, 1    
        )
        model_out = model(datapoint_normalised, raw_return = True)
        


        if index % 10 == 0 or True:
            print(i)

            data[1] = datapoint_normalised


        if(
                class_target not in list(torch.topk(model_out[0], stop_when_target_in_top)[1])
                    if target is not None else
                torch.argmax(model_out) == class_out
            ):


            # too far

            #i = max(0, i - i_inc)
            i -= i_inc
            i_inc = i_inc / 2.

            final = last.clone()
            
            print("back")

        else:
            i += i_inc
            index += 1





    # re normalise and clamp data so that model interacts with it correctly
    adv = transforms.Normalize(*mu_sigma)(last)
    adv = torch.clamp(adv, 0, 1)

    #noise = transforms.Normalize(*mu_sigma)(noise)
    #noise = torch.clamp(noise, 0, 1)

    #hook.remove()

    print(class_, class_out)

    if data is not None:
        data[1] = adv
        data[2] = noiseSum






    return adv if not returnNoiseAdded else (adv, noiseSum)




























