from models import *
from datasets import *
from torch_common import *
from FSGM_ import fsgm_attack
from gui import Adversarial_GUI

from feature_map_ import *

cpu, gpu = torch.device("cpu"), torch.device("cuda")



if __name__ == "__main__":





    vgg16 = get_vgg16(True, gpu); #print(vgg16)

    ans = 1
    X = get_X_given_ans(vgg16, ans, [1, 3, 224, 224], minloss_pow10=20, lr=1e-2)
    with open("target.pt", "wb") as f:
        torch.save(X, f)


    imgnet = getImageNet(targetImgSize=224, shuffle=True)

    dp = get_datapoint(imgnet, device = gpu)[0]


    gui = Adversarial_GUI(vgg16, imgnet, feature_map_attack_repeated, device = gpu)
    gui.main(); quit()
    sadasda
    



    #adv = fsgm_attack(vgg16, dp,  device = gpu)
    adv = feature_map_attack(vgg16, dp,  device = gpu)

    showtensor(dp); showtensor(adv)

    print(vgg16(dp)); print(vgg16(adv))



    