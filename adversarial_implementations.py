from models import *
from datasets import *
from torch_common import *
from FSGM_ import fsgm_attack
from gui import Adversarial_GUI

cpu, gpu = torch.device("cpu"), torch.device("cuda")



if __name__ == "__main__":





    vgg16 = get_vgg16(True, gpu); print(vgg16)

    imgnet = getImageNet(targetImgSize=224, shuffle=True)

    gui = Adversarial_GUI(vgg16, imgnet, fsgm_attack, gpu)
    gui.main(); quit()
    
    dp = get_datapoint(imgnet, device = gpu)[0]



    adv = fsgm_attack(vgg16, dp,  device = gpu)

    showtensor(dp); showtensor(adv)

    print(vgg16(dp)); print(vgg16(adv))



    