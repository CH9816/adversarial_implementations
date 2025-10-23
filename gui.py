from PIL import Image
import pygame as pg
from torch import Tensor, nn, device
from torch.utils.data import DataLoader
from keyboard import is_pressed as isP
from time import time as now
from time import sleep
from feature_map_ import get_feature_map

from torch_common import *

from threading import Thread
from io import BytesIO


BLACK, WHITE = (0,0,0), (255,255,255)

# https://stackoverflow.com/questions/25202092/pil-and-pygame-image
def pilImageToSurface(pilImage : Image, scale = 1):

    if scale != 1:
        pilImage = pilImage.resize((pilImage.size[0] * scale, pilImage.size[1] * scale),
                                   Image.Resampling.LANCZOS)

    return pg.image.fromstring(
        pilImage.tobytes(), pilImage.size, pilImage.mode).convert()

class Adversarial_GUI:
    def __init__(self,
                 model : nn.Module,
                 dataset = DataLoader,
                 advFunc = None, 
                 device = device("cuda")):
        pass;
        pg.init()
        pg.font.init()

        self.model = model.eval()
        self.model = self.model.to(device)


        memData_ = BytesIO()
        torch.save(self.model, memData_)
        memData_.seek(0)
        self.cpu_model = torch.load(memData_, map_location="cpu", weights_only=False).eval()
        self.cpu_model.device = torch.device("cpu")

        self.dataset = dataset
        self.dataIter = iter(dataset)
        self.func = advFunc
        self.device = device

        self.scrw = 1800
        self.scrh = 900
        self.fps = 30
        self.run = True
        self.imgScale = 2
        self.imgSize = next(self.dataIter)[0][0].shape[2] * self.imgScale
        self.spacing = 20
        self.normalTextSize = 16
        self.bigTextSize = 50
        self.tickNNevery = .3
        self.lastTick = now()

        self.scr = pg.display.set_mode((self.scrw, self.scrh))
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont("Arial", self.normalTextSize)
        self.bigFont = pg.font.SysFont("Arial", self.bigTextSize)

        self.currentImage = None
        self.currentNoise = None
        self.currentAdv = None
        self.currentOutString_original = None
        self.currentOutString_adversarial = None
        self.text_updated = False


        self.newImageKey = "a"  
        self.noiseKey = "s"
        self.cancelKey = "c"
        self.saveFmKey = "f"

        self.eps = 10
        self.scroll = 0
        self.oldN = 0

        self.attack_out = (None, None)
        self.new_attack = False
        self.attack_data = []

        self.threads = [
            Thread(target = self.attack_thread)    
        ]

        for thread in self.threads:
            thread.start()
            pass

    def __del__(self):
        for thread in self.threads:
            thread.join()


    def eventLoop(self):
        self.scroll = 0
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.run = False
            if event.type == pg.MOUSEWHEEL:
                self.scroll = event.y;

    def draw(self):
        if self.currentImage is not None:
            cImg = pilImageToSurface(topil(self.currentImage), self.imgScale)
            self.scr.blit(cImg, cImg.get_rect(center = (self.imgSize // 2, self.imgSize // 2 + self.spacing)))

        if self.currentNoise is not None:

            #noise = (self.currentNoise + 1.) / 2. * self.eps * 2
            #noise = self.currentNoise * self.eps
            noise = self.currentNoise

            cImg = (pilImageToSurface(topil(noise), self.imgScale))



            self.scr.blit(cImg, cImg.get_rect(center = (
                self.imgSize * 1.5 + self.spacing,
                self.imgSize // 2 + self.spacing)
            ))

        if self.currentAdv is not None:
            cImg = pilImageToSurface(topil(self.currentAdv), self.imgScale)
            self.scr.blit(cImg, cImg.get_rect(center = (
                self.imgSize * 2.5 + self.spacing * 2,
                self.imgSize // 2 + self.spacing
            )))

        epsText = self.bigFont.render(f'eps = {round(self.eps, 4)}', True, "white")
        self.scr.blit(epsText, (self.imgSize * 1 + self.spacing, self.imgSize + self.spacing))

        stepsText = self.bigFont.render(f"steps = {self.attack_data[0]}", True, "white")
        self.scr.blit(stepsText, (self.imgSize * 1 + self.spacing, self.imgSize + self.spacing + self.bigTextSize))

        if self.currentOutString_original is not None:
            for i, text in enumerate(self.currentOutString_original.split("\n")):
                outText = self.font.render(text, True, "white")
                self.scr.blit(outText, (self.spacing, self.imgSize + self.spacing * 2 + self.normalTextSize * i))

        if self.currentOutString_adversarial is not None:
            for i, text in enumerate(self.currentOutString_adversarial.split("\n")):
                outText = self.font.render(text, True, "white")
                self.scr.blit(outText, (self.imgSize * 2.5 + self.spacing * 2, self.imgSize + self.spacing * 2 + self.normalTextSize * i))


    def attack_thread(self):

        start = now()
        self.attack_data = [None, None, None, True]
        sleepTime = .1

        while self.run:

            if self.new_attack:

                

                if self.currentImage is not None:


                    self.attack_out = self.func(
                        self.model, self.currentImage.to(self.device),
                        eps = self.eps,
                        returnNoiseAdded = True,
                        data = self.attack_data,

                        device = self.device
                    )

                    self.currentAdv, self.currentNoise = self.attack_out

                    self.text_updated = False
                    self.tickNN()

                    topil(self.currentAdv).save(f"outp/latest_adv.png")
                    topil(self.currentImage).save(f"outp/latest_orig.png")

                self.new_attack = False

            sleep(sleepTime)





    def tickNN(self):
        if isP(self.noiseKey):
            self.new_attack = True
            self.attack_data[3] = True
            

        #if len(self.attack_data) == 2:
        if self.attack_data[0] is not None:

            if self.new_attack:
                self.currentAdv = self.attack_data[1]
                self.currentNoise = self.attack_data[2]
            else:
                self.currentAdv, self.currentNoise = self.attack_out

            if self.oldN != self.attack_data[0]:
                self.oldN = self.attack_data[0]
                self.text_updated = False
                
                self.draw()

        device = torch.device("cpu")

        if not self.text_updated:

            self.text_updated = True

            if self.currentImage is not None:
                with torch.no_grad():
                    model = self.cpu_model.to(device)
                    self.currentOutString_original = model(self.currentImage.clone().to(device).cpu())
            
            if self.currentAdv is not None:
                with torch.no_grad():
                    #self.currentAdv, self.currentNoise = self.attack_out

                    model = self.cpu_model.to(device)
                    self.currentOutString_adversarial = model(self.currentAdv.clone().to(device).cpu())

                    #print(torch.argmax(model(self.currentAdv.clone().to(device), raw_return = True)))
           

            else:
                self.currentAdv = self.currentImage
                self.currentNoise = None


    def tickAll(self):
        if isP(self.newImageKey):
            self.currentImage = None
            
        if isP('g'):
            self.text_updated = False

        if isP(self.cancelKey):
            self.attack_data[3] = False
        
        if self.currentImage is None:
            self.text_updated = False

            try:
                batch = (next(self.dataIter)[0])
                N = list(batch.shape)[0]
                #n = randint(0, N - 1) 
                n = 0
                self.currentImage = batch[n]
                self.currentAdv = self.currentImage
            
            except StopIteration:
                self.dataIter = iter(self.dataset)
                
            #self.currentAdv = self.currentImage
            #self.currentNoise = None


        if now() - self.lastTick > self.tickNNevery:
            self.lastTick = now()
            self.tickNN()




        if self.scroll != 0:
            self.eps = max(0, self.eps + self.scroll / (10 if isP("shift") else 10000))


        if isP(self.saveFmKey) and False:
            with open("target.pt", "wb") as f:
                torch.save(self.currentImage, f)
            print(f"target saved! {self.currentOutString_original}")



    def main(self):
        while self.run:
            self.eventLoop()
            self.tickAll()
            self.scr.fill(BLACK)
            self.draw()
            pg.display.update()
            self.clock.tick(self.fps)
