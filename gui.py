from PIL import Image
import pygame as pg
from torch import Tensor, nn, device
from torch.utils.data import DataLoader
from keyboard import is_pressed as isP
from time import time as now

from torch_common import *



BLACK, WHITE = (0,0,0), (255,255,255)

# https://stackoverflow.com/questions/25202092/pil-and-pygame-image
def pilImageToSurface(pilImage):
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
        self.dataset = dataset
        self.dataIter = iter(dataset)
        self.func = advFunc
        self.device = device

        self.scrw = 1200
        self.scrh = 600
        self.fps = 30
        self.run = True
        self.imgSize = next(self.dataIter)[0][0].shape[2]
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

        self.newImageKey = "a"  
        self.noiseKey = "s"

        self.eps = 0
        self.scroll = 0




    def eventLoop(self):
        self.scroll = 0
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.run = False
            if event.type == pg.MOUSEWHEEL:
                self.scroll = event.y;

    def draw(self):
        if self.currentImage is not None:
            cImg = pilImageToSurface(topil(self.currentImage))
            self.scr.blit(cImg, cImg.get_rect(center = (self.imgSize, self.imgSize // 2 + self.spacing)))

        if self.currentNoise is not None:

            #noise = (self.currentNoise + 1.) / 2. * self.eps * 2
            #noise = self.currentNoise * self.eps
            noise = self.currentNoise

            cImg = (pilImageToSurface(topil(noise)))



            self.scr.blit(cImg, cImg.get_rect(center = (
                self.imgSize * 2 + self.spacing,
                self.imgSize // 2 + self.spacing)
            ))

        if self.currentAdv is not None:
            cImg = pilImageToSurface(topil(self.currentAdv))
            self.scr.blit(cImg, cImg.get_rect(center = (
                self.imgSize * 3 + self.spacing * 2,
                self.imgSize // 2 + self.spacing
            )))

        epsText = self.bigFont.render(f'eps = {round(self.eps, 4)}', True, "white")
        self.scr.blit(epsText, (self.imgSize * 2 - self.spacing, self.imgSize + self.spacing))

        if self.currentOutString_original is not None:
            for i, text in enumerate(self.currentOutString_original.split("\n")):
                outText = self.font.render(text, True, "white")
                self.scr.blit(outText, (self.spacing, self.imgSize + self.spacing * 2 + self.normalTextSize * i))

        if self.currentOutString_adversarial is not None:
            for i, text in enumerate(self.currentOutString_adversarial.split("\n")):
                outText = self.font.render(text, True, "white")
                self.scr.blit(outText, (self.imgSize * 3 + self.spacing * 2, self.imgSize + self.spacing * 2 + self.normalTextSize * i))


    def tickNN(self):
        if isP(self.noiseKey):

            self.currentAdv, self.currentNoise = self.func(
                self.model, self.currentImage.to(self.device),
                eps = self.eps,
                returnNoiseAdded = True
            )

            

        if self.currentImage is not None:
            model = self.model.to(self.device)
            self.currentOutString_original = model(self.currentImage.to(self.device))
            
        if self.currentAdv is not None:
            self.currentOutString_adversarial = self.model(self.currentAdv.to(self.device))
           

        else:
            self.currentAdv = self.currentImage
            self.currentNoise = None


    def tickAll(self):
        if isP(self.newImageKey):
            self.currentImage = None
        
        if self.currentImage is None:
            try:
                batch = (next(self.dataIter)[0])
                N = list(batch.shape)[0]
                self.currentImage = batch[randint(0, N - 1)]
                self.currentAdv = self.currentImage
            
            except StopIteration:
                self.dataIter = iter(self.dataset)

        if now() - self.lastTick > self.tickNNevery:
            self.lastTick = now()
            self.tickNN()




        if self.scroll != 0:
            self.eps += self.scroll / 1000



    def main(self):
        while self.run:
            self.eventLoop()
            self.tickAll()
            self.scr.fill(BLACK)
            self.draw()
            pg.display.update()
            self.clock.tick(self.fps)
