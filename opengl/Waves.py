import pygame
from pygame.locals import *
import math, sys
import Numeric
Fs=11025  # sample rate
pygame.mixer.init(Fs, -16, 0)   # mono, 16-bit
pygame.init()
ScreenX = 800
ScreenY = 601
pygame.display.set_icon(pygame.image.load('Icon.png'))
pygame.display.set_caption("Wave Interference Simulator v.2.0.0 - Ian Mallett - 2008")
Surface = pygame.display.set_mode((ScreenX,ScreenY))
#Waves: [Amplitude, Frequency, Points]
F_Read = open('Waves.txt','r')
WaveData = F_Read.readlines()
try:
    Data = []
    for line in WaveData:
        if line.endswith('\n'):
            line = line[:-1]
        line = line.split(" ")
        Data.append(line)
    Wave1 = [int((Data[0])[0]),float((Data[0])[1]),[0]]
    Wave2 = [int((Data[1])[0]),float((Data[1])[1]),[0]]
    Wave3 = [int((Data[2])[0]),float((Data[2])[1]),[0]]
    Wave4 = [int((Data[3])[0]),float((Data[3])[1]),[0]]
except: sys.exit()
WaveGeneratorProgress = [0,0,0,0,0] #Wave Generation Progress * 4, Number of points Plotted
Resolution = 1
MakeResultant = False
Wave5 = []
def Draw():
    Surface.fill((0,0,0))
    pygame.draw.line(Surface,(255,255,255),(0,0),(ScreenX,0),3)
    pygame.draw.line(Surface,(255,255,255),(0,100),(ScreenX,100),3)
    pygame.draw.line(Surface,(255,255,255),(0,200),(ScreenX,200),3)
    pygame.draw.line(Surface,(255,255,255),(0,300),(ScreenX,300),3)
    pygame.draw.line(Surface,(255,255,255),(0,400),(ScreenX,400),3)
    pygame.draw.line(Surface,(255,255,255),(0,600),(ScreenX,600),3)
    pygame.draw.line(Surface,(0,0,255),(0,50),(ScreenX,50),1)
    pygame.draw.line(Surface,(0,0,255),(0,150),(ScreenX,150),1)
    pygame.draw.line(Surface,(0,0,255),(0,250),(ScreenX,250),1)
    pygame.draw.line(Surface,(0,0,255),(0,350),(ScreenX,350),1)
    pygame.draw.line(Surface,(0,0,255),(0,500),(ScreenX,500),1)
    x = 0
    for Point in Wave1[2]:
        Surface.set_at((x,int(Point+50)),(255,255,255));  x += Resolution
    x = 0
    for Point in Wave2[2]:
        Surface.set_at((x,int(Point+150)),(255,255,255));  x += Resolution
    x = 0
    for Point in Wave3[2]:
        Surface.set_at((x,int(Point+250)),(255,255,255));  x += Resolution
    x = 0
    for Point in Wave4[2]:
        Surface.set_at((x,int(Point+350)),(255,255,255));  x += Resolution
    if MakeResultant:
        x = 0
        for Point in Wave5:
            Surface.set_at((x,int(Point+500)),(255,0,0));  x += Resolution
    pygame.display.flip()
def MakeSounds():
    Waves = [Wave1,Wave2,Wave3,Wave4]
    Sounds = []
    for Wave in Waves:
        length = Fs * 2.002#3.0 seconds
        freq = 440.0 * ((Wave[1]*2)/3.0)
        amp = Wave[0]
        tmp = []
        for t in range(int(length)):
            v = amp * Numeric.sin(t*freq/Fs*2*Numeric.pi) 
            tmp.append(v)
        snd = pygame.sndarray.make_sound(Numeric.array(tmp,Numeric.Int0))
        snd.set_volume(0.1)
        Sounds.append(snd)
    for snd in Sounds:
        snd.play(-1)
def UpdateWaves():
    global WaveGeneratorProgress, MakeResultant
    if WaveGeneratorProgress[4] == ScreenX/Resolution:
        Wave1[2] = (Wave1[2])[1:]
        Wave2[2] = (Wave2[2])[1:]
        Wave3[2] = (Wave3[2])[1:]
        Wave4[2] = (Wave4[2])[1:]
        WaveGeneratorProgress[4] -= 1
        MakeResultant = True
        if Wave5 == []:
            PointNumber = 0
            for x in range(0,ScreenX,Resolution):
                Point = Wave1[2][PointNumber] + Wave2[2][PointNumber] + Wave3[2][PointNumber] + Wave4[2][PointNumber]
                PointNumber += 1
                if abs(Point) <= 99:
                    Wave5.append(Point)
                else:
                    Wave5.append(0.0)
            MakeSounds()
    Wave1[2].append( Wave1[0]*math.sin(math.radians(WaveGeneratorProgress[0])) )
    Wave2[2].append( Wave2[0]*math.sin(math.radians(WaveGeneratorProgress[1])) )
    Wave3[2].append( Wave3[0]*math.sin(math.radians(WaveGeneratorProgress[2])) )
    Wave4[2].append( Wave4[0]*math.sin(math.radians(WaveGeneratorProgress[3])) )
    WaveGeneratorProgress[0] += Wave1[1]
    WaveGeneratorProgress[1] += Wave2[1]
    WaveGeneratorProgress[2] += Wave3[1]
    WaveGeneratorProgress[3] += Wave4[1]
    WaveGeneratorProgress[4] += 1
def GetInput():
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit(); sys.exit()
def main():
    while True:
        GetInput()
        UpdateWaves()
        Draw()
if __name__ == '__main__': main()
