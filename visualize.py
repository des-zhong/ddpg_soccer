import sys
import pygame
from config import *
import time



white = (255,255,255)
def text_objects(text, font):
    textSurface = font.render(text, True, white)
    return textSurface, textSurface.get_rect()

def draw(state):
    teamA = state[0:4 * teamA_num]
    teamB = state[4 * teamA_num:4 * teamA_num + 4 * teamB_num]
    soccer_coord = [state[-4], state[-3]]
    ratio = 3
    pygame.init()
    my_font = pygame.font.SysFont("arial", 16)
    # 设置主屏窗口 ；设置全屏格式：flags=pygame.FULLSCREEN
    screen = pygame.display.set_mode((field_width / ratio, field_length / ratio))
    # 设置窗口标题
    pygame.display.set_caption('soccer game')
    screen.fill('white')
    wid = 20
    rect = pygame.Rect(field_width / ratio - wid, (field_length - gate_length) / 2 / ratio, wid, gate_length / ratio)
    pygame.draw.rect(screen, (190, 190, 190), rect)
    # largeText = pygame.font.Font('freesansbold.ttf', 115)
    # TextSurf, TextRect = text_objects('123', largeText)
    # TextRect.center = ((field_width / ratio / 2), (field_width / ratio / 2))
    # screen.blit(TextSurf, TextRect)
    for i in range(teamA_num):
        pygame.draw.circle(screen, (255, 0, 0), (
            (teamA[4 * i] + field_width / 2) / ratio, (teamA[4 * i + 1] + field_length / 2) / ratio),
                           radius_player / ratio)
    for i in range(teamB_num):
        pygame.draw.circle(screen, (0, 0, 255), (
            (teamB[4 * i] + field_width / 2) / ratio, (teamB[4 * i + 1] + field_length / 2) / ratio),
                           radius_player / ratio, width=1)
    pygame.draw.circle(screen, (0, 255, 0),
                       ((soccer_coord[0] + field_width / 2) / ratio, (soccer_coord[1] + field_length / 2) / ratio),
                       radius_soccer / ratio, width=1)
    pygame.display.flip()  # 更新屏幕内容
