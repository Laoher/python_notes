import pygame

pygame.init()
screen = pygame.display.set_mode((400,300))
run =True

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    pygame.display.flip()