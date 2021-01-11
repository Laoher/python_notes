import os
import pygame

class Game:
    def __init__(self):
        self.width=1000
        self.height = 700
        self.win = pygame.display.set_mode((self.width, self.height))
        self.enemies = []
        self.towers = []
        self.lives = 10
        self.money = 100
        self.bg = pygame.image.load(os.path.join("game_assets","bg.png"))
        self.pos
    def run(self):
        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                pos = pygame.mouse.get_pos()  # pos is position

                if event.type == pygame.MOUTHBUTTONDOWN:
                    print(pos)
            self.draw()
        pygame.quit()

    def draw(self):
        self.win.blit(self.bg,(0,0))
        pygame.display.update()
