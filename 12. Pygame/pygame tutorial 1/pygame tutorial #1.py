import pygame
import os

os.chdir(r'/Users/tyler/PycharmProjects/12. Pygame/pygame tutorial 1')
print(os.getcwd())
pygame.init()

win = pygame.display.set_mode((500, 500))

pygame.display.set_caption("First Game")

walkRight = [pygame.image.load('R1.png'), pygame.image.load('R2.png'), pygame.image.load('R3.png'), pygame.image.load('R4.png'), pygame.image.load('R5.png'), pygame.image.load('R6.png'), pygame.image.load('R7.png'), pygame.image.load('R8.png'), pygame.image.load('R9.png')]
walkLeft = [pygame.image.load('L1.png'), pygame.image.load('L2.png'), pygame.image.load('L3.png'), pygame.image.load('L4.png'), pygame.image.load('L5.png'), pygame.image.load('L6.png'), pygame.image.load('L7.png'), pygame.image.load('L8.png'), pygame.image.load('L9.png')]
bg = pygame.image.load('bg.jpg')
char = pygame.image.load('standing.png')


clock = pygame.time.Clock()
x = 250
y = 420
width = 40
height = 60
vel = 5

isJump = False
jumpCount = 10
left = False
right = False
walkcount = 0


def redrawGameWindow():
    global walkcount
    win.blit(bg,(0,0))

    if walkcount +1 >=27:
        walkcount = 0
    if left:
        win.blit(walkLeft[walkcount//3],(x,y))
        walkcount +=1
    elif right:
        win.blit(walkRight[walkcount//3],(x,y))
        walkcount += 1
    else:
        win.blit(char,(x,y))

    pygame.display.update()
#mainloop
run = True
while run:
    clock.tick(27)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT] and x >= vel:
        x -= vel
        left = True
        right = False
    elif keys[pygame.K_RIGHT] and x < 500 - width:
        x += vel
        left = False
        right = True
    else:
        left = False
        right = False
        walkcount = 0

    if not (isJump):
        if keys[pygame.K_SPACE]:
            isJump = True
            right = False
            left = False
            walkcount = 0
    else:
        if jumpCount >= -10:
            neg = 1
            if jumpCount <0:
                neg = -1
            y -= (jumpCount ** 2) * 0.5 *neg
            jumpCount -=1
        else:
            isJump = False
            jumpCount = 10

    redrawGameWindow()

pygame.quit()
