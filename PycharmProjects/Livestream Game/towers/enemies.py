import pygame

class enemies:
    imgs = []
    def __init__(self,x,y, width, height):
        self.x  =x
        self.y = y
        self.width = width
        self.height = height
        self.animation_count = 0
        self.health = 1
        self.path = []
        self.img = None

    def draw(self,win):
        self.animation_count+=1
        self.img = self.imgs[self.animation_count]
        if self.animation_count >= len(self.imgs):
            self.animation_count = 0
        win.blit(self.img,(self.x,self.y))
        self.move()


    def collide(self,  X, Y):
        if X<=self.x + self.width and X>=self.x:
            if Y<=self.y +self.height and Y>=self.y:
                return True
        return False

    def move(self):
        pass

    def hit(self):
        self.health -= 1
        if self.health <= 0:
            return True

