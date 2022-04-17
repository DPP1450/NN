import math
class Car:
    def __init__(self,x,y,angle):
        self.forward = float(math.pi*angle/180)
        self.left = float(math.pi*(angle+45)/180)
        self.right = float(math.pi*(angle-45)/180)
        self.x = 0
        self.y = 0
        self.radius = 3
        self.ld = 0
        self.rd = 0
        self.fd = 0
        self.go =False