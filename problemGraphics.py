from graphics import Window
class pacmanGraphic(Window):
    xStep = 40
    yStep = 40
    def gCoord (self, x, y):
        return x*pacmanGraphic.xStep + 10, y*pacmanGraphic.yStep + 10
    
    def drawPacman(self, p, pos):
        x, y = self.gCoord(pos[0], pos[1])
        
        xEye = x + pacmanGraphic.xStep / 2
        yEye = y + pacmanGraphic.yStep / 4
        
        self.pacman = self.arc(x+3, y+3,
                               x+pacmanGraphic.xStep-2,
                               y+pacmanGraphic.yStep-2,
                          startAngle=25, endAngle=315,
                          outline="#000", fill="#ffff00",
                          width=2)
        
        self.pacmanEye =  self.oval(xEye, yEye, xEye+3, yEye+3,
                          fill="#000", width=0.1)

    def drawDots(self, p):
        self.dots = []
        for (x, y) in p.dots:
            c, r = self.gCoord(x, y)
            
            self.dots.append (self.oval(c, r,
                            c+pacmanGraphic.xStep-10,
                            r+pacmanGraphic.yStep-10,
                            fill = '#ffff00',
                            width=0.1))
            
    def drawMonster(self, p):
        self.monsters = []
        for color, (x, y) in p.monsters:
            c, r = self.gCoord(x, y)
            if color == 'R': mColor = 'red'
            elif color == 'G': mColor = 'green'
            elif color == 'B': mColor = 'blue'

            m = self.oval(c, r,
                    c+pacmanGraphic.xStep-10,
                    r+pacmanGraphic.yStep-10,
                    fill = mColor, #'#ffff00',
                    width=0.1)
            e1 = self.oval(c+8, r+4,
                    c+15,
                    r+11,
                    fill = 'yellow', 
                    width=0.1)
            e2 = self.oval(c+17, r+4,
                    c+24,
                    r+11,
                    fill = 'yellow', 
                    width=0.1)
            self.monsters.append ((m, e1, e2))
                          

    def addText(self, x, y, str, color='blue', fontSize=10):
        x, y = self.gCoord(x, y)
        return self.text(x, y, str, color, fontSize)
    
    def setup(self, p):
        
        for x, y in p.walls:
            c, r = self.gCoord(x, y)
            self.rec(c, r, c+pacmanGraphic.xStep,
                     r+pacmanGraphic.yStep,
                     outline='#000',
                     fill='#fff',
                     width=2)
        self.drawDots(p)
        self.drawPacman(p, p.pacman)
        

        self.refresh()

    def move_pacman (self, dx, dy):
        gx, gy = dx * pacmanGraphic.xStep, dy * pacmanGraphic.yStep
        self.move (self.pacman, gx, gy)
        self.move(self.pacmanEye, gx,  gy)
        self.refresh()
        self.wait(0.1)

    def remove_dot(self, index):
        self.remove(self.dots[index])
        self.refresh()

    def move_monster(self, dx, dy, index):
        gx, gy = self.gCoord(dx, dy)
        m, e1, e2 = self.monsters[index]
        self.move(m, gx, gy)
        self.move(e1, gx, gy)
        self.move(e2, gx, gy)
        self.refresh()

    
