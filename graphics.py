from tkinter import *
import time
class Window:
    def __init__ (self, w=700, h=700):
        
        self.root = Tk()
        self.canvas = Canvas (self.root, width = w, height=h)
        self.canvas.pack()
        self.refresh()
        

    def arc(self, x1, y1, x2, y2, startAngle, endAngle, outline, fill, width):
        return self.canvas.create_arc(x1, y1, x2, y2,
                               start=startAngle, extent=endAngle,
                               outline=outline, fill=fill, width=width)

    def oval(self, x1, y1, x2, y2, fill, width):
        return self.canvas.create_oval(x1, y1, x2, y2,
                                fill=fill, width=width)

    def text(self, x, y, str, color='blue', fontSize=32):
        return self.canvas.create_text(x, y, font=("Helvetika", fontSize), text=str, fill=color)

    
    def rec(self, x1, y1, x2, y2, outline, fill, width):
        return self.canvas.create_rectangle(x1, y1, x2, y2, outline=outline, fill=fill, width=width)

    def move(self, widget, deltaX, deltaY):
        self.canvas.move (widget, deltaX, deltaY)

    def remove(self, widget):
        self.canvas.delete(widget)

    def refresh(self):
        self.canvas.update()
    def wait(self, t):
        time.sleep(t)
