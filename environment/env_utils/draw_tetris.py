import sys
import pygame as pg
import numpy as np
import time

class renderer:
    def __init__(self, resolution):
        pg.init()
        self.border_width = 2
        self.border_fade = 0.2
        self.resolution = self.res_x, self.res_y = resolution
        self.screen = self.createScreen(self.res_x,self.res_y)
        self.field_row_size = 4
        self.piece_size = self.pieceSize(20)
        self.fg_colormap = [[20,20,20],
                            [255,0,0],
                            [0,255,0],
                            [115,145,255],
                            [255,0,255],
                            [0,255,255],
                            [255,255,0],
                            [255,255,255],
                            [170,170,170]]
        self.bg_colormap = []
        for r in self.fg_colormap:
            x = [self.border_fade * c for c in r]
            self.bg_colormap.append(x)

    def createScreen(self, newwidth, newheight):
        self.size = self.width, self.height = newwidth, newheight
        screen = pg.display.set_mode(self.size)
        screen.fill([0,0,0])
        return screen

    def pieceSize(self, newsize):
        self.screen.fill([0,0,0])
        return newsize

    def rowSize(self, newsize):
        # global field_row_size
        self.field_row_size = newsize
        screen.fill([0,0,0])

    def drawField(self, field, x, y):
        field_width = field[0].size
        d = 0.5 * self.border_width
        fg_size = self.piece_size - 2*d

        bg_rect = pg.Rect(x,y,self.piece_size,self.piece_size)
        fg_rect = pg.Rect(x+d,y+d, fg_size,fg_size)
        count = 0
        for i in np.nditer(field):
            self.screen.fill(self.bg_colormap[i], rect=bg_rect)
            self.screen.fill(self.fg_colormap[i], rect=fg_rect)
            bg_rect.left += self.piece_size
            fg_rect.left += self.piece_size
            count += 1
            if count == field_width:
                fg_rect.left = x + d
                bg_rect.left = x
                fg_rect.top += self.piece_size
                bg_rect.top += self.piece_size
                count = 0

    def pollEvents(self, ):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN: return True
        return False

    def drawAllFields(self, fields):
        ret = self.pollEvents()
        x = 0
        y = 0
        height, width = fields[0].shape
        n = self.field_row_size = len(fields)

        x_ratio = (self.piece_size*width*n) / self.res_x
        y_ratio = (self.piece_size*height) / self.res_y
        r = max(x_ratio, y_ratio)
        if r > 1.05 or r < 0.95:
            self.piece_size /= r

        width = (width + 1) * self.piece_size
        height = (height + 1) * self.piece_size
        count = 0
        for f in fields:
            self.drawField(f,x,y)
            x += width
            count += 1
            if count == self.field_row_size:
                count = 0
                x = 0
                y += height
        pg.display.flip()
        return ret
