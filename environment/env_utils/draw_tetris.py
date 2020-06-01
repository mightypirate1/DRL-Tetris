import sys
import pygame as pg
import numpy as np
import time

global_renderer = None

def get_global_renderer(resolution=None, color_theme=None):
    global global_renderer
    if global_renderer is None:
        global_renderer = _renderer(resolution, color_theme=color_theme)
    return global_renderer

def hex2intlist(x):
    counter = 0
    while counter < len(x):
        counter += 2
        yield int(x[counter-2:counter], 16)

class _renderer:
    def __init__(self, resolution, color_theme=None):
        pg.init()
        self.border_width = 2   #Border of dark pixels around blocks
        self.border_fade = 0.2  #Border color. 1=piececolor, 0=black
        self.padding = 0.5      #Distance between fields measured in units of piece sizes
        self.scaled = False     #This is set when the piece size is scaled to the inputs

        self.resolution = self.res_x, self.res_y = resolution
        self.screen = self.createScreen(self.res_x,self.res_y)
        self.field_row_size = 4
        self.piece_size = self.pieceSize(20)
        if color_theme is None:
            self.fg_colormap = [[25,25,25],
                                [255,0,0],
                                [0,255,0],
                                [115,145,255],
                                [255,0,255],
                                [0,255,255],
                                [255,255,0],
                                [255,255,255],
                                [170,170,170]]
        else:
            self.fg_colormap = [ list(hex2intlist(hex_str)) for hex_str in color_theme ]
        self.bg_colormap = []
        for r in self.fg_colormap:
            x = [int(self.border_fade * c) for c in r]
            self.bg_colormap.append(x)

    def createScreen(self, newwidth, newheight):
        self.size = self.width, self.height = newwidth, newheight
        screen = pg.display.set_mode(self.size)
        screen.fill([0,0,0])
        return screen

    def clearScreen(self):
        self.screen.fill([0,0,0])

    def pieceSize(self, newsize):
        self.screen.fill([0,0,0])
        return newsize

    def pollEvents(self, ):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN: return True
        return False

    def pause_on_event(self):
        if self.pollEvents():
            print("----------------------")
            print("--------PAUSED--------")
            print("----------------------")
            while not self.pollEvents():
                time.sleep(1.0)

    def drawField(self, field, x, y, width, height):
        if field is None:
            return
        d = 0.5 * self.border_width
        fg_size = max(1, self.piece_size - 2*d)

        bg_rect = pg.Rect(x,y,self.piece_size,self.piece_size)
        fg_rect = pg.Rect(x+d,y+d, fg_size,fg_size)
        count = 0

        for i in np.nditer(field):
            self.screen.fill(self.bg_colormap[i], rect=bg_rect)
            self.screen.fill(self.fg_colormap[i], rect=fg_rect)
            bg_rect.left += self.piece_size
            fg_rect.left += self.piece_size
            count += 1
            if count == width:
                fg_rect.left = x + d
                bg_rect.left = x
                fg_rect.top += self.piece_size
                bg_rect.top += self.piece_size
                count = 0

    def drawAllFields(self, fields, force_rescale=False, pause_on_event=False):
        #We assume that fields is a list of c rows.
        #Each row contains fields. All rows assuemd to be of equal length except
        #for possibly the last one which may be shorter.

        #Pausing capability
        if pause_on_event:
            self.pause_on_event()

        #Measure fields
        n_rows = len(fields)
        if n_rows == 0: return
        n_cols = len(fields[0])
        if n_cols == 0: return
        if type(fields[0][0]) is list:
            height = len(fields[0][0])
            width  = len(fields[0][0][0])
        else:
            height, width = fields[0][0].shape

        #Rescale pice-sizes to match drawing area
        if force_rescale or not self.scaled:
            x_ratio = (self.piece_size*(width +self.padding)*n_cols) / self.res_x
            y_ratio = (self.piece_size*(height+self.padding)*n_rows) / self.res_y
            r = max(x_ratio, y_ratio)
            if r > 1.0 or r < 0.95:
                self.piece_size /= r

        #Draw field
        self.clearScreen()
        x, y = 0, 0 #First field drawn in top left corner
        width_px  = (width  +   self.padding) * self.piece_size #px-sizes of a field
        height_px = (height + 0*self.padding) * self.piece_size
        count = 0
        for row in fields:
            for f in row:
                self.drawField(f,x,y, width,height)
                x += width_px
            x = 0
            y += height_px
        pg.display.flip()
