import sys, pygame as pg, numpy as np
pg.init()

size = width, height = 860, 440

piece_size = 20
field_row_size = 4

screen = pg.display.set_mode(size)
screen.fill([0,0,0])

colormap = [[20,20,20],
            [255,0,0],
            [0,255,0],
            [115,145,255],
            [255,0,255],
            [0,255,255],
            [255,255,0],
            [255,255,255],
            [170,170,170]]

def screenSize(newwidth, newheight):
    size = width, height = newwidth, newheight
    screen = pg.display.set_mode(size)
    screen.fill([0,0,0])

def pieceSize(newsize):
    global piece_size
    piece_size = newsize
    screen.fill([0,0,0])

def rowSize(newsize):
    global field_row_size
    field_row_size = newsize
    screen.fill([0,0,0])

def drawField(field, x, y):
    field_width = field[0].size
    rect = pg.Rect(x,y,piece_size,piece_size)
    count = 0
    for i in np.nditer(field):
        screen.fill(colormap[i], rect=rect)
        rect.left += piece_size
        count += 1
        if count == field_width:
            rect.left = x
            rect.top += piece_size
            count = 0

def pollEvents():
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
        if event.type == pg.KEYDOWN: return True
    return False

def drawAllFields(fields):
    ret = pollEvents()
    x = 0
    y = 0
    height, width = fields[0].shape
    width = (width + 1) * piece_size
    height = (height + 1) * piece_size
    count = 0
    for f in fields:
        drawField(f,x,y)
        x += width
        count += 1
        if count == field_row_size:
            count = 0
            x = 0
            y += height
    pg.display.flip()
    return ret
