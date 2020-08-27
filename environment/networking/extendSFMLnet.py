import pySFMLnet as p
from enum import Enum

class Values(Enum):
    u8 = 0
    u16 = 1
    u32 = 2
    u64 = 3
    i8 = 4
    i16 = 5
    i32 = 6
    i64 = 7
    fl = 8
    dl = 9
    str = 10

def read(*args):
    ret = []
    for f in args:
        if f == Values.u8:
            ret.append(p.read_uint8())
        elif f == Values.u16:
            ret.append(p.read_uint16())
        elif f == Values.u32:
            ret.append(p.read_uint32())
        elif f == Values.u64:
            ret.append(p.read_uint64())
        elif f == Values.i8:
            ret.append(p.read_int8())
        elif f == Values.i16:
            ret.append(p.read_int16())
        elif f == Values.i32:
            ret.append(p.read_int32())
        elif f == Values.i64:
            ret.append(p.read_int64())
        elif f == Values.fl:
            ret.append(p.read_float())
        elif f == Values.dl:
            ret.append(p.read_double())
        elif f == Values.str:
            ret.append(p.read_string())
    return ret

def write(*args):
    current_write_function = p.write_uint8
    for f in args:
        if f == Values.u8:
            current_write_function = p.write_uint8
        elif f == Values.u16:
            current_write_function = p.write_uint16
        elif f == Values.u32:
            current_write_function = p.write_uint32
        elif f == Values.u64:
            current_write_function = p.write_uint64
        elif f == Values.i8:
            current_write_function = p.write_int8
        elif f == Values.i16:
            current_write_function = p.write_int16
        elif f == Values.i32:
            current_write_function = p.write_int32
        elif f == Values.i64:
            current_write_function = p.write_int64
        elif f == Values.fl:
            current_write_function = p.write_float
        elif f == Values.dl:
            current_write_function = p.write_double
        elif f == Values.str:
            current_write_function = p.write_string
        else:
            current_write_function(f)
