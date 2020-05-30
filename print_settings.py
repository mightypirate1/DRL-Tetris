import experiments.presets
from aux.settings_printer import settings_printer
import docopt

docoptstring = \
'''
Settings printer!

Usage:
    settings_printer.py <settings-file1> [--cmp <settings-file2>]
'''
docoptsettings = docopt.docopt(docoptstring)
sp = settings_printer(docoptsettings["<settings-file1>"])
if docoptsettings["--cmp"]:
    s2 = settings_printer(docoptsettings["<settings-file2>"])
    sp.compare(s2)
else:
    sp._print()
