import aux.utils as utils
import docopt
import pickle

class settings_printer:
    def __init__(self, s):
        if type(s) not in [str, dict]:
            raise Exception("give me a file or a dict!")
        if type(s) is not dict:
            s = self.load_file(s)
        self.settings = utils.parse_settings(s)
        self.title = s["run-id"] if "run-id" in s else "unknown"
    def load_file(self,s):
        with open(s, 'rb') as f:
            settings = pickle.load(f)
        return utils.parse_settings(settings)
    def print(self):
        print("--- {} settings---".format(self.title))
        for key in sorted(self.settings.keys()):
            print("\t{}".format(key).ljust(35),self.settings[key])
    def compare(self, x):
        if type(x) not in [settings_printer, dict]:
            raise Exception("give me a settings_printer or a dict!")
        other = x.settings if type(x) is settings_printer else x
        for key in list(self.settings.keys())+list(other.keys()):
            if key in self.settings and key in other:
                if self.settings[key] == other[key]:
                    continue
            print("---", key, "---")
            for candidate in [self.settings, other]:
                if key in candidate:
                    print("\t{}".format(candidate["run-id"]).ljust(20),candidate[key])

if __name__ == "__main__":
    from agents.vector_agent import vector_agent, vector_agent_trainer
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
        sp.print()
