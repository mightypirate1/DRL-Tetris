from pathlib import Path

### Path-like joins for keys
def keyjoin(x,y):
    if not x:
        return y
    if not y:
        return x
    return str(Path(x)/Path(y))
