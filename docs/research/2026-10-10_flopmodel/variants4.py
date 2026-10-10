import sys, os, json
sys.path.insert(0, os.path.expanduser("~/scratch"))
from variants3 import *
T = json.load(open('kink_tables.json'))
def set_kas(role, table):
    def p(mod):
        mod._KAS_K = dict(mod._KAS_K); mod._KAS_K[role] = list(table)
    return p
if __name__ == "__main__":
    run([set_kas("H", T["2.5"]["active"])], "2a H = kink-active(2.5) cols")
    run([set_kas("D", T["2.5"]["active"])], "2b D = kink-active(2.5) rows")
    run([set_kas("H", T["2.5"]["active"]), set_kas("D", T["2.5"]["active"])], "2a+2b")
    run([set_kas("H", T["3.0"]["active"])], "2a' H = kink-active(3.0)")
    run([set_kas("D", T["3.0"]["active"])], "2b' D = kink-active(3.0)")
    run([set_kas("R", T["2.5"]["active"])], "(R = kink-active 2.5, diag)")
