# Version of the Python module.
__version__ = "0.2.6"


def print_(msg, print_it=True):
    if print_it:
        pre = "openworm.ai  >>> "
        txt = str(msg).replace("\n", "\n" + pre) if msg is not None else msg
        print("%s %s" % (pre, txt))
