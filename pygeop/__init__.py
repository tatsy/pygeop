import os

current_dir = os.path.dirname(os.path.realpath(__file__))
modules = [ m for m in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, m)) ]
modules = filter(lambda m : m != '__init__.py' and not m.startswith('.'), modules)
for m in modules:
    base = m.split('.')[0]
    cmd = 'from .%s import *' % base
    exec(cmd)
