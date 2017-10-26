from PyQt5 import uic
import os

f = open('UserInterface.py', 'w+')
uic.compileUi('plotgui.ui', f)
f.close()
