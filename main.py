import sys
import os

# 將 src 加入路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from PyQt5 import QtWidgets
from vp_calib.gui import Ui_MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
