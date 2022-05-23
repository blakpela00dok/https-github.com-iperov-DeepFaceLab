from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

class QImageDB():
    @staticmethod
    def initialize(image_path):
        QImageDB.intro = QImage ( str(image_path / 'intro.png') )
