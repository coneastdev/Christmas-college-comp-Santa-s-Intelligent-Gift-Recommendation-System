import sys
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtQuick import QQuickView

def main():
    app = QGuiApplication(sys.argv)
    view = QQuickView()
    view.setSource(QUrl("main.qml"))
    view.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
    view.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()