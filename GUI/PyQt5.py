import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox

# class MyApp(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.initUI()
#
#     def initUI(self):
#         self.setGeometry(100, 100, 400, 300)  # Set the window size and position
#         self.setWindowTitle('Simple PyQt5 App')  # Set the window title
#
#         # Create a button
#         btn = QPushButton('Click Me', self)
#         btn.setToolTip('Click to show a message box')  # Tooltip for the button
#         btn.move(150, 150)  # Set button position
#         btn.clicked.connect(self.showMessageBox)  # Connect the button click event to the method
#
#         self.show()  # Show the window
#
#     def showMessageBox(self):
#         # Create a message box
#         msg_box = QMessageBox(self)
#         msg_box.setWindowTitle('Message Box')
#         msg_box.setText('You clicked the button!')
#         msg_box.setIcon(QMessageBox.Information)
#         msg_box.setStandardButtons(QMessageBox.Ok)
#         msg_box.exec_()  # Show the message box
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)  # Create the application object
#     ex = MyApp()  # Create the MyApp instance
#     sys.exit(app.exec_())  # Start the application event loop
