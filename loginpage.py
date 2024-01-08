import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QDialog
from PyQt5.QtCore import pyqtSignal, Qt
import subprocess

class RegisterPage(QWidget):
    def __init__(self, credentials):
        super().__init__()

        self.credentials = credentials
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Register Page')
        self.setGeometry(300, 300, 300, 150)

        # Widgets
        self.lbl_new_username = QLabel('New Username:')
        self.lbl_new_password = QLabel('New Password:')
        self.txt_new_username = QLineEdit()
        self.txt_new_password = QLineEdit()
        self.btn_register = QPushButton('Register')

        # Password field settings
        self.txt_new_password.setEchoMode(QLineEdit.Password)

        # Layout
        vbox = QVBoxLayout()
        hbox_username = QHBoxLayout()
        hbox_password = QHBoxLayout()

        hbox_username.addWidget(self.lbl_new_username)
        hbox_username.addWidget(self.txt_new_username)

        hbox_password.addWidget(self.lbl_new_password)
        hbox_password.addWidget(self.txt_new_password)

        vbox.addLayout(hbox_username)
        vbox.addLayout(hbox_password)
        vbox.addWidget(self.btn_register)

        self.setLayout(vbox)

        # Event Handling
        self.btn_register.clicked.connect(self.register)

        self.show()

    def register(self):
        new_username = self.txt_new_username.text()
        new_password = self.txt_new_password.text()

        # Check if the username already exists
        if new_username in self.credentials:
            QMessageBox.warning(self, 'Registration Error', 'Username already exists. Please choose a different one.')
            return

        # Store the new credentials in the dictionary
        self.credentials[new_username] = new_password

        # Save credentials to a file (for demonstration purposes, you might want to use a more secure storage method)
        with open('credentials.txt', 'a') as file:
            file.write(f'{new_username}:{new_password}\n')

        print('Registration successful!')
        self.close()


class ResetPasswordPage(QDialog):
    def __init__(self, credentials, username):
        super().__init__()

        self.credentials = credentials
        self.username = username
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Reset Password')
        self.setGeometry(300, 300, 300, 150)

        # Widgets
        self.lbl_new_password = QLabel('New Password:')
        self.txt_new_password = QLineEdit()
        self.btn_reset_password = QPushButton('Reset Password')

        # Password field settings
        self.txt_new_password.setEchoMode(QLineEdit.Password)

        # Layout
        vbox = QVBoxLayout()
        hbox_password = QHBoxLayout()

        hbox_password.addWidget(self.lbl_new_password)
        hbox_password.addWidget(self.txt_new_password)

        vbox.addLayout(hbox_password)
        vbox.addWidget(self.btn_reset_password)

        self.setLayout(vbox)

        # Event Handling
        self.btn_reset_password.clicked.connect(self.reset_password)

        self.show()

    def reset_password(self):
        new_password = self.txt_new_password.text()

        # Update the password in the credentials dictionary
        self.credentials[self.username] = new_password

        # Save updated credentials to the file
        with open('credentials.txt', 'w') as file:
            for u, p in self.credentials.items():
                file.write(f'{u}:{p}\n')

        print('Password reset successful!')
        self.accept()


class ForgotPasswordPage(QWidget):
    def __init__(self, credentials):
        super().__init__()

        self.credentials = credentials
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Forgot Password')
        self.setGeometry(300, 300, 300, 150)

        # Widgets
        self.lbl_username = QLabel('Username:')
        self.txt_username = QLineEdit()
        self.btn_reset_password = QPushButton('Reset Password')

        # Layout
        vbox = QVBoxLayout()
        hbox_username = QHBoxLayout()

        hbox_username.addWidget(self.lbl_username)
        hbox_username.addWidget(self.txt_username)

        vbox.addLayout(hbox_username)
        vbox.addWidget(self.btn_reset_password)

        self.setLayout(vbox)

        # Event Handling
        self.btn_reset_password.clicked.connect(self.reset_password)

        self.show()

    def reset_password(self):
        username = self.txt_username.text()

        # Check if the username exists
        if username in self.credentials:
            reset_password_dialog = ResetPasswordPage(self.credentials, username)
            if reset_password_dialog.exec_() == QDialog.Accepted:
                print('Password reset successful!')
            else:
                print('Password reset canceled.')
        else:
            QMessageBox.warning(self, 'Password Reset Error', 'Username not found. Please enter a valid username.')


class PrintRunner(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Print Runner')
        self.setGeometry(400, 400, 300, 100)

        label = QLabel('Hello, World!')
        layout = QVBoxLayout()
        layout.addWidget(label)

        self.setLayout(layout)

        self.show()


class LoginPage(QWidget):
    successful_login = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Load existing credentials from the file or create an empty dictionary if the file doesn't exist
        self.credentials = {}
        try:
            with open('credentials.txt', 'r') as file:
                for line in file:
                    username, password = line.strip().split(':')
                    self.credentials[username] = password
        except FileNotFoundError:
            pass

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Login Page')
        self.setGeometry(300, 300, 300, 150)

        # Widgets
        self.lbl_username = QLabel('Username:')
        self.lbl_password = QLabel('Password:')
        self.txt_username = QLineEdit()
        self.txt_password = QLineEdit()
        self.btn_login = QPushButton('Login')
        self.btn_register = QPushButton('Register')
        self.btn_forgot_password = QPushButton('Forgot Password')

        # Password field settings
        self.txt_password.setEchoMode(QLineEdit.Password)

        # Layout
        vbox = QVBoxLayout()
        hbox_username = QHBoxLayout()
        hbox_password = QHBoxLayout()
        hbox_buttons = QHBoxLayout()

        hbox_username.addWidget(self.lbl_username)
        hbox_username.addWidget(self.txt_username)

        hbox_password.addWidget(self.lbl_password)
        hbox_password.addWidget(self.txt_password)

        hbox_buttons.addWidget(self.btn_login)
        hbox_buttons.addWidget(self.btn_register)
        hbox_buttons.addWidget(self.btn_forgot_password)

        vbox.addLayout(hbox_username)
        vbox.addLayout(hbox_password)
        vbox.addLayout(hbox_buttons)

        self.setLayout(vbox)

        # Event Handling
        self.btn_login.clicked.connect(self.login)
        self.btn_register.clicked.connect(self.show_register_page)
        self.btn_forgot_password.clicked.connect(self.show_forgot_password_page)

        self.show()

    def login(self):
        username = self.txt_username.text()
        password = self.txt_password.text()

        # Check credentials
        if username in self.credentials and self.credentials[username] == password:
            print('Login successful!')
            self.successful_login.emit()  # Emit signal on successful login
        else:
            QMessageBox.warning(self, 'Login Error', 'Invalid credentials. Please try again.')

    def show_register_page(self):
        self.register_page = RegisterPage(self.credentials)
        self.register_page.show()

    def show_forgot_password_page(self):
        self.forgot_password_page = ForgotPasswordPage(self.credentials)
        self.forgot_password_page.show()

    def on_successful_login(self):
        # Run print.py using subprocess
        try:
            subprocess.run(['python', 'print.py'], check=True)
        except subprocess.CalledProcessError as e:
            print(f'Error running print.py: {e}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_page = LoginPage()

    def on_successful_login():
        login_page.on_successful_login()

    login_page.successful_login.connect(on_successful_login)

    sys.exit(app.exec_())