import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt  # Import the Qt module for alignment

class DarkRecorderGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.runThreads = True 

        self.setWindowTitle("Recorder")
        self.setGeometry(100, 100, 400, 200)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create a vertical layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Label in the middle
        label = QLabel("Recording Status")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: white; font-family: Arial; font-size: 18px;")

        # Button at the bottom
        self.record_button = QPushButton("Start Recording")
        self.record_button.setStyleSheet("background-color: orange; color: white; font-family: Arial; font-size: 16px;")
        self.record_button.clicked.connect(self.toggle_recording)

        # Add label and button to the layout
        layout.addWidget(label)
        layout.addWidget(self.record_button)

        # Recording status
        self.recording = False

    def toggle_recording(self):
        if self.recording:
            self.record_button.setText("Start Recording")
            self.recording = False
            self.runThreads = True
            self.threading()
            # Stop recording logic here (e.g., save the recording)
        else:
            self.record_button.setText("Stop Recording")
            self.recording = True
            self.runThreads = False
            # Start recording logic here (e.g., start recording)

    def threading(self):
        print(f"{self.runThreads} threading")
        while self.runThreads:
            print("recording...")
        
        print("STOP RECORDING")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("QMainWindow { background-color: #333; }")
    window = DarkRecorderGUI()
    window.show()
    sys.exit(app.exec_())
