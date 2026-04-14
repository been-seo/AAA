"""
PyQt5 관제 GUI (별도 프로세스)
- 항공기 생성 윈도우
- 지시 입력 윈도우
- 메인 시뮬레이션 프로세스와 Pipe 통신
"""
from PyQt5 import QtWidgets, QtCore
import sys
import time


def launch_gui_process(conn):
    app = QtWidgets.QApplication(sys.argv)
    awaiting_coords = False

    class CreateAircraftWindow(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("항공기 생성")
            self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
            layout = QtWidgets.QGridLayout()
            layout.addWidget(QtWidgets.QLabel("CALLSIGN"), 0, 0)
            self.callsign = QtWidgets.QLineEdit()
            layout.addWidget(self.callsign, 0, 1)
            layout.addWidget(QtWidgets.QLabel("LAT (DDMMSS)"), 1, 0)
            self.lat = QtWidgets.QLineEdit()
            layout.addWidget(self.lat, 1, 1)
            layout.addWidget(QtWidgets.QLabel("LON (DDDMMSS)"), 2, 0)
            self.lon = QtWidgets.QLineEdit()
            layout.addWidget(self.lon, 2, 1)
            mouse_btn = QtWidgets.QPushButton("마우스로 위치 입력")
            mouse_btn.clicked.connect(self._start_mouse)
            layout.addWidget(mouse_btn, 3, 0, 1, 2)
            layout.addWidget(QtWidgets.QLabel("ALT(ft)"), 4, 0)
            self.alt = QtWidgets.QLineEdit("10000")
            layout.addWidget(self.alt, 4, 1)
            layout.addWidget(QtWidgets.QLabel("SPD(kts)"), 5, 0)
            self.spd = QtWidgets.QLineEdit("300")
            layout.addWidget(self.spd, 5, 1)
            layout.addWidget(QtWidgets.QLabel("HDG(\u00b0)"), 6, 0)
            self.hdg = QtWidgets.QLineEdit("360")
            layout.addWidget(self.hdg, 6, 1)
            btn = QtWidgets.QPushButton("생성")
            btn.clicked.connect(self._submit)
            layout.addWidget(btn, 7, 0, 1, 2)
            self.setLayout(layout)
            self.show()
            self.raise_()
            self.activateWindow()

        def _submit(self):
            conn.send({
                "type": "create_aircraft",
                "callsign": self.callsign.text(),
                "lat": self.lat.text(), "lon": self.lon.text(),
                "alt": self.alt.text(), "spd": self.spd.text(), "hdg": self.hdg.text(),
            })
            self.close()

        def _start_mouse(self):
            nonlocal awaiting_coords
            awaiting_coords = True
            conn.send("await_coords")
            self.hide()

            def check():
                nonlocal awaiting_coords
                if conn.poll():
                    msg = conn.recv()
                    if isinstance(msg, dict) and msg["type"] == "coords":
                        self.lat.setText(str(msg["data"]["lat"]))
                        self.lon.setText(str(msg["data"]["lon"]))
                        self.show()
                        self.raise_()
                        self.activateWindow()
                        awaiting_coords = False
                        return
                if awaiting_coords:
                    QtCore.QTimer.singleShot(100, check)
            check()

    class InstructionWindow(QtWidgets.QWidget):
        def __init__(self, data):
            super().__init__()
            self.setWindowTitle(f"{data['callsign']} 지시")
            self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
            layout = QtWidgets.QGridLayout()
            layout.addWidget(QtWidgets.QLabel("ALT(ft)"), 0, 0)
            self.alt = QtWidgets.QLineEdit(str(int(data['alt'])))
            layout.addWidget(self.alt, 0, 1)
            layout.addWidget(QtWidgets.QLabel("SPD(kts)"), 1, 0)
            self.spd = QtWidgets.QLineEdit(str(int(data['spd'])))
            layout.addWidget(self.spd, 1, 1)
            layout.addWidget(QtWidgets.QLabel("HDG(\u00b0)"), 2, 0)
            self.hdg = QtWidgets.QLineEdit(str(int(data['hdg'])))
            layout.addWidget(self.hdg, 2, 1)
            btn = QtWidgets.QPushButton("지시")
            btn.clicked.connect(self._submit)
            layout.addWidget(btn, 3, 0, 1, 2)
            self.setLayout(layout)
            self.activateWindow()

        def _submit(self):
            try:
                conn.send({
                    "type": "instruction",
                    "alt": float(self.alt.text()),
                    "spd": float(self.spd.text()),
                    "hdg": float(self.hdg.text()),
                })
            except ValueError:
                pass
            self.close()

    # 메인 이벤트 루프
    windows = {}
    while True:
        try:
            if not awaiting_coords and conn.poll():
                msg = conn.recv()
                if isinstance(msg, dict) and msg["type"] == "action":
                    act = msg["action"]
                    if act == "create_aircraft":
                        if act not in windows or not windows[act].isVisible():
                            windows[act] = CreateAircraftWindow()
                            windows[act].show()
                        windows[act].activateWindow()
                        windows[act].raise_()
                    elif act == "instruction":
                        key = f"inst_{msg['aircraft_data']['callsign']}"
                        if key not in windows or not windows[key].isVisible():
                            windows[key] = InstructionWindow(msg["aircraft_data"])
                            windows[key].show()
                        windows[key].activateWindow()
                        windows[key].raise_()
                    elif act == "shutdown":
                        break

            closed = [k for k, w in windows.items() if not w.isVisible()]
            for k in closed:
                del windows[k]

            QtWidgets.QApplication.processEvents()
            time.sleep(0.05)
        except (EOFError, BrokenPipeError):
            break

    sys.exit(0)
