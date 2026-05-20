import argparse
import os
import time

import serial
from serial import SerialException
from serial.tools import list_ports


class SerialRelayController:
    """通过串口发送固定 HEX 指令控制继电器。"""

    RELAY_ON_FRAME = bytes.fromhex("A0 01 01 A2")
    RELAY_OFF_FRAME = bytes.fromhex("A0 01 00 A1")

    def __init__(
        self,
        port,
        baudrate=9600,
        timeout=1.0,
        write_timeout=1.0,
        auto_open=True,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.serial_port = None

        if auto_open:
            self.open()

    def open(self):
        if self.serial_port and self.serial_port.is_open:
            return

        self.serial_port = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
            write_timeout=self.write_timeout,
        )
        self.serial_port.reset_input_buffer()
        self.serial_port.reset_output_buffer()
        print(f"已打开继电器串口: {self.port} @ {self.baudrate}")

    def close(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print(f"已关闭继电器串口: {self.port}")

    def send_hex(self, frame):
        if isinstance(frame, str):
            frame = bytes.fromhex(frame)

        if not self.serial_port or not self.serial_port.is_open:
            self.open()

        written = self.serial_port.write(frame)
        self.serial_port.flush()
        print(f"继电器发送: {frame.hex(' ').upper()} ({written} bytes)")
        return written

    def relay_on(self):
        self.send_hex(self.RELAY_ON_FRAME)

    def relay_off(self):
        self.send_hex(self.RELAY_OFF_FRAME)

    def set_close_finger(self, close_finger_uses_relay_on=True):
        if close_finger_uses_relay_on:
            self.relay_on()
        else:
            self.relay_off()

    def set_release_finger(self, close_finger_uses_relay_on=True):
        if close_finger_uses_relay_on:
            self.relay_off()
        else:
            self.relay_on()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_available_ports():
    return list(list_ports.comports())


def print_available_ports():
    ports = get_available_ports()
    if not ports:
        print("当前系统未发现可用串口。")
        return

    print("当前可用串口:")
    for port in ports:
        desc = port.description or "unknown"
        hwid = port.hwid or "unknown"
        print(f"  {port.device} | {desc} | {hwid}")


def main():
    parser = argparse.ArgumentParser(description="Linux 串口继电器控制")
    parser.add_argument("action", nargs="?", choices=["on", "off"], help="on=发送 A0 01 01 A2, off=发送 A0 01 00 A1")
    parser.add_argument("--port", help="串口设备，例如 /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=9600, help="波特率，默认 9600")
    parser.add_argument("--timeout", type=float, default=1.0, help="串口超时，默认 1.0 秒")
    parser.add_argument("--hold-seconds", type=float, default=0.0, help="发送后额外等待的秒数")
    parser.add_argument("--list-ports", action="store_true", help="列出当前系统识别到的串口")
    args = parser.parse_args()

    if args.list_ports:
        print_available_ports()
        return

    if not args.action or not args.port:
        parser.error("未使用 --list-ports 时，必须提供 action 和 --port")

    if not os.path.exists(args.port):
        print(f"串口不存在: {args.port}")
        print_available_ports()
        print("请检查:")
        print("  1. 继电器USB转串口是否已插好")
        print("  2. 实际设备名是否是 /dev/ttyACM0 或其他编号")
        print("  3. 是否在虚拟机/容器里，设备没有透传进来")
        return

    try:
        with SerialRelayController(
            port=args.port,
            baudrate=args.baudrate,
            timeout=args.timeout,
        ) as relay:
            if args.action == "on":
                relay.relay_on()
            else:
                relay.relay_off()

            if args.hold_seconds > 0:
                time.sleep(args.hold_seconds)
    except SerialException as exc:
        print(f"串口打开失败: {exc}")
        print_available_ports()
        if "Permission denied" in str(exc):
            print("当前像是权限问题。常见做法是把用户加入 dialout 组后重新登录。")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
# python LCUS.py off --port /dev/ttyUSB0
# python LCUS.py on --port /dev/ttyUSB0