"""
NEURONEXUS InspectAI — Dedicated Coca-Cola 250ml Inspection Station
Classic Top-Down UI Architecture:
- Top Viewport: Camera ABOVE taking ~75% screen height over DOMS 12.5 cm Conveyor Bed Box Jig.
- Bottom Console: Detection DOWNSIDE across full width showing 3 section status and isolated failure.
Integrated with Non-Blocking DroidCam Stream & Arduino Nano Pin D2 IR Trigger & Pin D9 Pusher.
"""

import sys
import os
import time
import socket
import threading
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

# GUI Theme Colors
BG_DARK = "#020617"
BG_PANEL = "#040814"
BG_CARD = "#0B1329"
BORDER_CLR = "#1E293B"
CYAN = "#00F2FE"
COKE_RED = "#F40009"
PASS_CLR = "#10B981"
DEF_CLR = "#EF4444"
PEND_CLR = "#F59E0B"
TXT_MAIN = "#F8FAFC"
TXT_MUTED = "#94A3B8"

DEFAULT_CAMERA_IP = "http://10.91.85.242:4747/video"
FALLBACK_CAMERA_IP = "http://10.209.18.47:4747/video"
DEFAULT_ARDUINO_PORT = "COM3"
BAUD_RATE = 9600


class PusherController:
    """Controls conveyor synchronization & servo rejection via Arduino Nano on Pin D9 & IR Sensor on Pin D2."""
    def __init__(self, preferred_port: str = DEFAULT_ARDUINO_PORT):
        self.preferred_port = preferred_port
        self.ser = None
        self.is_connected = False
        self.last_fire_time = 0.0
        self.debounce_seconds = 1.0
        self.last_trigger_time = 0.0
        self.auto_connect()

    def auto_connect(self):
        try:
            import serial
            import serial.tools.list_ports
            available = [p.device for p in serial.tools.list_ports.comports()]
            candidates = [self.preferred_port] + [p for p in available if p != self.preferred_port]
            for p in candidates:
                try:
                    self.ser = serial.Serial(p, BAUD_RATE, timeout=0.05)
                    time.sleep(1.2)
                    self.is_connected = True
                    print(f"[InspectAI] Arduino Nano connected on {p} (IR Sensor D2 & Pusher D9 Active)")
                    return
                except Exception:
                    continue
        except ImportError:
            print("[InspectAI] PySerial not installed. Emulating hardware pusher.")
        self.is_connected = False

    def check_ir_trigger(self) -> bool:
        """Polls for 'TRIGGER' from Arduino Nano Pin D2 with strict single-bottle lockout."""
        if self.is_connected and self.ser:
            try:
                found_trigger = False
                while self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if "TRIGGER" in line:
                        found_trigger = True
                
                if found_trigger:
                    now = time.time()
                    # Minimum 1.8s cooldown: guarantees 1 bottle = 1 inspection
                    if (now - self.last_trigger_time) >= 1.8:
                        self.last_trigger_time = now
                        print("[InspectAI] >>> HARDWARE IR BEAM BROKEN (Pin D2): Bottle entered box (1x Trigger)! <<<")
                        return True
            except Exception as e:
                print(f"[InspectAI] Serial read error: {e}")
                self.is_connected = False
        return False

    def send_decision(self, is_pass: bool) -> bool:
        """Sends 'P' (Pass) or 'R' (Reject) to Arduino to coordinate conveyor & timed servo sweep."""
        cmd = b"P" if is_pass else b"R"
        if not is_pass:
            def _beep():
                try:
                    if sys.platform == "win32":
                        import winsound
                        winsound.Beep(1200, 180)
                    else:
                        print("\a", end="", flush=True)
                except Exception:
                    pass
            threading.Thread(target=_beep, daemon=True).start()

        if self.is_connected and self.ser:
            try:
                self.ser.write(cmd)
                self.ser.flush()
                print(f"[InspectAI] >>> Serial '{cmd.decode()}' sent to Arduino <<<")
                return True
            except Exception as e:
                print(f"[InspectAI] Serial write error: {e}")
                self.is_connected = False
                return False
        else:
            print(f"[InspectAI] >>> Simulated Serial '{cmd.decode()}' Triggered <<<")
            return True

    def trigger_rejection(self) -> bool:
        return self.send_decision(is_pass=False)


class CameraController:
    """Non-blocking DroidCam Wi-Fi IP and USB Webcam controller."""
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.current_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.active_source: str = ""

    def connect(self, target_ip: str = "") -> Tuple[bool, str]:
        self.disconnect()
        candidates = []
        if target_ip and target_ip.strip():
            candidates.append(target_ip.strip())
        candidates.extend([DEFAULT_CAMERA_IP, FALLBACK_CAMERA_IP, 0])

        deduped = []
        for c in candidates:
            if c not in deduped:
                deduped.append(c)

        for source in deduped:
            if isinstance(source, str) and source.startswith("http://"):
                try:
                    host = source.split("//")[1].split(":")[0]
                    port = int(source.split(":")[2].split("/")[0])
                    s = socket.create_connection((host, port), timeout=0.4)
                    s.close()
                except Exception:
                    continue

            cap = cv2.VideoCapture(source)
            if cap and cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self.cap = cap
                    self.active_source = str(source)
                    self.is_running = True
                    self.current_frame = test_frame
                    self.thread = threading.Thread(target=self._capture_worker, daemon=True)
                    self.thread.start()
                    return True, f"Connected: {self.active_source}"
                else:
                    cap.release()
        return False, "Unable to reach DroidCam or local webcam"

    def disconnect(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.6)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        with self.lock:
            self.current_frame = None
        self.active_source = ""

    def _capture_worker(self):
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.current_frame = frame
            else:
                time.sleep(0.02)
            time.sleep(0.005)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None


class InspectAIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NEURONEXUS InspectAI — Dedicated Coca-Cola 250ml Quality Inspector")
        self.geometry("1280x780")
        self.minsize(1080, 680)
        self.configure(bg=BG_DARK)

        self.pusher = PusherController()
        self.cam = CameraController()

        try:
            from ai_model import AIInspectionEngine, COCA_COLA_250ML
            self.ai_engine = AIInspectionEngine()
            self.BOTTLE_NAME = COCA_COLA_250ML
        except Exception:
            self.ai_engine = None
            self.BOTTLE_NAME = "Coca-Cola 250ml"

        self.total_count = 0
        self.pass_count = 0
        self.defect_count = 0
        self.stencilScale = 1.0
        self.boxOffsetX = 0
        self.crit_widgets: List[Dict[str, Any]] = []

        self.is_detecting = True
        self.bottle_state = "IDLE"
        self.latched_ev = None
        self.empty_streak = 0
        self.latch_time = 0.0

        self._build_ui()
        self._start_stream_loop()
        threading.Thread(target=lambda: (time.sleep(0.3), self.cam.connect()), daemon=True).start()

    def _build_ui(self):
        # Universal Header
        hdr = tk.Frame(self, bg=BG_PANEL, height=48, bd=0)
        hdr.pack(side="top", fill="x")

        f_title = tk.Frame(hdr, bg=BG_PANEL)
        f_title.pack(side="left", padx=16, pady=8)
        tk.Label(f_title, text="⬡ NEURONEXUS", font=("Segoe UI", 12, "bold"), fg=CYAN, bg=BG_PANEL).pack(side="left")
        tk.Label(f_title, text="InspectAI", font=("Segoe UI", 12, "bold"), fg=TXT_MAIN, bg=BG_PANEL).pack(side="left", padx=4)

        tk.Label(hdr, text="ACTIVE LINE: COCA-COLA 250ml (15.2 cm)", font=("Segoe UI", 10, "bold"), fg=COKE_RED, bg=BG_PANEL).pack(side="left", padx=24)

        self.lbl_arduino = tk.Label(
            hdr, text=f"ARDUINO: {'PIN D2 IR + PIN D9 ARMED' if self.pusher.is_connected else 'SIMULATION'}",
            font=("Segoe UI", 9, "bold"), fg=PASS_CLR if self.pusher.is_connected else PEND_CLR, bg=BG_PANEL
        )
        self.lbl_arduino.pack(side="right", padx=16)

        # Camera Viewport (~75% Height)
        self.f_camera_top = tk.Frame(self, bg="#000000", bd=2, relief="solid", highlightbackground=COKE_RED)
        self.f_camera_top.pack(fill="both", expand=True, padx=12, pady=(8, 4))

        self.vcanvas = tk.Canvas(self.f_camera_top, bg="#000000", bd=0, highlightthickness=0)
        self.vcanvas.pack(fill="both", expand=True)

        # Stencil Controls Overlay (Top Right)
        f_scale = tk.Frame(self.vcanvas, bg="#040b1c", bd=1, relief="solid")
        f_scale.place(relx=0.98, rely=0.03, anchor="ne")
        
        tk.Button(f_scale, text="🎯 AUTO-ALIGN", font=("Segoe UI", 8, "bold"), bg="#022c22", fg="#22c55e", relief="flat", command=self._auto_align_box).pack(side="left", padx=3)
        tk.Label(f_scale, text="POS X:", font=("Segoe UI", 8, "bold"), fg=TXT_MUTED, bg="#040b1c").pack(side="left", padx=(4, 2))
        tk.Button(f_scale, text="◀", font=("Segoe UI", 8, "bold"), bg="#0b1329", fg=CYAN, relief="flat", width=2, command=lambda: self._shift_box(-20)).pack(side="left", padx=1)
        tk.Button(f_scale, text="▶", font=("Segoe UI", 8, "bold"), bg="#0b1329", fg=CYAN, relief="flat", width=2, command=lambda: self._shift_box(20)).pack(side="left", padx=1)

        tk.Label(f_scale, text="SCALE:", font=("Segoe UI", 8, "bold"), fg=TXT_MUTED, bg="#040b1c").pack(side="left", padx=(6, 2))
        tk.Button(f_scale, text="−", font=("Segoe UI", 8, "bold"), bg="#0b1329", fg=CYAN, relief="flat", width=2, command=lambda: self._adjust_scale(-0.05)).pack(side="left", padx=1)
        self.lbl_scale = tk.Label(f_scale, text="1.00x", font=("Consolas", 8, "bold"), fg=CYAN, bg="#040b1c", width=5)
        self.lbl_scale.pack(side="left", padx=1)
        tk.Button(f_scale, text="+", font=("Segoe UI", 8, "bold"), bg="#0b1329", fg=CYAN, relief="flat", width=2, command=lambda: self._adjust_scale(0.05)).pack(side="left", padx=1)

        # Camera & Arming Controls (Bottom Left)
        f_cam_ctrl = tk.Frame(self.vcanvas, bg="#040b1c", bd=1, relief="solid")
        f_cam_ctrl.place(relx=0.02, rely=0.96, anchor="sw")
        self.ent_ip = tk.Entry(f_cam_ctrl, font=("Consolas", 9), bg="#020611", fg=CYAN, bd=1, relief="solid", width=26)
        self.ent_ip.insert(0, DEFAULT_CAMERA_IP)
        self.ent_ip.pack(side="left", padx=4, pady=4)
        self.btn_toggle_cam = tk.Button(f_cam_ctrl, text="CONNECT", font=("Segoe UI", 8, "bold"), bg=COKE_RED, fg="#fff", relief="flat", padx=8, command=self._toggle_camera)
        self.btn_toggle_cam.pack(side="left", padx=4, pady=4)

        self.btn_detect_toggle = tk.Button(f_cam_ctrl, text="⏹ STOP DETECTING", font=("Segoe UI", 8, "bold"), bg="#10b981", fg="#ffffff", relief="flat", padx=10, command=self._toggle_detecting)
        self.btn_detect_toggle.pack(side="left", padx=4, pady=4)

        # SLM Status Bar
        f_slm_bar = tk.Frame(self, bg="#071126", height=36, bd=1, relief="solid")
        f_slm_bar.pack(side="bottom", fill="x", padx=12, pady=(0, 8))
        f_slm_bar.pack_propagate(False)

        tk.Label(f_slm_bar, text="🤖 AGENT-3 [SLM BUS]:", font=("Segoe UI", 8, "bold"), fg=CYAN, bg="#071126").pack(side="left", padx=(8, 4))
        self.lbl_slm_diag = tk.Label(f_slm_bar, text="[SLM Standby]: Awaiting next bottle via IR sensor...", font=("Consolas", 8), fg=TXT_MAIN, bg="#071126")
        self.lbl_slm_diag.pack(side="left", padx=4)

        # Bottom Inspection Console (~25% Height)
        self.f_console_bot = tk.Frame(self, bg=BG_PANEL, height=195, bd=1, relief="solid")
        self.f_console_bot.pack(side="bottom", fill="x", padx=12, pady=(0, 6))
        self.f_console_bot.pack_propagate(False)

        self.f_verdict = tk.Frame(self.f_console_bot, bg="#0f172a", width=340, bd=2, relief="solid")
        self.f_verdict.pack(side="left", fill="y", padx=8, pady=8)
        self.f_verdict.pack_propagate(False)

        self.lbl_verdict_title = tk.Label(self.f_verdict, text="⚪ AWAITING BOTTLE", font=("Segoe UI", 16, "bold"), fg="#ffffff", bg="#0f172a")
        self.lbl_verdict_title.pack(expand=True, pady=(10, 0))
        self.lbl_verdict_sub = tk.Label(self.f_verdict, text="IR Sensor Armed: Trigger will fire as bottle enters box", font=("Segoe UI", 8), fg=TXT_MUTED, bg="#0f172a", wraplength=310)
        self.lbl_verdict_sub.pack(expand=True, pady=(0, 10))

        f_crit = tk.Frame(self.f_console_bot, bg=BG_PANEL)
        f_crit.pack(side="left", fill="both", expand=True, padx=4, pady=8)

        tiles_meta = [
            ("FLUID LEVEL", "11.8cm MENISCUS", "Agent 1"),
            ("LABEL & BRAND", "3.4cm SCARLET RED", "Agent 2"),
            ("TARGET BOX FIT", "DOMS 12.5cm JIG", "Coordinator")
        ]

        self.crit_widgets = []
        for title, subtitle, agent_tag in tiles_meta:
            tile = tk.Frame(f_crit, bg=BG_CARD, bd=1, relief="solid")
            tile.pack(side="left", fill="both", expand=True, padx=4)
            tk.Label(tile, text=f"⬢ {title}", font=("Segoe UI", 8, "bold"), fg=CYAN, bg=BG_CARD).pack(anchor="w", padx=6, pady=(6, 2))
            w_word = tk.Label(tile, text="STANDBY", font=("Segoe UI", 10, "bold"), fg=TXT_MUTED, bg="#0f172a", bd=1, relief="solid", padx=6, pady=2)
            w_word.pack(anchor="w", padx=6, pady=4)
            tk.Label(tile, text=subtitle, font=("Segoe UI", 7), fg=TXT_MUTED, bg=BG_CARD).pack(anchor="w", padx=6)
            self.crit_widgets.append({"tile": tile, "word": w_word})

        f_stats = tk.Frame(self.f_console_bot, bg=BG_CARD, width=170, bd=1, relief="solid")
        f_stats.pack(side="right", fill="y", padx=8, pady=8)
        f_stats.pack_propagate(False)

        tk.Label(f_stats, text="PRODUCTION", font=("Segoe UI", 8, "bold"), fg=TXT_MUTED, bg=BG_CARD).pack(pady=(6, 2))
        self.lbl_tot = tk.Label(f_stats, text="0", font=("Consolas", 14, "bold"), fg=TXT_MAIN, bg=BG_CARD)
        self.lbl_tot.pack()
        self.lbl_pass = tk.Label(f_stats, text="0", font=("Consolas", 12, "bold"), fg=PASS_CLR, bg=BG_CARD)
        self.lbl_pass.pack()
        self.lbl_def = tk.Label(f_stats, text="0", font=("Consolas", 12, "bold"), fg=DEF_CLR, bg=BG_CARD)
        self.lbl_def.pack()

    def _shift_box(self, delta: int):
        self.boxOffsetX += delta

    def _auto_align_box(self):
        frame = self.cam.get_frame()
        if frame is not None:
            b, g, r = cv2.split(frame)
            is_red = (r > 120) & (r > g.astype(int) * 1.25) & (r > b.astype(int) * 1.25)
            red_pts = np.where(is_red)
            if len(red_pts[1]) > 50:
                mean_x = int(np.mean(red_pts[1]))
                self.boxOffsetX = mean_x - (frame.shape[1] // 2)

    def _adjust_scale(self, delta: float):
        self.stencilScale = max(0.6, min(1.6, self.stencilScale + delta))
        self.lbl_scale.config(text=f"{self.stencilScale:.2f}x")

    def _toggle_camera(self):
        if self.cam.is_running:
            self.cam.disconnect()
            self.btn_toggle_cam.config(text="CONNECT", bg=COKE_RED)
        else:
            ip = self.ent_ip.get().strip()
            self.btn_toggle_cam.config(text="CONNECTING...", bg=PEND_CLR)
            def _thread():
                ok, msg = self.cam.connect(ip)
                if ok:
                    self.btn_toggle_cam.config(text="DISCONNECT", bg=DEF_CLR)
                else:
                    self.btn_toggle_cam.config(text="RETRY", bg=PEND_CLR)
                    messagebox.showwarning("Camera Stream", msg)
            threading.Thread(target=_thread, daemon=True).start()

    def _toggle_detecting(self):
        self.is_detecting = not self.is_detecting
        self.bottle_state = "IDLE"
        self.latched_ev = None
        if self.is_detecting:
            self.btn_detect_toggle.config(text="⏹ STOP DETECTING", bg="#ef4444")
            self.lbl_verdict_title.config(text="⚪ ARMED: AWAITING BOTTLE", fg="#ffffff")
        else:
            self.btn_detect_toggle.config(text="▶ START DETECTING", bg="#10b981")
            self.lbl_verdict_title.config(text="⚪ STANDBY (PAUSED)", fg="#ffffff")

    def _start_stream_loop(self):
        self._stream_step()

    def _stream_step(self):
        vw = max(100, self.vcanvas.winfo_width())
        vh = max(100, self.vcanvas.winfo_height())
        frame = self.cam.get_frame()

        if frame is not None and self.ai_engine:
            if not self.is_detecting:
                ev = self.ai_engine._create_pending_evaluation("Click 'START DETECTING' to arm the optical inspector")
            else:
                ir_fired = self.pusher.check_ir_trigger()
                now = time.time()

                if self.pusher.is_connected:
                    if ir_fired:
                        ev = self.ai_engine.evaluate_inspection(frame)
                        self.latched_ev = ev
                        self.bottle_state = "LATCHED"
                        self.latch_time = now

                        self.total_count += 1
                        is_pass = (ev.verdict == "PASS")
                        if is_pass:
                            self.pass_count += 1
                        else:
                            self.defect_count += 1

                        self.pusher.send_decision(is_pass)
                    elif self.bottle_state == "LATCHED":
                        if (now - getattr(self, "latch_time", now)) < 1.8:
                            ev = self.latched_ev
                        else:
                            self.bottle_state = "IDLE"
                            self.latched_ev = None
                            ev = self.ai_engine._create_pending_evaluation("Armed: Awaiting next bottle via IR sensor...")
                    else:
                        ev = self.ai_engine._create_pending_evaluation("Armed: Awaiting next bottle via IR sensor...")
                else:
                    has_bottle = self.ai_engine.check_bottle_presence(frame)
                    if not has_bottle:
                        self.empty_streak += 1
                        if self.empty_streak >= 3:
                            self.bottle_state = "IDLE"
                            self.latched_ev = None
                        ev = self.ai_engine._create_pending_evaluation("Armed: Awaiting bottle in box...")
                    else:
                        self.empty_streak = 0
                        if self.bottle_state == "IDLE":
                            ev = self.ai_engine.evaluate_inspection(frame)
                            self.latched_ev = ev
                            self.bottle_state = "LATCHED"
                            self.latch_time = now
                            self.total_count += 1
                            if ev.verdict == "PASS":
                                self.pass_count += 1
                            else:
                                self.defect_count += 1
                            if ev.rejection_needed:
                                self.pusher.trigger_rejection()
                        else:
                            if (now - getattr(self, "latch_time", now)) < 1.8:
                                ev = self.latched_ev
                            else:
                                self.bottle_state = "IDLE"
                                self.latched_ev = None
                                ev = self.ai_engine._create_pending_evaluation("Armed: Awaiting next bottle...")

            self._update_inspection_ui(ev)
            display_img = self._render_calibrated_hud(frame, vw, vh, ev)
        else:
            display_img = self._generate_simulated_station(vw, vh)

        tk_img = ImageTk.PhotoImage(image=display_img)
        self.vcanvas.delete("all")
        self.vcanvas.create_image(vw // 2, vh // 2, image=tk_img)
        self.vcanvas._current_img = tk_img
        self.after(30, self._stream_step)

    def _update_inspection_ui(self, ev):
        box_color = getattr(ev, 'box_color', 'WHITE')

        if box_color == "WHITE":
            bg_col = "#0f172a"
            self.f_verdict.config(bg=bg_col, highlightbackground="#ffffff")
            self.lbl_verdict_title.config(text="⚪ AWAITING BOTTLE", fg="#ffffff", bg=bg_col)
            self.lbl_verdict_sub.config(text=ev.primary_defect_reason, bg=bg_col)
            for i in range(len(self.crit_widgets)):
                self.crit_widgets[i]["tile"].config(bg=BG_CARD)
                self.crit_widgets[i]["word"].config(text="STANDBY", fg=TXT_MUTED, bg="#0f172a")
            self.lbl_tot.config(text=str(self.total_count))
            self.lbl_pass.config(text=str(self.pass_count))
            self.lbl_def.config(text=str(self.defect_count))
            return

        is_pass = (ev.verdict == "PASS")
        bg_col = "#022C22" if is_pass else "#450A0A"
        bd_col = PASS_CLR if is_pass else DEF_CLR

        self.f_verdict.config(bg=bg_col, highlightbackground=bd_col)
        if is_pass:
            self.lbl_verdict_title.config(text="✔ PASS", fg=PASS_CLR, bg=bg_col)
            self.lbl_verdict_sub.config(text="Fluid, Label & Symmetry Verified", bg=bg_col)
        else:
            self.lbl_verdict_title.config(text="✖ DEFECT DETECTED", fg=DEF_CLR, bg=bg_col)
            self.lbl_verdict_sub.config(text=ev.primary_defect_reason, bg=bg_col)

        self.lbl_tot.config(text=str(self.total_count))
        self.lbl_pass.config(text=str(self.pass_count))
        self.lbl_def.config(text=str(self.defect_count))
        self.lbl_slm_diag.config(text=getattr(ev, "slm_diagnosis", ""), fg=PASS_CLR if is_pass else "#fca5a5")

        for i, chk in enumerate(ev.checks):
            if i < len(self.crit_widgets):
                w_tile = self.crit_widgets[i]["tile"]
                w_lbl = self.crit_widgets[i]["word"]
                if chk.status == "PASS":
                    w_tile.config(bg=BG_CARD)
                    w_lbl.config(text="DETECTED", fg=PASS_CLR, bg="#022c22")
                else:
                    w_tile.config(bg="#3b0f15")
                    w_lbl.config(text=chk.detail[:16].upper(), fg=DEF_CLR, bg="#450a0a")

    def _render_calibrated_hud(self, frame: np.ndarray, vw: int, vh: int, ev: Any) -> Image.Image:
        fh, fw = frame.shape[:2]
        scale = min(vw / float(fw), vh / float(fh))
        nw, nh = int(fw * scale), int(fh * scale)
        resized = cv2.resize(frame, (nw, nh))

        hud = np.zeros((vh, vw, 3), dtype=np.uint8)
        x_off = (vw - nw) // 2
        y_off = (vh - nh) // 2
        hud[y_off:y_off+nh, x_off:x_off+nw] = resized

        cx = (vw // 2) + int(self.boxOffsetX * scale)
        base_y = int(vh * 0.88)
        bed_w = int(vw * 0.58)

        # Baseline Conveyor Bed Jig
        cv2.rectangle(hud, (cx - bed_w // 2, base_y), (cx + bed_w // 2, base_y + 24), (0, 242, 254), 2)
        cv2.putText(hud, "DOMS CONVEYOR BED: 12.5 cm CALIBRATION JIG", (cx - 140, base_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 242, 254), 1)

        b_height = int(vh * 0.55 * self.stencilScale)
        b_width = int(b_height * (5.5 / 15.2) * 1.15)
        top_y = base_y - b_height

        box_state = getattr(ev, 'box_color', 'WHITE')
        if box_state == 'GREEN':
            box_col = (34, 197, 94)
            tag_text = "✔ FITS IN BOX (PASS)"
        elif box_state == 'RED':
            box_col = (68, 68, 239)
            tag_text = "✖ DEFECT DETECTED"
        else:
            box_col = (100, 100, 100)
            tag_text = "AWAITING BOTTLE..."

        if box_state == 'WHITE':
            cv2.rectangle(hud, (cx - b_width // 2, top_y), (cx + b_width // 2, base_y), box_col, 1)
            cv2.putText(hud, tag_text, (cx - 70, top_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1)
        else:
            cv2.rectangle(hud, (cx - b_width // 2, top_y), (cx + b_width // 2, base_y), box_col, 2)
            cv2.putText(hud, f"BOTTLE #{self.total_count}: {tag_text}", (cx - b_width // 2, top_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_col, 1)
            cv2.line(hud, (cx, top_y), (cx, base_y), (0, 242, 254), 1)

            # Fluid Line
            men_y = base_y - int(b_height * (11.8 / 15.2))
            cv2.line(hud, (cx - b_width // 2 - 8, men_y), (cx + b_width // 2 + 8, men_y), (254, 242, 0), 2)
            cv2.putText(hud, "FLUID: 11.8cm", (cx + b_width // 2 + 8, men_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (254, 242, 0), 1)

            # Label Bracket
            lbl_b = base_y - int(b_height * (5.9 / 15.2))
            lbl_t = base_y - int(b_height * (9.3 / 15.2))
            cv2.rectangle(hud, (cx - b_width // 2 + 4, lbl_t), (cx + b_width // 2 - 4, lbl_b), (244, 0, 9), 2)
            cv2.putText(hud, "LABEL: 3.4cm", (cx + b_width // 2 + 8, (lbl_t + lbl_b) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (244, 0, 9), 1)

        return Image.fromarray(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB))

    def _generate_simulated_station(self, vw: int, vh: int) -> Image.Image:
        bg = np.zeros((vh, vw, 3), dtype=np.uint8)
        bg[:] = (10, 8, 4)
        cv2.putText(bg, "AWAITING COCA-COLA 250ml BOTTLE...", (vw // 2 - 180, vh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 242, 254), 1)
        return Image.fromarray(bg)


if __name__ == "__main__":
    app = InspectAIApp()
    app.mainloop() 