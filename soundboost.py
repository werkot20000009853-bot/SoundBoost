# -*- coding: utf-8 -*-
"""
SoundBoost — усилитель системного звука для Windows.

Возможности:
  • DSP-усиление звука (WASAPI loopback -> усилитель -> устройство вывода)
  • Ползунок усиления 100–500 %, softmax-лимитер против клиппинга
  • Управление системной громкостью и громкостью каждого приложения
  • Авто-настройка: делает свободный выход «виртуальным источником» и
    направляет усиленный звук на реальные динамики/наушники
  • Сохранение настроек

Зависимости: customtkinter, pycaw, pyaudiowpatch, numpy
"""

import json
import os
import threading
import tkinter.messagebox as messagebox

import customtkinter as ctk
import numpy as np
import pyaudiowpatch as pyaudio
from pycaw.constants import ERole
from pycaw.pycaw import AudioUtilities

APP_TITLE = "SoundBoost — усилитель системного звука"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(BASE_DIR, "soundboost_settings.json")
FRAMES = 512

DEFAULT_SETTINGS = {
    "gain": 150.0,
    "limiter": True,
    "source": "",
    "sink": "",
    "system_volume": 100.0,
    "system_mute": False,
    "original_default_id": "",
}


# ---------------------------------------------------------------------------
# Работа с аудиоустройствами
# ---------------------------------------------------------------------------

def wasapi_devices():
    """Список устройств WASAPI (pyaudiowpatch)."""
    pa = pyaudio.PyAudio()
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["hostApi"] != wasapi:
                continue
            devices.append({
                "index": i,
                "name": info["name"],
                "inputs": int(info["maxInputChannels"]),
                "outputs": int(info["maxOutputChannels"]),
                "rate": int(info["defaultSampleRate"]),
            })
        return devices
    finally:
        pa.terminate()


def output_devices(devices):
    return [d for d in devices if d["outputs"] > 0 and "[Loopback]" not in d["name"]]


def loopback_devices(devices):
    return [d for d in devices if d["inputs"] > 0 and "[Loopback]" in d["name"]]


def device_id_by_name(name):
    """ID устройства MMDevice по отображаемому имени (для смены default)."""
    for d in AudioUtilities.GetAllDevices():
        if d.FriendlyName.lower() == name.lower():
            return d.id
    return None


def endpoint_volume(device_name=None):
    """Громкость конкретного устройства вывода (или устройства по умолчанию)."""
    if device_name:
        try:
            for device in AudioUtilities.GetAllDevices():
                if device.FriendlyName.lower() == device_name.lower():
                    return device.EndpointVolume
        except Exception:
            pass
    return AudioUtilities.GetSpeakers().EndpointVolume


def set_system_volume(value, device_name=None):
    """value: 0.0–1.0"""
    vol = endpoint_volume(device_name)
    vol.SetMasterVolumeLevelScalar(max(0.0, min(1.0, value)), None)


def get_system_volume(device_name=None):
    return endpoint_volume(device_name).GetMasterVolumeLevelScalar()


def set_system_mute(muted, device_name=None):
    endpoint_volume(device_name).SetMute(muted, None)


def get_system_mute(device_name=None):
    return bool(endpoint_volume(device_name).GetMute())


# ---------------------------------------------------------------------------
# DSP-движок
# ---------------------------------------------------------------------------

def remap_channels(data, in_ch, out_ch):
    if in_ch == out_ch:
        return data
    mono = data.mean(axis=1, keepdims=True)
    if out_ch == 1:
        return mono
    if in_ch == 1:
        return np.repeat(mono, out_ch, axis=1)
    stereo = data[:, :2]
    if out_ch == 2:
        return stereo
    return np.concatenate([stereo, np.repeat(stereo[:, -1:], out_ch - 2, axis=1)], axis=1)


def apply_gain(data, gain, limiter):
    out = data * gain
    if limiter:
        threshold = 0.92
        magnitude = np.abs(out)
        over = magnitude > threshold
        if np.any(over):
            sign = np.sign(out[over])
            compressed = threshold + (1.0 - threshold) * np.tanh(
                (magnitude[over] - threshold) / (1.0 - threshold)
            )
            out[over] = sign * compressed
        np.clip(out, -1.0, 1.0, out=out)
    return out.astype(np.float32)


class BoostEngine(threading.Thread):
    """Поток захвата системного звука, усиления и вывода."""

    def __init__(self, source_index, sink_index, on_status, on_error):
        super().__init__(daemon=True)
        self.source_index = source_index
        self.sink_index = sink_index
        self.on_status = on_status
        self.on_error = on_error
        self.gain = 1.0
        self.limiter = True
        self._stop_event = threading.Event()
        self._pa = None
        self._stream_in = None
        self._stream_out = None
        self.rate = 48000
        self.in_channels = 2
        self.out_channels = 2

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            self._pa = pyaudio.PyAudio()
            src = self._pa.get_device_info_by_index(self.source_index)
            sink = self._pa.get_device_info_by_index(self.sink_index)
            self.rate = int(src["defaultSampleRate"]) or 48000
            self.in_channels = int(src["maxInputChannels"]) or 2
            self.out_channels = int(sink["maxOutputChannels"]) or 2

            self._stream_in = self._pa.open(
                format=pyaudio.paFloat32,
                channels=self.in_channels,
                rate=self.rate,
                input=True,
                input_device_index=self.source_index,
                frames_per_buffer=FRAMES,
            )
            self._stream_out = self._pa.open(
                format=pyaudio.paFloat32,
                channels=self.out_channels,
                rate=self.rate,
                output=True,
                output_device_index=self.sink_index,
                frames_per_buffer=FRAMES,
            )

            self.on_status(
                f"Работает · {self.rate // 1000} кГц · "
                f"{self.in_channels} → {self.out_channels} кан."
            )

            while not self._stop_event.is_set():
                raw = self._stream_in.read(FRAMES, exception_on_overflow=False)
                samples = np.frombuffer(raw, dtype=np.float32)
                usable = samples.size - samples.size % self.in_channels
                if usable <= 0:
                    continue
                frame = samples[:usable].reshape(-1, self.in_channels).copy()
                frame = remap_channels(frame, self.in_channels, self.out_channels)
                frame = apply_gain(frame, self.gain, self.limiter)
                self._stream_out.write(frame.tobytes())

        except Exception as exc:  # noqa: BLE001 — показываем ошибку пользователю
            if not self._stop_event.is_set():
                self.on_error(str(exc))
        finally:
            for stream in (self._stream_in, self._stream_out):
                if stream is not None:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        pass
            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
            self.on_status("Остановлено")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class SoundBoostApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.settings = self.load_settings()
        self.engine = None
        self.devices = wasapi_devices()
        self.source_by_name = {d["name"]: d["index"] for d in self.devices}
        self.sink_by_name = {d["name"]: d["index"] for d in self.devices}
        self.mixer_rows = {}
        self._mixer_job = None

        self.title(APP_TITLE)
        self.geometry("920x860")
        self.minsize(820, 720)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(6, weight=1)

        self.build_header()
        self.build_status()
        self.build_amplifier()
        self.build_devices()
        self.build_system_volume()
        self.build_mixer()
        self.build_footer()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(600, self.refresh_mixer)

    # ---------- загрузка/сохранение настроек ----------

    def load_settings(self):
        settings = dict(DEFAULT_SETTINGS)
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                settings.update(json.load(f))
        except (OSError, ValueError):
            pass
        return settings

    def save_settings(self):
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    # ---------- построение интерфейса ----------

    def build_header(self):
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 4))
        ctk.CTkLabel(
            frame, text="🔊  SoundBoost", font=ctk.CTkFont(size=26, weight="bold")
        ).pack(side="left")
        ctk.CTkLabel(
            frame,
            text="Усиление системного звука Windows",
            font=ctk.CTkFont(size=13),
            text_color="gray70",
        ).pack(side="left", padx=(14, 0), pady=(8, 0))

    def build_status(self):
        frame = ctk.CTkFrame(self, corner_radius=12, height=44)
        frame.grid(row=1, column=0, sticky="ew", padx=18, pady=6)
        frame.grid_propagate(False)
        ctk.CTkLabel(frame, text="Состояние:", font=ctk.CTkFont(size=12, weight="bold")).pack(
            side="left", padx=(16, 8)
        )
        self.status_label = ctk.CTkLabel(
            frame, text="Остановлено", font=ctk.CTkFont(size=13), text_color="#fca5a5"
        )
        self.status_label.pack(side="left")

    def build_amplifier(self):
        card = ctk.CTkFrame(self, corner_radius=16)
        card.grid(row=2, column=0, sticky="ew", padx=18, pady=8)
        card.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            card, text="УСИЛЕНИЕ ЗВУКА", font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#7dd3fc",
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=20, pady=(16, 0))

        self.gain_value = ctk.CTkLabel(
            card, text="", font=ctk.CTkFont(size=38, weight="bold")
        )
        self.gain_value.grid(row=1, column=0, sticky="w", padx=20, pady=(2, 0))
        self.gain_db = ctk.CTkLabel(
            card, text="", font=ctk.CTkFont(size=13), text_color="gray70"
        )
        self.gain_db.grid(row=2, column=0, sticky="w", padx=20)

        gain = float(self.settings["gain"])
        self.gain_slider = ctk.CTkSlider(
            card, from_=100, to=500, number_of_steps=80, command=self.on_gain_change
        )
        self.gain_slider.set(gain)
        self.gain_slider.grid(row=1, column=1, rowspan=2, sticky="ew", padx=(24, 20))
        self.update_gain_labels(gain)

        presets = ctk.CTkFrame(card, fg_color="transparent")
        presets.grid(row=3, column=0, columnspan=2, sticky="w", padx=20, pady=(12, 4))
        for value in (100, 150, 200, 300, 500):
            ctk.CTkButton(
                presets,
                text=f"{value}%",
                width=54,
                height=28,
                font=ctk.CTkFont(size=11),
                fg_color="gray25",
                hover_color="gray35",
                command=lambda v=value: self.gain_slider.set(v),
            ).pack(side="left", padx=(0, 6))

        self.limiter_switch = ctk.CTkSwitch(
            card,
            text="Лимитер (защита от искажений)",
            font=ctk.CTkFont(size=12),
            command=self.on_limiter_change,
        )
        if self.settings["limiter"]:
            self.limiter_switch.select()
        else:
            self.limiter_switch.deselect()
        self.limiter_switch.grid(row=3, column=2, sticky="e", padx=(0, 20), pady=(12, 4))

        self.toggle_button = ctk.CTkButton(
            card,
            text="▶  ВКЛЮЧИТЬ УСИЛЕНИЕ",
            height=46,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#16a34a",
            hover_color="#15803d",
            command=self.toggle_engine,
        )
        self.toggle_button.grid(
            row=4, column=0, columnspan=3, sticky="ew", padx=20, pady=(12, 18)
        )

    def build_devices(self):
        card = ctk.CTkFrame(self, corner_radius=16)
        card.grid(row=3, column=0, sticky="ew", padx=18, pady=8)
        card.columnconfigure(1, weight=1)
        card.columnconfigure(3, weight=1)

        ctk.CTkLabel(
            card, text="МАРШРУТИЗАЦИЯ ЗВУКА", font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#7dd3fc",
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=20, pady=(14, 6))

        ctk.CTkLabel(card, text="Источник (loopback):", font=ctk.CTkFont(size=12)).grid(
            row=1, column=0, sticky="w", padx=(20, 6), pady=6
        )
        self.source_names = [d["name"] for d in loopback_devices(self.devices)]
        self.source_combo = ctk.CTkComboBox(
            card, values=self.source_names or ["— нет —"], width=260
        )
        self.source_combo.set(self.settings["source"] or (self.source_names[0] if self.source_names else "— нет —"))
        self.source_combo.grid(row=1, column=1, sticky="ew", padx=(0, 16), pady=6)

        ctk.CTkLabel(card, text="Вывод (динамики):", font=ctk.CTkFont(size=12)).grid(
            row=1, column=2, sticky="w", padx=(6, 6), pady=6
        )
        self.sink_names = [d["name"] for d in output_devices(self.devices)]
        self.sink_combo = ctk.CTkComboBox(
            card, values=self.sink_names or ["— нет —"], width=260
        )
        self.sink_combo.set(self.settings["sink"] or (self.sink_names[0] if self.sink_names else "— нет —"))
        self.sink_combo.grid(row=1, column=3, sticky="ew", padx=(0, 20), pady=6)

        buttons = ctk.CTkFrame(card, fg_color="transparent")
        buttons.grid(row=2, column=0, columnspan=4, sticky="ew", padx=20, pady=(8, 6))
        ctk.CTkButton(
            buttons, text="Авто-настройка", width=150, height=32,
            command=self.auto_setup,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            buttons, text="Обновить список", width=150, height=32,
            fg_color="gray25", hover_color="gray35", command=self.refresh_devices,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            buttons, text="Восстановить вывод", width=170, height=32,
            fg_color="gray25", hover_color="gray35", command=self.restore_default_device,
        ).pack(side="left")

        ctk.CTkLabel(
            card,
            text="Источник — это [Loopback] неиспользуемого выхода; данные идут на реальные динамики. "
                 "Если оборудования без перезаписи нет — установите виртуальный кабель (VB-Cable).",
            font=ctk.CTkFont(size=11),
            text_color="gray65",
            wraplength=860,
            justify="left",
        ).grid(row=3, column=0, columnspan=4, sticky="w", padx=20, pady=(2, 14))

    def build_system_volume(self):
        card = ctk.CTkFrame(self, corner_radius=16)
        card.grid(row=4, column=0, sticky="ew", padx=18, pady=8)
        card.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            card, text="СИСТЕМНАЯ ГРОМКОСТЬ", font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#7dd3fc",
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=20, pady=(14, 6))

        self.vol_value = ctk.CTkLabel(card, text="", font=ctk.CTkFont(size=18, weight="bold"), width=52)
        self.vol_value.grid(row=1, column=0, padx=(20, 8))

        volume = float(self.settings["system_volume"])
        self.volume_slider = ctk.CTkSlider(
            card, from_=0, to=100, number_of_steps=100, command=self.on_volume_change
        )
        self.volume_slider.set(volume)
        self.volume_slider.grid(row=1, column=1, sticky="ew", padx=8)
        self.update_volume_label(volume)

        self.mute_switch = ctk.CTkSwitch(
            card, text="Без звука", font=ctk.CTkFont(size=12), command=self.on_mute_change
        )
        if self.settings["system_mute"]:
            self.mute_switch.select()
        self.mute_switch.grid(row=1, column=2, padx=(8, 16))

        ctk.CTkButton(
            card, text="🔈  Максимум", width=130, height=32,
            command=self.set_max_volume,
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=20, pady=(10, 16))

        try:
            current = get_system_volume(self.volume_device_name())
            self.volume_slider.set(current * 100)
            self.update_volume_label(current * 100)
        except Exception:
            pass

    def build_mixer(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=5, column=0, sticky="ew", padx=18, pady=(10, 4))
        ctk.CTkLabel(
            header, text="ГРОМКОСТЬ ПРИЛОЖЕНИЙ",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#7dd3fc",
        ).pack(side="left")
        self.mixer_count = ctk.CTkLabel(
            header, text="", font=ctk.CTkFont(size=11), text_color="gray70"
        )
        self.mixer_count.pack(side="left", padx=10)
        ctk.CTkButton(
            header, text="Обновить", width=90, height=26, font=ctk.CTkFont(size=11),
            fg_color="gray25", hover_color="gray35", command=self.refresh_mixer,
        ).pack(side="right")

        self.mixer_scroll = ctk.CTkScrollableFrame(self, corner_radius=14)
        self.mixer_scroll.grid(row=6, column=0, sticky="nsew", padx=18, pady=(0, 8))

    def build_footer(self):
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=7, column=0, sticky="ew", padx=18, pady=(0, 14))
        ctk.CTkLabel(
            footer,
            text="Подсказка: включите усиление, затем запустите музыку/видео. "
                 "При искажениях уменьшите % или включите лимитер.",
            font=ctk.CTkFont(size=11), text_color="gray65",
        ).pack(side="left")
        ctk.CTkButton(
            footer, text="Инструкция", width=100, height=26,
            font=ctk.CTkFont(size=11), fg_color="gray25", hover_color="gray35",
            command=self.show_help,
        ).pack(side="right")

    # ---------- обработчики ----------

    def volume_device_name(self):
        """Управляем выбранным выходом, а не «тихим» источником захвата."""
        name = self.sink_combo.get()
        return name if name in self.sink_by_name else None

    def update_gain_labels(self, value):
        gain = float(value) / 100.0
        self.gain_value.configure(text=f"{int(value)}%")
        db = 20 * np.log10(gain)
        self.gain_db.configure(text=f"≈ +{db:.1f} dB" if db > 0 else "0 dB")

    def on_gain_change(self, value):
        self.settings["gain"] = float(value)
        self.update_gain_labels(value)
        if self.engine is not None:
            self.engine.gain = float(value) / 100.0

    def on_limiter_change(self):
        self.settings["limiter"] = bool(self.limiter_switch.get())
        if self.engine is not None:
            self.engine.limiter = self.settings["limiter"]

    def update_volume_label(self, value):
        self.vol_value.configure(text=f"{int(value)}%")

    def on_volume_change(self, value):
        self.settings["system_volume"] = float(value)
        self.update_volume_label(value)
        try:
            set_system_volume(float(value) / 100.0, self.volume_device_name())
        except Exception as exc:
            self.status_label.configure(text=f"Ошибка громкости: {exc}", text_color="#fca5a5")

    def on_mute_change(self):
        muted = bool(self.mute_switch.get())
        self.settings["system_mute"] = muted
        try:
            set_system_mute(muted, self.volume_device_name())
        except Exception as exc:
            self.status_label.configure(text=f"Ошибка: {exc}", text_color="#fca5a5")

    def set_max_volume(self):
        try:
            set_system_volume(1.0, self.volume_device_name())
            set_system_mute(False, self.volume_device_name())
            self.mute_switch.deselect()
            self.volume_slider.set(100)
            self.settings["system_volume"] = 100.0
            self.settings["system_mute"] = False
            self.update_volume_label(100)
            for row in self.mixer_rows.values():
                try:
                    row["session"].SimpleAudioVolume.SetMasterVolume(1.0, None)
                    row["session"].SimpleAudioVolume.SetMute(False, None)
                    row["slider"].set(1.0)
                    row["label"].configure(text="100%")
                except Exception:
                    pass
        except Exception as exc:
            self.status_label.configure(text=f"Ошибка: {exc}", text_color="#fca5a5")

    # ---------- микшер приложений ----------

    def create_app_row(self, key, session):
        row = {}
        frame = ctk.CTkFrame(self.mixer_scroll, fg_color=("gray88", "gray18"), corner_radius=8)
        frame.grid_columnconfigure(1, weight=1)

        name = session.Process.name() if session.Process else str(key[0])
        ctk.CTkLabel(
            frame, text=name, font=ctk.CTkFont(size=12, weight="bold"), width=170,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(14, 8), pady=8)

        slider = ctk.CTkSlider(frame, from_=0, to=1, number_of_steps=100,
                               command=lambda v, k=key: self.on_app_volume(k, v))
        slider.grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        slider.bind("<Button-1>", lambda e, k=key: self.on_drag_start(k))
        slider.bind("<ButtonRelease-1>", lambda e, k=key: self.on_drag_end(k))

        label = ctk.CTkLabel(frame, text="", width=46, font=ctk.CTkFont(size=11))
        label.grid(row=0, column=2, padx=(4, 8))

        mute = ctk.CTkButton(
            frame, text="Вкл", width=48, height=24, font=ctk.CTkFont(size=10),
            fg_color="gray30", hover_color="gray40",
            command=lambda k=key: self.on_app_mute(k),
        )
        mute.grid(row=0, column=3, padx=(0, 12))

        row.update({"frame": frame, "slider": slider, "label": label,
                    "mute": mute, "session": session, "dragging": False})
        self.mixer_rows[key] = row
        slider.set(1.0)
        label.configure(text="100%")

    def on_drag_start(self, key):
        if key in self.mixer_rows:
            self.mixer_rows[key]["dragging"] = True

    def on_drag_end(self, key):
        if key in self.mixer_rows:
            self.mixer_rows[key]["dragging"] = False

    def on_app_volume(self, key, value):
        row = self.mixer_rows.get(key)
        if not row:
            return
        try:
            row["session"].SimpleAudioVolume.SetMasterVolume(float(value), None)
            row["label"].configure(text=f"{int(value * 100)}%")
            self.settings.setdefault("app_volumes", {})[str(key[0])] = float(value)
        except Exception:
            pass

    def on_app_mute(self, key):
        row = self.mixer_rows.get(key)
        if not row:
            return
        try:
            vol = row["session"].SimpleAudioVolume
            muted = not vol.GetMute()
            vol.SetMute(muted, None)
            row["mute"].configure(text="Выкл" if muted else "Вкл",
                                  fg_color="#991b1b" if muted else "gray30")
        except Exception:
            pass

    def refresh_mixer(self):
        try:
            sessions = AudioUtilities.GetAllSessions()
        except Exception:
            sessions = []
        current = {}
        for session in sessions:
            if session.Process is None:
                continue
            key = (session.ProcessId, session.Process.name())
            current[key] = session

        for key in list(self.mixer_rows):
            if key not in current:
                self.mixer_rows.pop(key)["frame"].destroy()

        for key, session in current.items():
            if key not in self.mixer_rows:
                self.create_app_row(key, session)
            row = self.mixer_rows[key]
            row["session"] = session
            if not row["dragging"]:
                try:
                    vol = session.SimpleAudioVolume
                    value = vol.GetMasterVolume()
                    row["slider"].set(value)
                    row["label"].configure(text=f"{int(value * 100)}%")
                    muted = bool(vol.GetMute())
                    row["mute"].configure(text="Выкл" if muted else "Вкл",
                                          fg_color="#991b1b" if muted else "gray30")
                except Exception:
                    pass

        for index, row in enumerate(sorted(self.mixer_rows.values(),
                                           key=lambda r: r["frame"].winfo_children()[0].cget("text"))):
            row["frame"].grid(row=index, column=0, sticky="ew", pady=2)

        self.mixer_count.configure(text=f"{len(current)} прил.")
        self._mixer_job = self.after(2500, self.refresh_mixer)

    # ---------- движок ----------

    def engine_status(self, text):
        active = text.startswith("Работает")
        self.status_label.configure(
            text=text, text_color="#86efac" if active else ("#fca5a5" if "Ошибка" in text else "gray70")
        )

    def engine_error(self, message):
        self.engine_status(f"Ошибка: {message}")
        self.toggle_button.configure(
            text="▶  ВКЛЮЧИТЬ УСИЛЕНИЕ", fg_color="#16a34a", hover_color="#15803d"
        )
        self.engine = None

    def toggle_engine(self):
        if self.engine is not None:
            self.engine.stop()
            self.engine = None
            self.toggle_button.configure(
                text="▶  ВКЛЮЧИТЬ УСИЛЕНИЕ", fg_color="#16a34a", hover_color="#15803d"
            )
            self.engine_status("Остановлено")
            return

        source = self.source_combo.get()
        sink = self.sink_combo.get()
        if source not in self.source_by_name or sink not in self.sink_by_name:
            messagebox.showwarning(
                "SoundBoost", "Выберите корректные устройства источника и вывода."
            )
            return
        if source.replace(" [Loopback]", "").lower() == sink.lower():
            if not messagebox.askyesno(
                "Возможна обратная связь",
                "Источник и вывод относятся к одному устройству — появится эхо/гул.\n"
                "Продолжить всё равно?",
            ):
                return

        engine = BoostEngine(
            self.source_by_name[source], self.sink_by_name[sink],
            self.engine_status, self.engine_error,
        )
        engine.gain = float(self.settings["gain"]) / 100.0
        engine.limiter = bool(self.settings["limiter"])
        self.engine = engine
        self.engine_status("Запуск…")
        engine.start()
        self.toggle_button.configure(
            text="■  ОСТАНОВИТЬ УСИЛЕНИЕ", fg_color="#b91c1c", hover_color="#991b1b"
        )
        self.settings["source"] = source
        self.settings["sink"] = sink

    def refresh_devices(self):
        self.devices = wasapi_devices()
        self.source_by_name = {d["name"]: d["index"] for d in self.devices}
        self.sink_by_name = {d["name"]: d["index"] for d in self.devices}
        self.source_names = [d["name"] for d in loopback_devices(self.devices)]
        self.sink_names = [d["name"] for d in output_devices(self.devices)]
        self.source_combo.configure(values=self.source_names or ["— нет —"])
        self.sink_combo.configure(values=self.sink_names or ["— нет —"])
        messagebox.showinfo("SoundBoost", "Список устройств обновлён.")

    def auto_setup(self):
        """Настроить «виртуальный источник» из второго физического выхода."""
        if self.engine is not None:
            messagebox.showwarning("SoundBoost", "Сначала остановите усиление.")
            return

        current_default = AudioUtilities.GetSpeakers().FriendlyName
        loops = loopback_devices(self.devices)
        outputs = {d["name"].lower() for d in output_devices(self.devices)}
        candidates = []
        for lp in loops:
            base = lp["name"].replace(" [Loopback]", "")
            if base.lower() == current_default.lower():
                continue
            if base.lower() in outputs:
                candidates.append((lp, base))
        if not candidates:
            messagebox.showwarning(
                " SoundBoost",
                "Не нашёл второй выход для захвата.\n\n"
                "Подключите второй динамик/наушники/HDMI либо установите "
                "виртуальный аудиокабель (VB-Cable) и повторите.",
            )
            return

        loop, base = candidates[0]
        text = (
            "Для усиления звука без виртуального кабеля приложения должны "
            "проигрывать звук на «тихий» выход, а усиленный звук пойдёт на ваши динамики.\n\n"
            f"Источник: {base}\n"
            f"Финальный вывод: {current_default}\n\n"
            "Сменить устройство вывода по умолчанию на источник и запустить усиление?"
        )
        if not messagebox.askyesno("Авто-настройка SoundBoost", text):
            return

        dev_id = device_id_by_name(base)
        if not dev_id:
            messagebox.showerror("SoundBoost", "Не удалось найти ID устройства.")
            return
        try:
            AudioUtilities.SetDefaultDevice(
                dev_id,
                roles=[ERole.eConsole, ERole.eMultimedia, ERole.eCommunications],
            )
            self.settings["original_default_id"] = device_id_by_name(current_default) or ""
        except Exception as exc:
            messagebox.showerror("SoundBoost", f"Не удалось сменить устройство: {exc}")
            return

        self.source_combo.set(loop["name"])
        self.sink_combo.set(current_default)
        self.devices = wasapi_devices()
        self.source_by_name = {d["name"]: d["index"] for d in self.devices}
        self.sink_by_name = {d["name"]: d["index"] for d in self.devices}
        self.settings["source"] = loop["name"]
        self.settings["sink"] = current_default
        self.toggle_engine()
        messagebox.showinfo(
            "SoundBoost",
            "Готово! Звук приложений идёт на «тихий» выход, а вы слышите "
            "усиленную версию на своих динамиках.\n\n"
            "Вернуть прежний вывод можно кнопкой «Восстановить вывод».",
        )

    def restore_default_device(self):
        original = self.settings.get("original_default_id")
        if not original:
            messagebox.showinfo("SoundBoost", "Исходное устройство не сохранено.")
            return
        if self.engine is not None:
            self.toggle_engine()
        try:
            AudioUtilities.SetDefaultDevice(
                original,
                roles=[ERole.eConsole, ERole.eMultimedia, ERole.eCommunications],
            )
            self.settings["original_default_id"] = ""
            messagebox.showinfo("SoundBoost", "Устройство вывода по умолчанию восстановлено.")
        except Exception as exc:
            messagebox.showerror("SoundBoost", f"Ошибка восстановления: {exc}")

    def show_help(self):
        window = ctk.CTkToplevel(self)
        window.title("Инструкция — SoundBoost")
        window.geometry("680x520")
        text = ctk.CTkTextbox(window, font=ctk.CTkFont(size=13), wrap="word")
        text.pack(fill="both", expand=True, padx=16, pady=16)
        text.insert("1.0", (
            "КАК РАБОТАЕТ\n\n"
            "Windows ограничивает громкость 100 %. SoundBoost перехватывает поток\n"
            "с выхода (WASAPI Loopback), умножает его на выбранный коэффициент\n"
            "и выводит на другие динамики. Это даёт настоящее усиление до 500 %.\n\n"
            "БЫСТРЫЙ СТАРТ\n\n"
            "1. Нажмите «Авто-настройка» — приложение само выберет свободный\n"
            "   выход как источник и вернёт усиленный звук на ваши динамики.\n"
            "2. Нажмите «ВКЛЮЧИТЬ УСИЛЕНИЕ» и запустите музыку/видео.\n"
            "3. Ползунком «Усиление» подберите комфортный уровень (обычно 150–250 %).\n\n"
            "ВАРИАНТ С ВИРТУАЛЬНЫМ КАБЕЛЕМ\n\n"
            "Если второй физический выход недоступен, установите VB-Cable:\n"
            "  • в Windows поставьте «Cable Input» устройством вывода по умолчанию;\n"
            "  • в SoundBoost выберите источник «Cable Output [Loopback]»;\n"
            "  • в качестве вывода — ваши реальные динамики или наушники.\n\n"
            "ВАЖНО\n\n"
            "• Не выбирайте одно и то же устройство для захвата и вывода — будет эхо.\n"
            "• Лимитер включён по умолчанию и защищает от клиппинга/искажений.\n"
            "• Обработка добавляет задержку ~20–40 мс: для игр и звонков\n"
            "  лучше использовать системную громкость без DSP.\n"
            "• Кнопка «Восстановить вывод» возвращает прежнее устройство по умолчанию.\n"
        ))
        text.configure(state="disabled")

    # ---------- закрытие ----------

    def on_close(self):
        if self._mixer_job is not None:
            try:
                self.after_cancel(self._mixer_job)
            except Exception:
                pass
        if self.engine is not None:
            self.engine.stop()
        try:
            current = get_system_volume(self.volume_device_name())
            self.settings["system_volume"] = current * 100
            self.settings["system_mute"] = get_system_mute(self.volume_device_name())
        except Exception:
            pass
        self.save_settings()
        self.destroy()


def main():
    app = SoundBoostApp()
    # Тестовый режим: SOUNDBOOST_AUTOCLOSE=1 — окно само закроется через 1.5 с.
    if os.environ.get("SOUNDBOOST_AUTOCLOSE"):
        try:
            with open(os.path.join(BASE_DIR, "selftest.ok"), "w", encoding="utf-8") as f:
                f.write("started")
        except OSError:
            pass
        app.after(1500, app.destroy)
    app.mainloop()


if __name__ == "__main__":
    main()
