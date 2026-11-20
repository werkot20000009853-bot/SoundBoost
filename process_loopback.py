# -*- coding: utf-8 -*-
"""
Захват звука конкретного процесса через Windows Process Loopback API.

Позволяет слушать усиленную копию звука программы, не попадая в петлю
обратной связи: CaptureFetch захватывает только поток выбранного PID,
а наш собственный вывод в динамики этим захватом не затрагивается.

Требуется Windows 10 build 20348+ / Windows 11.
На основе открытого примера VisualAudioOverlay/process_loopback.py.
"""

import ctypes
import threading
from ctypes import POINTER, byref, c_uint32, c_uint64, c_void_p
from ctypes import wintypes

from comtypes import COMMETHOD, COMObject, GUID, HRESULT, IUnknown

from pycaw.api.audioclient import IAudioClient
from pycaw.api.audioclient import WAVEFORMATEX as _PycawWAVEFORMATEX


class WAVEFORMATEX(ctypes.Structure):
    """Корректная 18-байтовая структура WAVEFORMATEX (в pycaw она неверная)."""
    _pack_ = 1
    _fields_ = [
        ("wFormatTag", wintypes.WORD),
        ("nChannels", wintypes.WORD),
        ("nSamplesPerSec", wintypes.DWORD),
        ("nAvgBytesPerSec", wintypes.DWORD),
        ("nBlockAlign", wintypes.WORD),
        ("wBitsPerSample", wintypes.WORD),
        ("cbSize", wintypes.WORD),
    ]


AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
AUDCLNT_STREAMFLAGS_EVENTCALLBACK = 0x00040000
AUDCLNT_BUFFERFLAGS_SILENT = 0x2
WAVE_FORMAT_IEEE_FLOAT = 0x0003
VT_BLOB = 0x41
AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE = 0
VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK = "VAD\\Process_Loopback"
COINIT_MULTITHREADED = 0x0
RPC_E_CHANGED_MODE = 0x80010106

_kernel32 = ctypes.windll.kernel32
_mmdevapi = ctypes.windll.Mmdevapi
_ole32 = ctypes.windll.ole32


def co_initialize_mta():
    """Инициализировать COM текущего потока как MTA. True — мы инициализировали."""
    hr = _ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
    if hr == RPC_E_CHANGED_MODE:
        return False
    return hr >= 0


def co_uninitialize():
    _ole32.CoUninitialize()


class AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS(ctypes.Structure):
    _fields_ = [
        ("TargetProcessId", wintypes.DWORD),
        ("ProcessLoopbackMode", ctypes.c_int),
    ]


class AUDIOCLIENT_ACTIVATION_PARAMS(ctypes.Structure):
    _fields_ = [
        ("ActivationType", ctypes.c_int),
        ("ProcessLoopbackParams", AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS),
    ]


class _BLOB(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.ULONG), ("pBlobData", c_void_p)]


class PROPVARIANT(ctypes.Structure):
    _fields_ = [
        ("vt", wintypes.WORD),
        ("wReserved1", wintypes.WORD),
        ("wReserved2", wintypes.WORD),
        ("wReserved3", wintypes.WORD),
        ("blob", _BLOB),
    ]


class IActivateAudioInterfaceAsyncOperation(IUnknown):
    _iid_ = GUID("{72A22D78-CDE4-431D-B8CC-843A71199B6D}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetActivateResult",
                  (["out"], POINTER(HRESULT), "activateResult"),
                  (["out"], POINTER(POINTER(IUnknown)), "activatedInterface")),
    ]


class IAgileObject(IUnknown):
    _iid_ = GUID("{94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90}")
    _methods_ = []


class IActivateAudioInterfaceCompletionHandler(IUnknown):
    _iid_ = GUID("{41D949AB-9862-444A-80F6-C261334DA5EB}")
    _methods_ = [
        COMMETHOD([], HRESULT, "ActivateCompleted",
                  (["in"], POINTER(IActivateAudioInterfaceAsyncOperation),
                   "activateOperation")),
    ]


class IAudioCaptureClient(IUnknown):
    _iid_ = GUID("{C8ADBD64-E71E-48A0-A4DE-185C395CD317}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetBuffer",
                  (["out"], POINTER(POINTER(wintypes.BYTE)), "ppData"),
                  (["out"], POINTER(c_uint32), "pNumFramesToRead"),
                  (["out"], POINTER(wintypes.DWORD), "pdwFlags"),
                  (["out"], POINTER(c_uint64), "pu64DevicePosition"),
                  (["out"], POINTER(c_uint64), "pu64QPCPosition")),
        COMMETHOD([], HRESULT, "ReleaseBuffer",
                  (["in"], c_uint32, "NumFramesRead")),
        COMMETHOD([], HRESULT, "GetNextPacketSize",
                  (["out"], POINTER(c_uint32), "pNumFramesInNextPacket")),
    ]


class _CompletionHandler(COMObject):
    _com_interfaces_ = [IActivateAudioInterfaceCompletionHandler, IAgileObject]

    def __init__(self):
        super().__init__()
        self.done = threading.Event()

    def ActivateCompleted(self, this, activateOperation):
        self.done.set()
        return 0


_ActivateAudioInterfaceAsync = _mmdevapi.ActivateAudioInterfaceAsync
_ActivateAudioInterfaceAsync.restype = ctypes.HRESULT
_ActivateAudioInterfaceAsync.argtypes = [
    ctypes.c_wchar_p,
    POINTER(GUID),
    POINTER(PROPVARIANT),
    POINTER(IActivateAudioInterfaceCompletionHandler),
    POINTER(POINTER(IActivateAudioInterfaceAsyncOperation)),
]


def is_supported():
    """Process Loopback доступен с Windows 10 20348 / Windows 11."""
    import sys
    try:
        v = sys.getwindowsversion()
        return v.major > 10 or (v.major == 10 and v.build >= 20348)
    except Exception:
        return False


class ProcessLoopbackCapture:
    """Захват аудиопотока одного процесса (и его дочерних процессов)."""

    def __init__(self, pid, samplerate=48000, channels=2):
        import numpy as np
        self.pid = int(pid)
        self.samplerate = samplerate
        self.channels = channels
        self._np = np
        self._client = None
        self._capture = None
        self._event = None
        self._leftover = None
        self._started = False
        self._com_inited = False

    def start(self):
        self._com_inited = co_initialize_mta()
        try:
            self._client = self._activate_client()
            self._init_stream(self._client)
            self._client.Start()
            self._started = True
        except Exception:
            self.close()
            raise

    def _activate_client(self):
        params = AUDIOCLIENT_ACTIVATION_PARAMS()
        params.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK
        params.ProcessLoopbackParams.TargetProcessId = self.pid
        params.ProcessLoopbackParams.ProcessLoopbackMode = \
            PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE

        pv = PROPVARIANT()
        pv.vt = VT_BLOB
        pv.blob.cbSize = ctypes.sizeof(params)
        pv.blob.pBlobData = ctypes.cast(byref(params), c_void_p)

        handler = _CompletionHandler()
        op = POINTER(IActivateAudioInterfaceAsyncOperation)()
        hr = _ActivateAudioInterfaceAsync(
            VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
            byref(IAudioClient._iid_),
            byref(pv),
            handler,
            byref(op),
        )
        if hr < 0:
            raise RuntimeError(f"ActivateAudioInterfaceAsync HRESULT 0x{hr & 0xFFFFFFFF:08X}")
        if not handler.done.wait(timeout=3.0):
            raise RuntimeError("Активация захвата процесса не завершилась")
        activate_hr, unknown = op.GetActivateResult()
        if activate_hr != 0:
            raise RuntimeError(f"GetActivateResult HRESULT 0x{activate_hr & 0xFFFFFFFF:08X}")
        return unknown.QueryInterface(IAudioClient)

    def _init_stream(self, client):
        wfx = WAVEFORMATEX()
        wfx.wFormatTag = WAVE_FORMAT_IEEE_FLOAT
        wfx.nChannels = self.channels
        wfx.nSamplesPerSec = self.samplerate
        wfx.wBitsPerSample = 32
        wfx.nBlockAlign = self.channels * 4
        wfx.nAvgBytesPerSec = self.samplerate * wfx.nBlockAlign
        wfx.cbSize = 0
        wfx_ptr = ctypes.cast(byref(wfx), POINTER(_PycawWAVEFORMATEX))
        client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
            0, 0, wfx_ptr, None,
        )
        self._event = _kernel32.CreateEventW(None, False, False, None)
        if not self._event:
            raise RuntimeError("CreateEventW failed")
        client.SetEventHandle(self._event)
        ptr = client.GetService(byref(IAudioCaptureClient._iid_))
        self._capture = ptr.QueryInterface(IAudioCaptureClient)

    def read_now(self, numframes):
        """Вернуть ровно numframes кадров, не блокируясь (тишина = нули)."""
        np = self._np
        chunks = []
        if self._leftover is not None:
            chunks.append(self._leftover)
            self._leftover = None
        try:
            packet = self._capture.GetNextPacketSize()
            while packet and packet > 0:
                data_ptr, frames, flags, _, _ = self._capture.GetBuffer()
                if frames:
                    if flags & AUDCLNT_BUFFERFLAGS_SILENT:
                        arr = np.zeros((frames, self.channels), dtype=np.float32)
                    else:
                        fptr = ctypes.cast(data_ptr, POINTER(ctypes.c_float))
                        arr = np.ctypeslib.as_array(
                            fptr, shape=(frames * self.channels,)
                        ).reshape(frames, self.channels).copy()
                    chunks.append(arr)
                self._capture.ReleaseBuffer(frames)
                packet = self._capture.GetNextPacketSize()
        except Exception:
            self._started = False
            raise

        if chunks:
            data = np.concatenate(chunks, axis=0)
        else:
            data = np.zeros((0, self.channels), dtype=np.float32)

        if data.shape[0] >= numframes:
            self._leftover = data[numframes:]
            return data[:numframes]
        pad = np.zeros((numframes - data.shape[0], self.channels), dtype=np.float32)
        return np.concatenate([data, pad], axis=0)

    def close(self):
        self._started = False
        try:
            if self._client is not None:
                self._client.Stop()
        except Exception:
            pass
        self._capture = None
        self._client = None
        if self._event:
            try:
                _kernel32.CloseHandle(self._event)
            except Exception:
                pass
            self._event = None
        if self._com_inited:
            try:
                co_uninitialize()
            except Exception:
                pass
            self._com_inited = False
