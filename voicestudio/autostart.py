"""Start The Piper Mill automatically when you log in: a systemd user service on Linux, a LaunchAgent on macOS and a scheduled task on
Windows. Nothing is installed system-wide and no administrator rights are needed; turning it off removes exactly what was added."""
import os, shutil, subprocess, sys
from pathlib import Path

NAME = "The Piper Mill"
ROOT = Path(__file__).resolve().parent.parent            # the folder that contains the voicestudio package
UNIT = "the-piper-mill.service"
PLIST_ID = "io.github.thepipermill"


def _python():
    if os.name == "nt":                                  # pythonw has no console window
        w = Path(sys.executable).with_name("pythonw.exe")
        if w.exists(): return str(w)
    return sys.executable


def _run(cmd, **kw): return subprocess.run(cmd, capture_output=True, text=True, timeout=30, **kw)


# ---- Linux
def _unit_path(): return Path.home() / ".config" / "systemd" / "user" / UNIT
def _desktop_path(): return Path.home() / ".config" / "autostart" / "the-piper-mill.desktop"
def _have_systemd():
    return bool(shutil.which("systemctl")) and _run(["systemctl", "--user", "show-environment"]).returncode == 0


def _linux_status():
    if _have_systemd():
        r = _run(["systemctl", "--user", "is-enabled", UNIT])
        return r.stdout.strip() == "enabled"
    return _desktop_path().exists()


def _linux_set(on):
    if _have_systemd():
        if on:
            p = _unit_path(); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"[Unit]\nDescription={NAME} (voice training web page)\n\n[Service]\nWorkingDirectory={ROOT}\n"
                         f"ExecStart={_python()} -m voicestudio serve\nRestart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n")
            _run(["systemctl", "--user", "daemon-reload"])
            r = _run(["systemctl", "--user", "enable", UNIT])
            if r.returncode: raise RuntimeError((r.stderr or r.stdout).strip() or "systemctl could not enable the service")
        else:
            _run(["systemctl", "--user", "disable", UNIT]); _unit_path().unlink(missing_ok=True); _run(["systemctl", "--user", "daemon-reload"])
        return
    d = _desktop_path()                                  # no systemd: the freedesktop autostart folder
    if on:
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(f"[Desktop Entry]\nType=Application\nName={NAME}\nPath={ROOT}\nExec={_python()} -m voicestudio serve\nTerminal=false\n")
    else: d.unlink(missing_ok=True)


# ---- macOS
def _plist_path(): return Path.home() / "Library" / "LaunchAgents" / f"{PLIST_ID}.plist"


def _mac_set(on):
    p = _plist_path(); uid = str(os.getuid()); dom = f"gui/{uid}"
    if on:
        log = Path.home() / "Library" / "Logs" / "ThePiperMill.log"; log.parent.mkdir(parents=True, exist_ok=True); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
                     f'<plist version="1.0"><dict>\n<key>Label</key><string>{PLIST_ID}</string>\n'
                     f'<key>ProgramArguments</key><array><string>{_python()}</string><string>-m</string><string>voicestudio</string><string>serve</string></array>\n'
                     f'<key>WorkingDirectory</key><string>{ROOT}</string>\n<key>RunAtLoad</key><true/>\n<key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>\n'
                     f'<key>StandardOutPath</key><string>{log}</string><key>StandardErrorPath</key><string>{log}</string>\n</dict></plist>\n')
        _run(["launchctl", "bootout", dom, str(p)])      # ignore "not loaded"; it only matters when re-enabling
        # registered for the next login; it is not started now, so it never fights the copy that is already running
    else:
        _run(["launchctl", "bootout", dom, str(p)]); p.unlink(missing_ok=True)


# ---- Windows
def _win_cmd(): return f'"{_python()}" -m voicestudio serve'          # the package is installed in this environment, so no working folder is needed


def _win_status(): return _run(["schtasks", "/Query", "/TN", NAME]).returncode == 0


def _win_set(on):
    if on:
        r = _run(["schtasks", "/Create", "/TN", NAME, "/SC", "ONLOGON", "/TR", _win_cmd(), "/RL", "LIMITED", "/F"])
        if r.returncode: raise RuntimeError((r.stderr or r.stdout).strip() or "Windows could not create the task")
    else: _run(["schtasks", "/Delete", "/TN", NAME, "/F"])


def platform():
    return "windows" if os.name == "nt" else "mac" if sys.platform == "darwin" else "linux"


def status():
    pl = platform()
    try: on = _win_status() if pl == "windows" else _plist_path().exists() if pl == "mac" else _linux_status()
    except Exception: on = False
    how = {"windows": "a scheduled task that runs when you sign in", "mac": "a launch agent that runs when you log in",
           "linux": "a user service that runs when you log in"}[pl]
    return {"platform": pl, "enabled": on, "how": how}


def set_enabled(on):
    pl = platform()
    (_win_set if pl == "windows" else _mac_set if pl == "mac" else _linux_set)(bool(on))
    return status()
