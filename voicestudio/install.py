"""The optional 'Install' step: put a finished voice where your Piper reads voices from.

Configured in config.toml under [install]. Downloading the two files from the web page always works.
"""
import shlex, shutil, subprocess
from .config import CFG


def configured():
    return CFG["install"]["mode"] in ("local", "ssh", "script")


def describe():
    """One plain sentence about what the Install button will do with the current settings (shown in the confirmation)."""
    c = CFG["install"]; m = c["mode"]
    post = " Then it runs: " + c["post_command"] + "." if c.get("post_command") else ""
    if m == "local": return f"Copies the two voice files into the folder {c['local_dir']} on this computer." + post
    if m == "ssh": return f"Copies the two voice files to {c['ssh_host']}:{c['remote_dir']}." + post
    if m == "script": return f"Runs your install script ({shutil.os.path.basename(c['script'])}) with the voice name and the two files. The script decides where they go and whether Piper is restarted."
    return ""


def run(name):
    name = CFG.check_name(name)
    onnx, js = CFG.out / f"{name}.onnx", CFG.out / f"{name}.onnx.json"
    if not (onnx.exists() and js.exists()): raise SystemExit("Export the voice first.")
    c = CFG["install"]; mode = c["mode"]
    if mode == "local":
        d = shutil.os.path.expanduser(c["local_dir"])
        if not d: raise SystemExit("[install] local_dir is not set in config.toml")
        shutil.os.makedirs(d, exist_ok=True)
        for f in (onnx, js): shutil.copy(f, d)
        if c["post_command"]: subprocess.run(c["post_command"], shell=True, check=True)
    elif mode == "ssh":
        if not (c["ssh_host"] and c["remote_dir"]): raise SystemExit("[install] ssh_host and remote_dir must be set in config.toml")
        subprocess.run(["scp", "-q", "-P", str(c["ssh_port"]), str(onnx), str(js), f"{c['ssh_host']}:{c['remote_dir']}/"], check=True)
        if c["post_command"]:
            subprocess.run(["ssh", "-p", str(c["ssh_port"]), c["ssh_host"], c["post_command"]], check=True)
    elif mode == "script":
        if not c["script"]: raise SystemExit("[install] script is not set in config.toml")
        subprocess.run([shutil.os.path.expanduser(c["script"]), name, str(onnx), str(js)], check=True)
    else:
        raise SystemExit("Install is not configured. Download the two files from the page (or from the 'out' folder) "
                         "and put them in your Piper voices folder, or set [install] in config.toml.")
    print(f"Installed '{name}'. Restart/reload your Piper if it does not list new voices by itself.")
