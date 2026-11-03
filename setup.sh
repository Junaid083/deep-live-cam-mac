#!/bin/zsh
# One-command setup for Deep-Live-Cam on an Apple Silicon Mac.
#
#   ./setup.sh          Python env, packages, core models, `deeplive` command, OBS check
#   ./setup.sh --all    also the slower GFPGAN enhancer and the hair-layer model
#
# Safe to run again at any time: finished steps are skipped and interrupted
# downloads resume.
set -euo pipefail

app=${0:A:h}
cd "$app"

all=false
for arg in "$@"; do
  case $arg in
    --all) all=true ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)" >&2; exit 1 ;;
  esac
done

step() { print -P "\n%B%F{blue}==>%f $1%b"; }
ok() { print -P "    %F{green}✓%f $1"; }
warn() { print -P "    %F{yellow}!%f $1"; }
fail() { print -P "    %F{red}✗ $1%f" >&2; exit 1; }

step "Checking this Mac"
[[ $(uname -s) == Darwin ]] || fail "This setup is for macOS."
[[ $(uname -m) == arm64 ]] || warn "Not Apple Silicon: it will run on the CPU only and be slow."
ok "macOS $(sw_vers -productVersion) on $(uname -m)"
xcode-select -p >/dev/null 2>&1 \
  || fail "Apple's command line tools are needed: run  xcode-select --install  then ./setup.sh again."
ok "Command line tools present"

step "Python 3.14"
command -v brew >/dev/null \
  || fail "Homebrew is required: install it from https://brew.sh, then run ./setup.sh again."
python=$(command -v python3.14 || true)
if [[ -z $python ]]; then
  brew install python@3.14
  python=$(brew --prefix python@3.14)/bin/python3.14
fi
ok "$($python --version)"

step "Python packages (first run takes a few minutes)"
[[ -x venv/bin/python ]] || $python -m venv venv
if command -v uv >/dev/null; then
  VIRTUAL_ENV=$app/venv uv pip install --quiet -r requirements.txt
else
  venv/bin/python -m pip install --quiet --upgrade pip
  venv/bin/python -m pip install --quiet -r requirements.txt
fi
venv/bin/python - <<'PY'
import onnxruntime, insightface, PySide6, pyvirtualcam, cv2  # noqa: F401
providers = onnxruntime.get_available_providers()
assert "CoreMLExecutionProvider" in providers, f"CoreML missing from {providers}"
PY
ok "Packages installed, Apple CoreML acceleration available"

step "AI models (about 650 MB the first time)"
venv/bin/python - "$all" <<'PY'
import sys
from modules.model_downloader import ensure_insightface_pack, ensure_model

models = ["inswapper_128_fp16.onnx", "GPEN-BFR-256.onnx"]
if sys.argv[1] == "true":
    models.append("gfpgan-1024.onnx")
for name in models:
    if ensure_model(name) is None:
        sys.exit(f"Could not download {name}: check the connection and run ./setup.sh again.")
    print(f"    ✓ {name}")
if not ensure_insightface_pack("buffalo_l"):
    sys.exit("Could not download the face detector: check the connection and run ./setup.sh again.")
print("    ✓ face detector (buffalo_l)")
if sys.argv[1] == "true":
    import vcam_head
    if vcam_head._download_parser() is None:
        sys.exit("Could not download the hair model: run ./setup.sh --all again.")
    print("    ✓ hair layer model")
PY
$all || ok "Optional GFPGAN enhancer and hair model skipped (./setup.sh --all, or they download on first use)"

step "The deeplive command"
mkdir -p ~/.local/bin
ln -sf "$app/deeplive" ~/.local/bin/deeplive
if [[ :$PATH: == *:$HOME/.local/bin:* ]]; then
  ok "Run it from anywhere: deeplive"
else
  warn "Add this line to ~/.zshrc, then open a new terminal:"
  warn '  export PATH="$HOME/.local/bin:$PATH"'
fi

step "OBS Virtual Camera (to use it in Meet, Zoom, Teams)"
if [[ ! -d /Applications/OBS.app ]]; then
  warn "OBS is not installed. Install it with:  brew install --cask obs"
elif systemextensionsctl list 2>/dev/null | /usr/bin/grep -q "obs-studio.*activated enabled"; then
  ok "OBS Virtual Camera is ready"
else
  warn "One-time: open OBS → Start Virtual Camera → allow it in System Settings → General →"
  warn "Login Items & Extensions → Camera Extensions. Then stop it and quit OBS."
fi

step "Done"
print "    Start:  deeplive            (or: deeplive ~/path/to/face.jpg)"
print "    Then:   Select a face → Live, and pick 'OBS Virtual Camera' in your meeting app."
