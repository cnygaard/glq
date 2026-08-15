#!/usr/bin/env bash
#
# GLQ installer — https://github.com/cnygaard/glq
#
#   curl -fsSL https://raw.githubusercontent.com/cnygaard/glq/main/install.sh | bash
#
# Creates a venv at ~/.glq/venv, installs glq into it, and hands over to
# `python -m glq.installer`, which discovers published checkpoints, sizes them against
# your GPU and wires up serving, a chat UI and the pi coding agent.
#
# Everything is wrapped in main() and called on the LAST line. That is deliberate: piping
# curl into bash executes whatever has arrived so far, so a connection dropped mid-transfer
# would otherwise run half an installer. With this layout a truncated file defines some
# functions and does nothing.
#
# It does not use sudo. If a system package is missing it tells you the apt line and stops,
# rather than escalating on your behalf.

set -euo pipefail
IFS=$'\n\t'
umask 077

GLQ_HOME="${GLQ_HOME:-$HOME/.glq}"
GLQ_VENV="$GLQ_HOME/venv"
GLQ_VERSION="${GLQ_VERSION:-}"          # empty = latest on PyPI
DRY_RUN=0
ALLOW_ROOT=0
PREFLIGHT_ONLY=0
ASSUME_NO_GPU=0
PASSTHRU=()

# Overridable like GLQ_HOME so the distro mapping can be exercised without seven machines.
GLQ_OS_RELEASE="${GLQ_OS_RELEASE:-/etc/os-release}"

# Disk needed for torch + vLLM + one small checkpoint, measured on a clean box.
MIN_FREE_GB=25

# curl with the transport pinned: https only, TLS >= 1.2, fail on HTTP errors.
CURL=(curl --proto '=https' --tlsv1.2 -fsSL)

say()  { printf '%s\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  [dry-run] %s\n' "$*"
    else
        printf '  $ %s\n' "$*"
        "$@"
    fi
}

usage() {
    cat <<'USAGE'
GLQ installer

  --components LIST   core,vllm,picode,chat   (default: core,vllm,chat)
  --model REPO_ID     checkpoint to serve     (default: chosen interactively)
  --chat WHICH        gradio | openwebui | none
  --glq-version VER   pin glq (default: latest release)
  --yes               accept defaults, never prompt
  --list              list available checkpoints and exit
  --preflight         check prerequisites and exit (changes nothing)
  --dry-run           print every command, change nothing
  --allow-root        permit running as root (not recommended)
  -h, --help          this message

Anything not listed here is passed through to `python -m glq.installer`.
USAGE
}

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --preflight)    PREFLIGHT_ONLY=1; shift ;;
            --assume-no-gpu) ASSUME_NO_GPU=1; shift ;;
            --dry-run)      DRY_RUN=1; PASSTHRU+=("$1"); shift ;;
            --allow-root)   ALLOW_ROOT=1; shift ;;
            --glq-version)  [ $# -ge 2 ] || die "--glq-version needs a value"
                            GLQ_VERSION="$2"; shift 2 ;;
            -h|--help)      usage; exit 0 ;;
            *)              PASSTHRU+=("$1"); shift ;;
        esac
    done
}

# Distro id + the package manager that fixes a missing prerequisite there. Getting this
# wrong is worse than silence: telling a Fedora user to run apt-get reads as "this tool was
# not written for you". Derivatives are resolved through ID_LIKE (rocky/alma -> rhel,
# manjaro/steamos -> arch), with a generic fallback so an unknown distro still installs.
detect_distro() {
    DISTRO_ID=""; DISTRO_LIKE=""; DISTRO_NAME="unknown"
    if [ -r "$GLQ_OS_RELEASE" ]; then
        DISTRO_ID=$(   sed -n 's/^ID=//p'         "$GLQ_OS_RELEASE" | tr -d '"' | head -1)
        DISTRO_LIKE=$( sed -n 's/^ID_LIKE=//p'    "$GLQ_OS_RELEASE" | tr -d '"' | head -1)
        DISTRO_NAME=$( sed -n 's/^PRETTY_NAME=//p' "$GLQ_OS_RELEASE" | tr -d '"' | head -1)
        [ -n "$DISTRO_NAME" ] || DISTRO_NAME="$DISTRO_ID"
    fi
}

pkg_hint() {
    local id="$DISTRO_ID $DISTRO_LIKE"
    case " $id " in
        *ubuntu*|*debian*)          echo "sudo apt-get update && sudo apt-get install -y python3-venv python3-dev build-essential curl" ;;
        *fedora*|*rhel*|*centos*|*rocky*|*almalinux*|*amzn*)
                                    echo "sudo dnf install -y python3-devel gcc curl" ;;
        *steamos*)                  echo "sudo steamos-readonly disable && sudo pacman -S --needed python gcc curl   # or use distrobox" ;;
        *arch*|*manjaro*|*endeavouros*)
                                    echo "sudo pacman -S --needed python gcc curl" ;;
        *azurelinux*|*mariner*)     echo "sudo tdnf install -y python3-devel gcc curl" ;;
        *suse*|*sles*)              echo "sudo zypper install -y python3-devel gcc curl" ;;
        *)                          echo "install with your package manager: python3 (>=3.10) + venv, gcc, curl" ;;
    esac
}

preflight() {
    local blockers=0
    detect_distro
    say ""
    say "Pre-flight checks"
    say "  distro:  $DISTRO_NAME${DISTRO_ID:+  (id=$DISTRO_ID${DISTRO_LIKE:+, like=$DISTRO_LIKE})}"
    # Printed even when everything passes: it tells the user this installer recognised
    # their distro, and gives them the command up-front if a later step needs it.
    say "  packages on this distro: $(pkg_hint)"
    case " $DISTRO_ID " in
        *steamos*) say "           note: SteamOS ships a read-only root filesystem — a plain"
                   say "           pacman install fails until you run steamos-readonly disable" ;;
    esac

    # python
    local py="" pyver=""
    for c in python3.12 python3.11 python3.10 python3; do
        if command -v "$c" >/dev/null 2>&1 && \
           "$c" -c 'import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null; then
            py="$c"; pyver="$("$c" --version 2>&1)"; break
        fi
    done
    if [ -n "$py" ]; then
        say "  python:  $pyver  ($py)"
        if "$py" -c 'import venv' 2>/dev/null; then
            say "  venv:    available"
        else
            warn "  venv:    MISSING — the installer cannot create a virtualenv"; blockers=$((blockers+1))
        fi
    else
        warn "  python:  MISSING or older than 3.10 (glq needs >= 3.10)"; blockers=$((blockers+1))
    fi

    command -v curl >/dev/null 2>&1 && say "  curl:    present" || {
        warn "  curl:    MISSING"; blockers=$((blockers+1)); }

    # A C compiler is needed for the CUDA extension's first-run JIT build.
    if command -v gcc >/dev/null 2>&1 || command -v cc >/dev/null 2>&1; then
        say "  cc:      present (needed to JIT-build the CUDA extension)"
    else
        warn "  cc:      MISSING — the fused CUDA kernel cannot be built; GLQ falls back to CPU"
    fi

    # GPU. Absent is a warning, never a blocker: CPU dequantize-then-matmul is supported.
    if [ "$ASSUME_NO_GPU" -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1 \
       && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
        say "  gpu:     $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
        say "  cuda:    $(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)"
    else
        warn "  gpu:     no NVIDIA GPU detected — GLQ will run on CPU (dequantize-then-matmul, slow)"
        case " $DISTRO_ID " in
            *steamos*) warn "           SteamOS/Steam Deck is AMD; the CUDA fast path is unavailable there" ;;
        esac
    fi

    local freeg
    freeg=$(df -Pk "$HOME" 2>/dev/null | awk 'NR==2{printf "%d", $4/1024/1024}')
    if [ -n "$freeg" ]; then
        if [ "$freeg" -lt "$MIN_FREE_GB" ]; then
            warn "  disk:    ${freeg} GB free in \$HOME — torch + vLLM + a checkpoint want >= ${MIN_FREE_GB} GB"
        else
            say "  disk:    ${freeg} GB free in \$HOME"
        fi
    fi

    if [ "$blockers" -gt 0 ]; then
        say ""
        warn "$blockers prerequisite(s) missing. Install them with:"
        warn "    $(pkg_hint)"
        return 1
    fi
    say ""
    return 0
}

check_not_root() {
    # Running an installer as root puts a venv full of root-owned files in /root and is a
    # habit worth not teaching. Nothing here needs privileges.
    if [ "${EUID:-$(id -u)}" -eq 0 ] && [ "$ALLOW_ROOT" -eq 0 ]; then
        die "refusing to run as root. Re-run as a normal user, or pass --allow-root."
    fi
}

find_python() {
    local py
    for py in python3.12 python3.11 python3.10 python3; do
        if command -v "$py" >/dev/null 2>&1; then
            if "$py" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)'; then
                printf '%s' "$py"
                return 0
            fi
        fi
    done
    die "no Python >= 3.10 found. Install one, e.g.: sudo apt-get install -y python3"
}

check_venv_module() {
    local py="$1"
    "$py" -c 'import venv' 2>/dev/null && return 0
    die "Python is missing the venv module. Install it with:
    sudo apt-get update && sudo apt-get install -y python3-venv"
}

create_venv() {
    local py="$1"
    if [ -x "$GLQ_VENV/bin/python" ]; then
        say "== reusing existing venv at $GLQ_VENV"
        return 0
    fi
    say "== creating venv at $GLQ_VENV"
    run mkdir -p "$GLQ_HOME"
    run "$py" -m venv "$GLQ_VENV"
}

install_glq() {
    local spec="glq"
    [ -n "$GLQ_VERSION" ] && spec="glq==$GLQ_VERSION"
    say "== installing $spec (this pulls PyTorch and takes a few minutes)"
    run "$GLQ_VENV/bin/pip" install --upgrade pip
    run "$GLQ_VENV/bin/pip" install --upgrade "$spec"
}

hand_over() {
    say "== configuring"
    if [ "$DRY_RUN" -eq 1 ] && [ ! -x "$GLQ_VENV/bin/python" ]; then
        say "  [dry-run] would run: $GLQ_VENV/bin/python -m glq.installer ${PASSTHRU[*]:-}"
        return 0
    fi
    "$GLQ_VENV/bin/python" -m glq.installer ${PASSTHRU[@]+"${PASSTHRU[@]}"}
}

main() {
    parse_args "$@"
    check_not_root

    if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
        preflight || die "pre-flight failed — install the packages above and re-run"
        say "Pre-flight OK. Run without --preflight to install."
        exit 0
    fi
    preflight || die "pre-flight failed — install the packages above and re-run"

    say "GLQ installer"
    say "  target: $GLQ_VENV"
    [ "$DRY_RUN" -eq 1 ] && warn "  dry-run: nothing will be installed"

    local py
    py="$(find_python)"
    say "  python: $py ($("$py" --version 2>&1))"
    check_venv_module "$py"

    create_venv "$py"
    install_glq
    hand_over
}

main "$@"
