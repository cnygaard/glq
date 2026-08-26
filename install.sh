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
# rather than escalating on your behalf. It also fetches nothing itself — pip does all the
# downloading — so the only curl in the installer is the nvm bootstrap in glq/installer,
# which pins the transport (see test_every_curl_pins_https_and_tls).

set -euo pipefail
IFS=$'\n\t'
umask 077

GLQ_HOME="${GLQ_HOME:-$HOME/.glq}"
GLQ_VENV="$GLQ_HOME/venv"
GLQ_VERSION="${GLQ_VERSION:-}"          # empty = latest on PyPI
GLQ_SOURCE="${GLQ_SOURCE:-}"            # empty = PyPI; else any pip spec (wheel/path/VCS)
DRY_RUN=0
ALLOW_ROOT=0
PREFLIGHT_ONLY=0
ASSUME_NO_GPU=0
MODIFY_PATH=1
PASSTHRU=()

# Overridable like GLQ_HOME so the distro mapping can be exercised without seven machines.
GLQ_OS_RELEASE="${GLQ_OS_RELEASE:-/etc/os-release}"

# Disk needed for torch + vLLM + one small checkpoint, measured on a clean box.
MIN_FREE_GB=25

# CUDA's crt/host_config.h refuses a host gcc newer than this. Verified against BOTH the real
# toolkit (13.3.1, a 4.1 GB install) and the pip nvidia/cu13 headers: both cap at 15, so
# installing the full toolkit does NOT raise it. fedora:44 ships gcc 16, which is why a source
# build there dies with "unsupported GNU version" on every .cu file.
MAX_NVCC_GCC=15

say()  { printf '%s\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

run() {
    # This file runs under IFS=$'\n\t', which makes "$*" join with newlines and prints one
    # argument per line — unreadable, and it defeats the point of --dry-run. Restore a
    # space for the display expansion only; "$@" is unaffected.
    local IFS=' '
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

  --components LIST   core,vllm,picode,chat,quantize   (default: core,vllm,chat)
  --model REPO_ID     checkpoint to serve     (default: chosen interactively)
  --chat WHICH        gradio | openwebui | none
  --glq-version VER   pin glq (default: latest release)
  --glq-source SPEC   install glq from a wheel, path or VCS ref instead of PyPI
  --yes               accept defaults, never prompt (and never start GLQ)
  --start/--no-start  start GLQ + the chat when done, or never offer to
  --list              list available checkpoints and exit
  --preflight         check prerequisites and exit (changes nothing)
  --no-modify-path    don't append the venv bin dir to PATH in ~/.bashrc / ~/.zshrc
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
            --no-modify-path) MODIFY_PATH=0; shift ;;
            --dry-run)      DRY_RUN=1; PASSTHRU+=("$1"); shift ;;
            --allow-root)   ALLOW_ROOT=1; shift ;;
            --glq-version)  [ $# -ge 2 ] || die "--glq-version needs a value"
                            GLQ_VERSION="$2"; shift 2 ;;
            --glq-source)   [ $# -ge 2 ] || die "--glq-source needs a value"
                            GLQ_SOURCE="$2"; shift 2 ;;
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
        # RHEL 9 family BEFORE fedora: RHEL sets ID_LIKE=fedora, so a fedora-first pattern
        # would swallow it and hand out advice that does not work here. Two differences,
        # both measured in containers rather than assumed:
        #   * the default python3 is 3.9 — below glq's 3.10 floor — so an explicit
        #     python3.12 (in AppStream on UBI9/Alma/Amazon 2023) is required, otherwise
        #     pre-flight refuses again after the user has "fixed" it;
        #   * `curl` CONFLICTS with the preinstalled curl-minimal and aborts the whole dnf
        #     transaction, taking gcc with it. Asking for it is worse than omitting it.
        *rhel*|*centos*|*rocky*|*almalinux*|*amzn*)
                                    echo "sudo dnf install -y python3.12 python3.12-devel gcc-c++" ;;
        *fedora*)                   echo "sudo dnf install -y python3-devel gcc-c++ curl" ;;
        *steamos*)                  echo "sudo steamos-readonly disable && sudo pacman -Syu --needed --noconfirm python gcc curl   # or use distrobox" ;;
        *arch*|*manjaro*|*endeavouros*)
                                    echo "sudo pacman -Syu --needed --noconfirm python gcc curl" ;;
        # glibc-devel explicitly: on the minimal Azure Linux core image `gcc-c++` does NOT
        # pull the libc headers, unlike Fedora/RHEL's gcc-c++ or Debian's build-essential.
        # Without it pre-flight passes and the kernel build then dies on
        # `features.h: No such file or directory` — 16 times in one distro-matrix run.
        # Verified in the container: `tdnf install -y glibc-devel` provides /usr/include/features.h.
        *azurelinux*|*mariner*)     echo "sudo tdnf install -y python3-devel gcc-c++ glibc-devel curl" ;;
        *suse*|*sles*)              echo "sudo zypper install -y python3-devel gcc-c++ curl" ;;
        *)                          echo "install with your package manager: python3 (>=3.10) + venv, gcc, curl" ;;
    esac
}

# Where to get a gcc old enough for nvcc, per distro. Only the Fedora line is measured:
# `dnf install gcc15 gcc15-c++` on fedora:44 provides /usr/bin/g++-15, and pointing nvcc at it
# removed every "unsupported GNU version" error (6 -> 0). The others follow each distro's
# usual naming and are NOT verified — they are a starting point, not a promise.
compat_gcc_hint() {
    case " $DISTRO_ID $DISTRO_LIKE " in
        *fedora*)                   echo "sudo dnf install -y gcc${MAX_NVCC_GCC} gcc${MAX_NVCC_GCC}-c++" ;;
        *ubuntu*|*debian*)          echo "sudo apt-get install -y g++-${MAX_NVCC_GCC}   # unverified" ;;
        *arch*|*manjaro*|*endeavouros*)
                                    echo "sudo pacman -S --needed gcc${MAX_NVCC_GCC}   # unverified" ;;
        *suse*|*sles*)              echo "sudo zypper install -y gcc${MAX_NVCC_GCC}-c++   # unverified" ;;
        *)                          echo "install a gcc <= ${MAX_NVCC_GCC} with your package manager" ;;
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
    #
    # Specifically a C++ one: glq_bindings.cpp is C++, and nvcc drives a C++ host compiler.
    # This used to accept `gcc || cc`, which a C-only toolchain satisfies — so on RPM distros,
    # where the `gcc` package is the C compiler alone and `cc1plus` ships in `gcc-c++`,
    # pre-flight reported the machine ready and the build then died with
    # `gcc: fatal error: cannot execute 'cc1plus'`. Measured on fedora:43. Debian-family hid
    # it because build-essential pulls g++ along with gcc.
    if command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1; then
        say "  c++:     present (needed to JIT-build the CUDA extension)"
    elif command -v gcc >/dev/null 2>&1 || command -v cc >/dev/null 2>&1; then
        warn "  c++:     MISSING (a C compiler is present, but the extension is C++)"
        warn "           install it with:  $(pkg_hint)"
    else
        warn "  c++:     MISSING — the fused CUDA kernel cannot be built; GLQ falls back to CPU"
    fi

    # A host gcc newer than CUDA accepts. Deliberately NOT a blocker: since 0.8.6 the prebuilt
    # wheels cover cp310-cp314, so the common path compiles nothing and is unaffected. Only a
    # source install (--glq-source) reaches nvcc, and for that the fix is a compat compiler
    # plus NVCC_CCBIN — measured to remove the error entirely on fedora:44.
    #
    # On a pristine image there is no gcc yet, so this says nothing on the FIRST pre-flight
    # run and appears on the second, once the package advice above has been followed. That is
    # the order the installer is used in (pre-flight -> install packages -> pre-flight), and
    # guessing the version of a compiler that is not installed would be worse than silence.
    local gccmaj=""
    if command -v gcc >/dev/null 2>&1; then
        gccmaj=$(gcc -dumpversion 2>/dev/null | cut -d. -f1)
    fi
    if [ -n "$gccmaj" ] && [ "$gccmaj" -gt "$MAX_NVCC_GCC" ] 2>/dev/null; then
        say "  nvcc:    host gcc is $gccmaj; CUDA supports <= $MAX_NVCC_GCC"
        say "           prebuilt wheels are unaffected — they compile nothing"
        say "           to build from source:  $(compat_gcc_hint)"
        say "           then:  export NVCC_CCBIN=/usr/bin/g++-$MAX_NVCC_GCC"
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

install_cuda_toolchain() {
    # nvcc + the CCCL headers NVIDIA's cuda_fp16.h includes, at the version torch already
    # chose. Deliberately a SEPARATE pip call: resolved alongside glq, pip picks the newest
    # cuda-toolkit, finds it incompatible with torch's exact pin, backtracks, and silently
    # downgrades torch (measured: 2.13.0 -> 2.10.0, after which CUDA_HOME stops resolving).
    # Naming the version already installed cannot move torch, and hardcodes nothing.
    say "== installing the CUDA build toolchain"
    if [ "$DRY_RUN" -eq 1 ]; then
        say "  [dry-run] would install cuda-toolkit[nvcc,cccl] at torch's own pinned version"
        return 0
    fi
    local ver
    ver=$("$GLQ_VENV/bin/python" -c \
        'import importlib.metadata as m; print(m.version("cuda-toolkit"))' 2>/dev/null || true)
    if [ -z "$ver" ]; then
        warn "torch did not pin cuda-toolkit here; skipping nvcc/cccl."
        warn "  GLQ compiles its kernels on first use — if that fails it will say why."
        return 0
    fi
    run "$GLQ_VENV/bin/pip" install --upgrade "cuda-toolkit[nvcc,cccl]==$ver"
    repair_cuda_wheel_layout
}

repair_cuda_wheel_layout() {
    # Give the wheels the layout every CUDA build system expects. They install
    #
    #     nvidia/cu13/lib/libcudart.so.13
    #
    # with no unversioned `libcudart.so` (the only name `-lcudart` resolves through) and no
    # `lib64/` (the directory build systems pass to -L). glq works around this for its own
    # build, but it is not the only thing compiling CUDA in here: vLLM JIT-builds flashinfer
    # at engine start and dies on the same link line. Measured in a container — GLQ's kernels
    # baked, `--verify` green, and vLLM still could not start.
    #
    # Done here rather than delegated to `glq.installer`, because that is whatever glq got
    # installed: every published release predates this, so a new installer would depend on a
    # new glq to fix the venv it just created.
    #
    # Scope: the venv this script created, nothing else. Symlinks only, beside files pip
    # just wrote, never over an existing name — so a wheel that starts shipping them, or a
    # real system toolkit, is untouched.
    [ -x "$GLQ_VENV/bin/python" ] || {
        warn "not a venv at $GLQ_VENV — skipping the CUDA layout repair"
        return 0
    }
    say "== normalising the CUDA wheel layout in the venv (lib64/, libcudart.so)"
    local libdir real base made=0
    for libdir in "$GLQ_VENV"/lib/python*/site-packages/nvidia/*/lib; do
        [ -d "$libdir" ] || continue
        for real in "$libdir"/lib*.so.*; do
            [ -e "$real" ] || continue
            base="${real%%.so.*}.so"
            if [ ! -e "$base" ] && [ ! -L "$base" ]; then
                ln -s "$(basename "$real")" "$base" 2>/dev/null && made=$((made+1))
            fi
        done
        if [ ! -e "${libdir%/lib}/lib64" ] && [ ! -L "${libdir%/lib}/lib64" ]; then
            ln -s lib "${libdir%/lib}/lib64" 2>/dev/null && made=$((made+1))
        fi
    done
    say "   $made symlink(s) created under $GLQ_VENV"
}

install_glq() {
    # --glq-source takes any pip spec: a built wheel, a checkout, a VCS ref. install.sh is
    # a bootstrap that pip-installs glq from PyPI, so without this an install can never be
    # newer than the last release — a fork, an RC, or a fix under test cannot be installed
    # by its own installer, and a container cannot validate one.
    local spec="glq"
    [ -n "$GLQ_VERSION" ] && spec="glq==$GLQ_VERSION"
    [ -n "$GLQ_SOURCE" ] && spec="$GLQ_SOURCE"
    # GLQ compiles its CUDA kernels on first use, so the build toolchain is a runtime
    # requirement: ninja (torch's extension builder), plus nvcc and the CCCL headers that
    # NVIDIA's own cuda_fp16.h includes. ninja rides along with glq — it constrains nothing.
    say "== installing $spec + build toolchain (this pulls PyTorch and takes a few minutes)"
    run "$GLQ_VENV/bin/pip" install --upgrade pip
    run "$GLQ_VENV/bin/pip" install --upgrade "$spec" ninja
    install_cuda_toolchain
}

ensure_venv_on_path() {
    # Every instruction this installer prints, and every model-card snippet, uses the venv
    # by absolute path — so nothing ever puts $GLQ_VENV/bin on PATH. That is fine until a
    # tool inside the venv is invoked by bare name from another tool: FlashInfer JIT-builds
    # its sm_120 sampler at engine start and runs `ninja` via subprocess, which resolves
    # through PATH and kills EngineCore with FileNotFoundError on an otherwise perfect
    # install. Appending (not prepending) keeps the system python/pip winning; only names
    # the system lacks — ninja, vllm, glq-* — fall through to the venv.
    #
    # The interactive rc file (.bashrc / .zshrc) rather than ~/.profile: Ubuntu's stock
    # .bashrc returns early for non-interactive shells, and the interactive-ssh session is
    # exactly where users type the bare commands. Non-interactive invocations should keep
    # passing explicit env. The file follows the LOGIN shell ($SHELL), not the shell
    # running this script — the one-liner always executes under bash, so $SHELL is the
    # only signal a zsh user emits.
    if [ "$MODIFY_PATH" -eq 0 ]; then
        say "== leaving PATH alone (--no-modify-path)"
        return 0
    fi
    case ":$PATH:" in *":$GLQ_VENV/bin:"*)
        say "== $GLQ_VENV/bin is already on PATH — not touching shell rc files"
        return 0 ;;
    esac
    local rc
    case "$(basename "${SHELL:-bash}")" in
        bash) rc="$HOME/.bashrc" ;;
        zsh)  rc="$HOME/.zshrc" ;;
        *)
            # fish and friends do not speak POSIX `export`; writing bash syntax into
            # their config would break every new shell. Print the line, let the user
            # place it.
            say "== login shell $(basename "${SHELL:-?}") — add this yourself if wanted:"
            say '   export PATH="$PATH:'"$GLQ_VENV"'/bin"'
            return 0 ;;
    esac
    if [ -f "$rc" ] && grep -qF "$GLQ_VENV/bin" "$rc"; then
        say "== $rc already references $GLQ_VENV/bin — leaving it as is"
        return 0
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  [dry-run] append to %s: export PATH="$PATH:%s/bin"\n' "$rc" "$GLQ_VENV"
        return 0
    fi
    say "== appending $GLQ_VENV/bin to PATH in $rc (opt out: --no-modify-path)"
    printf '\n# added by glq install.sh — lets ninja/vllm/glq-* resolve by name (remove with the export)\nexport PATH="$PATH:%s/bin"\n' "$GLQ_VENV" >> "$rc"
    say "   takes effect in new shells; for this one: source $rc"
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

    # Both decide which glq gets installed; letting one silently win installs a version the
    # user did not ask for.
    if [ -n "$GLQ_SOURCE" ] && [ -n "$GLQ_VERSION" ]; then
        die "--glq-source and --glq-version are mutually exclusive: --glq-source already
    names exactly what to install."
    fi

    # --preflight comes BEFORE the root check on purpose. It changes nothing, and the
    # people most likely to run it as root are the ones who need it most: anyone inside a
    # container, where uid 0 is the default. Refusing to even report what is missing, on
    # account of a guard that exists to protect the *install*, helps nobody.
    if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
        preflight || die "pre-flight failed — install the packages above and re-run"
        say "Pre-flight OK. Run without --preflight to install."
        exit 0
    fi

    check_not_root
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
    ensure_venv_on_path
    hand_over
}

main "$@"
