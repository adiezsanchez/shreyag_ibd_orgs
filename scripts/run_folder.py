"""Cross-platform: run folder script (bash on Unix, PowerShell on Windows)."""
import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    config = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    if not folder:
        print("Usage: pixi run run_folder <image_folder> [config.yaml]", file=sys.stderr)
        sys.exit(1)

    print(f"run_folder: folder={folder!r}, config={config!r}", flush=True)
    print(f"Platform: {'Windows' if sys.platform == 'win32' else 'Unix'}", flush=True)
    print("Starting pipeline (output from main.py will appear below)...", flush=True)

    if sys.platform == "win32":
        script = project_root / "scripts" / "run_folder_powershell.ps1"
        subprocess.run(
            [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                "-ImageFolder",
                folder,
                "-Config",
                config,
            ],
            check=True,
            cwd=project_root,
        )
    else:
        script = project_root / "scripts" / "run_folder_bash.sh"
        subprocess.run(
            ["bash", str(script), folder, config],
            check=True,
            cwd=project_root,
        )

    print("run_folder: all images done.", flush=True)


if __name__ == "__main__":
    main()
