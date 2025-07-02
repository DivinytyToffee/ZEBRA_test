import subprocess
import sys


def install_dependencies():
    """Installs all required dependencies for the project."""
    required_packages = [
        "virtualenv",
        "numpy==1.26.4",
        "torch==2.3.0+cu118",
        "torchvision==0.18.0+cu118",
        "ultralytics==8.3.160",
        "opencv-python",
        "rarfile"
    ]

    print("Setting up virtual environment and installing dependencies...")
    subprocess.run([sys.executable, "-m", "virtualenv", ".venv"], check=True)

    activate_script = ".venv\\Scripts\\activate" if sys.platform == "win32" else ".venv/bin/activate"
    for package in required_packages:
        if "+cu118" in package:
            cmd = [sys.executable, "-m", "pip", "install", package, "--index-url",
                   "https://download.pytorch.org/whl/cu118"]
        else:
            cmd = [sys.executable, "-m", "pip", "install", package]
        subprocess.run(cmd, check=True, shell=True)

    print("Dependencies installed successfully!")


if __name__ == '__main__':
    install_dependencies()
