import subprocess
import sys
import os

def install_requirements():
    print("Installing Requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_dashboard():
    print("Starting dashboard at http://localhost:5000")
    os.system(f"{sys.executable} dashboard.py")

if __name__ == "__main__":
    install_requirements()
    run_dashboard()