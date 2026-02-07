"""
Startup script for the Enhanced GDELT API
"""
import subprocess
import sys
import os

def main():
    backend_dir = r"c:/Users/Emman/Documents/AI_dev/GDELT_ConflictPredictor/prototype/backend"
    os.chdir(backend_dir)
    
    print(" Starting Enhanced GDELT Conflict Predictor...")
    print(" Backend Directory:", backend_dir)
    print(" API URL: http://127.0.0.1:8080")
    print(" Documentation: http://127.0.0.1:8080/docs")
    
    try:
        # Activate venv and run server
        cmd = [
            r"./venv/Scripts/activate.bat", "&&",
            "python", "-m", "uvicorn", "enhanced_gru_api:app", 
            "--host", "127.0.0.1", "--port", "8080", "--reload"
        ]
        
        subprocess.run(" ".join(cmd), shell=True)
    except KeyboardInterrupt:
        print("\n Server stopped by user")
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    main()