import os, subprocess, sys
print("python", sys.version)
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout[:800])
print("inputs:", os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else None)
for root, dirs, files in os.walk("/kaggle/input"):
    print(root, len(files)); 
    if root.count("/") > 4: break
