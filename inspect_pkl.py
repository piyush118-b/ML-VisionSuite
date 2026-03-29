import pickle
import pathlib

PKL_DIR = pathlib.Path("./pkls")
IMAGE_PKL = PKL_DIR / "optimized_champion_package.pkl"

with open(IMAGE_PKL, "rb") as f:
    pkg = pickle.load(f)

print("IMAGE_PKL Keys:", pkg.keys())
if "classes" in pkg:
    print("Classes:", pkg["classes"])
