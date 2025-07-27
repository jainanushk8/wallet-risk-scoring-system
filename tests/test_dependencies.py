"""Quick dependency test"""
try:
    import pandas
    print("✅ pandas installed")
except ImportError:
    print("❌ pandas missing")

try:
    import requests
    print("✅ requests installed")
except ImportError:
    print("❌ requests missing")

try:
    import web3
    print("✅ web3 installed")
except ImportError:
    print("❌ web3 missing")

try:
    import dotenv
    print("✅ python-dotenv installed")
except ImportError:
    print("❌ python-dotenv missing")

try:
    import tqdm
    print("✅ tqdm installed")
except ImportError:
    print("❌ tqdm missing")

try:
    import numpy
    print("✅ numpy installed")
except ImportError:
    print("❌ numpy missing")
