import os
from pathlib import Path

def load_master_key():
    """Automatically load master key from environment file"""
    env_file = Path(__file__).parent / "master_key.env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('CREDENTIAL_MASTER_KEY='):
                    key = line.split('=', 1)[1].strip()
                    os.environ['CREDENTIAL_MASTER_KEY'] = key
                    return True
    return False

# Auto-load on import
load_master_key()
