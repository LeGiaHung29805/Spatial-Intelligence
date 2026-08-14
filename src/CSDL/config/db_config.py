import os

from sqlalchemy import create_engine

DB_SETTINGS = {
    "user": os.getenv("DB_USER", "postgres"),
    "pass": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "db": os.getenv("DB_NAME", "GuardBatXat"),
}

def get_engine():
    if not DB_SETTINGS["pass"]:
        raise RuntimeError("DB_PASSWORD must be configured")
    url = f"postgresql://{DB_SETTINGS['user']}:{DB_SETTINGS['pass']}@{DB_SETTINGS['host']}:{DB_SETTINGS['port']}/{DB_SETTINGS['db']}"
    return create_engine(url)
