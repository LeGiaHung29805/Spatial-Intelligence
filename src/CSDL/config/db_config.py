from sqlalchemy import create_engine

DB_SETTINGS = {
    "user": "postgres",
    "pass": "hieu040205",
    "host": "localhost",
    "port": "5432",
    "db": "GuardBatXat"
}

def get_engine():
    url = f"postgresql://{DB_SETTINGS['user']}:{DB_SETTINGS['pass']}@{DB_SETTINGS['host']}:{DB_SETTINGS['port']}/{DB_SETTINGS['db']}"
    return create_engine(url)