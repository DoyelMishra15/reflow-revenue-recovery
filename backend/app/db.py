import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./reflow.db")

# For file-based SQLite URLs, make sure the parent directory exists before
# the engine (and Base.metadata.create_all() in main.py) ever touches the
# file. sqlite3 will happily create the *file*, but not any missing parent
# directories - so on a fresh volume/container (e.g. /app/data_volume) the
# very first connection fails with "unable to open database file". Handles
# both relative ("sqlite:///./data_volume/reflow.db") and absolute
# ("sqlite:////app/data_volume/reflow.db") forms; ":memory:" is skipped
# since dirname() is empty for it.
if DATABASE_URL.startswith("sqlite:///"):
    _db_path = DATABASE_URL[len("sqlite:///"):]
    _db_dir = os.path.dirname(_db_path)
    if _db_dir:
        os.makedirs(_db_dir, exist_ok=True)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
