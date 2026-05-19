import oracledb
from config import DB_CONFIG

def get_db_connection():
    connection = oracledb.connect(
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        dsn=DB_CONFIG["dsn"]
    )
    return connection