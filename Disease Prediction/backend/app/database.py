from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

class Database:
    client: AsyncIOMotorClient = None
    db = None

db = Database()

async def connect_to_mongo():
    db.client = AsyncIOMotorClient(settings.MONGODB_URI)
    db.db = db.client[settings.DB_NAME]
    
    # Create indexes
    await db.db.users.create_index("email", unique=True)
    await db.db.predictions.create_index("user_id")
    await db.db.predictions.create_index("disease")
    await db.db.model_metrics.create_index([("disease", 1), ("algorithm", 1)])

async def close_mongo_connection():
    if db.client is not None:
        db.client.close()

def get_database():
    return db.db
