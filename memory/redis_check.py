import os

from redis import Redis

# Use the environment variable if set, otherwise default to localhost
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

redis_client = Redis.from_url(REDIS_URL)
print(redis_client.ping())