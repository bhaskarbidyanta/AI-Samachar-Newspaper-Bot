import pymongo
import urllib.parse
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Encode username and password properly
#username = urllib.parse.quote_plus(os.getenv("MONGO_USERNAME"))
#password = urllib.parse.quote_plus(os.getenv("MONGO_PASSWORD"))
#cluster = os.getenv("MONGO_CLUSTER")

# Construct the MongoDB URI

MONGO_URI = os.getenv("MONGO_URI")
# Connect to MongoDB

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client["pdf_chatbot"]

# Collections
users_collection = db["users"]
pdfs_collection = db["pdfs"]