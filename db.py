import pymongo
import urllib.parse
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fetch credentials
username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
cluster = os.getenv("MONGO_CLUSTER")

# Ensure credentials are set
if not username or not password or not cluster:
    raise ValueError("MONGO_USERNAME, MONGO_PASSWORD, or MONGO_CLUSTER is missing in environment variables.")

# Encode username and password properly
username = urllib.parse.quote_plus(username)
password = urllib.parse.quote_plus(password)

# Construct the MongoDB URI
MONGO_URI = f"mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority"

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client["pdf_chatbot"]

# Collections
users_collection = db["users"]
pdfs_collection = db["pdfs"]

print("Connected to MongoDB successfully!")
