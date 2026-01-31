import os
import httpx
from dotenv import load_dotenv

class Brave:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key
        }
        self.client = httpx.Client(headers=self.headers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def search(self, query: str, count: int = 5):
        params = {"q": query, "count": count}
        response = self.client.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()

# --- HOW TO USE IT (Identical to your You.com style) ---

load_dotenv()
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

with Brave(BRAVE_API_KEY) as brave:
    res = brave.search("president of USA")

    # Brave's JSON structure puts results inside 'web' -> 'results'
    results = res.get("web", {}).get("results", [])
    
    for result in results:
        print(f"{result.get('title')}")
        # Brave's equivalent to 'snippet' is 'description'
        snippet = result.get("description")
        if snippet:
            print(f"  {snippet}\n")
            
            
            
                   
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            