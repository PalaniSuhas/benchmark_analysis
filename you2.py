from youdotcom import You
from dotenv import load_dotenv
import os

load_dotenv()

YOU_API_KEY = os.getenv("YOUDOTCOM_API_KEY")

with You(YOU_API_KEY) as you:
    res = you.search.unified(
        query="best practices for scaling microservices architecture in production",
    )

    # Print results with snippets
    # Snippets are query-relevant text excerpts extracted from each page,
    # highlighting the passages most relevant to your search query
    if res.results and res.results.web:
        for result in res.results.web:
            print(f"{result.title}")
            if result.snippets:
                print(f"  {result.snippets[0]}\n")
                
                
                
                
                
                
                
                