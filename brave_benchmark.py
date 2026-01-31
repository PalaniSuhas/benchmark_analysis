import os
import time
import json
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

load_dotenv()

# --- BRAVE WRAPPER (No-Conflict Version) ---
class Brave:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key
        }
        self.client = httpx.Client(headers=self.headers)

    def __enter__(self): return self
    def __exit__(self, *args): self.client.close()

    def search(self, query: str):
        # We enable extra_snippets=True for better AI context
        params = {"q": query, "count": 3, "extra_snippets": "true"}
        resp = self.client.get(self.base_url, params=params)
        resp.raise_for_status()
        return resp.json()

# --- CLIENTS & CONFIG ---
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

MODELS = [
    {"name": "gpt-5.2", "provider": "openai", "label": "GPT-5.2 (Flagship)"},
    {"name": "gpt-4.1-mini", "provider": "openai", "label": "GPT-4.1 Mini"},
    {"name": "gemini-3-pro-preview", "provider": "gemini", "label": "Gemini 3 Pro"},
    {"name": "gemini-3-flash-preview", "provider": "gemini", "label": "Gemini 3 Flash"}
]

CATEGORIES = {
    "geopolitics": "latest global geopolitical shifts 2026",
    "sports": "major sports highlights January 2026",
    "tech": "latest AI and hardware breakthroughs 2026"
}

def fetch_brave_context(query):
    try:
        with Brave(BRAVE_API_KEY) as brave:
            data = brave.search(query)
            results = data.get("web", {}).get("results", [])
            if not results: return None, None
            
            first_hit = results[0]
            # Combine main description with extra snippets for a rich context
            main_desc = first_hit.get("description", "")
            extra = " ".join(first_hit.get("extra_snippets", []))
            context = f"{main_desc} {extra}".strip()
            
            return context, first_hit.get("url")
    except Exception as e:
        print(f"Brave Error: {e}")
        return None, None

def benchmark_llm(model_id, provider, prompt):
    start = time.perf_counter()
    try:
        if provider == "openai":
            resp = client_openai.chat.completions.create(model=model_id, messages=[{"role": "user", "content": prompt}])
            text = resp.choices[0].message.content
        else:
            resp = client_gemini.models.generate_content(model=model_id, contents=prompt)
            text = resp.text
        return text, round(time.perf_counter() - start, 3)
    except Exception as e:
        return f"Error: {str(e)}", 0

def run_benchmark():
    results = []
    for cat, query in CATEGORIES.items():
        print(f"\n🦁 Category: {cat.upper()} (via Brave)")
        context, url = fetch_brave_context(query)
        if not context: continue

        prompt = f"Summarize this context in 2 sentences:\n{context}"
        cat_results = {"category": cat, "context_source": url, "benchmarks": []}

        for m in MODELS:
            print(f"  ⚡ Testing {m['label']}...")
            output, latency = benchmark_llm(m['name'], m['provider'], prompt)
            cat_results["benchmarks"].append({
                "model": m['label'],
                "latency": latency,
                "summary": output
            })
        results.append(cat_results)

    # --- JUDGMENT PHASE ---
    print("\n🧠 AI Analyst is evaluating...")
    analysis_prompt = f"Analyze these 2026 benchmarks. Who is the winner for speed vs quality? {json.dumps(results)}"
    
    judge = client_openai.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "system", "content": "You are a performance analyst."}, {"role": "user", "content": analysis_prompt}]
    )
    
    with open("brave_benchmark_2026.json", "w") as f:
        json.dump({"results": results, "judgment": judge.choices[0].message.content}, f, indent=4)
    
    print("\n🏆 FINAL BRAVE-GROUNDED JUDGMENT:")
    print(judge.choices[0].message.content)

if __name__ == "__main__":
    run_benchmark()