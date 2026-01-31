import os
import time
import json
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from youdotcom import You

load_dotenv()

# Clients
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
YOU_API_KEY = os.getenv("YOUDOTCOM_API_KEY")

# 2026 Model Mapping
MODELS = [
    {"name": "gpt-5.2", "provider": "openai", "label": "GPT-5.2 (Flagship)"},
    {"name": "gpt-4.1-mini", "provider": "openai", "label": "GPT-4.1 Mini"},
    {"name": "gemini-3-pro-preview", "provider": "gemini", "label": "Gemini 3 Pro"},
    {"name": "gemini-3-flash-preview", "provider": "gemini", "label": "Gemini 3 Flash"},
    {"name": "gemini-2.5-flash", "provider": "gemini", "label": "Gemini 2.5 Flash (Legacy Stable)"}
]

CATEGORIES = {
    "geopolitics": "latest global geopolitical shifts 2026",
    "sports": "major sports highlights January 2026",
    "tech": "latest AI and hardware breakthroughs 2026"
}

def fetch_you_context(query):
    try:
        with You(YOU_API_KEY) as you:
            res = you.search.unified(query=query)
            if not res.results or not res.results.web: return None, None
            first_hit = res.results.web[0]
            context = " ".join(first_hit.snippets) if first_hit.snippets else first_hit.description
            return context, first_hit.url
    except: return None, None

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
        print(f"\n🌍 Category: {cat.upper()}")
        context, url = fetch_you_context(query)
        if not context: continue

        prompt = f"Summarize this context in 2 sentences:\n{context}"
        cat_results = {"category": cat, "context": context, "benchmarks": []}

        for m in MODELS:
            print(f"  ⚡ Testing {m['label']}...")
            output, latency = benchmark_llm(m['name'], m['provider'], prompt)
            cat_results["benchmarks"].append({
                "model": m['label'],
                "latency": latency,
                "summary": output
            })
        results.append(cat_results)

    # --- LLM VALIDATION PHASE ---
    print("\n🧠 AI is analyzing the winners...")
    analysis_prompt = f"Compare these benchmarking results. Based on latency and summary quality, which model is the overall winner for speed-sensitive summarization? Results JSON: {json.dumps(results)}"
    
    # We use the flagship GPT-5.2 to perform the final judging
    judge_response = client_openai.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "system", "content": "You are a performance analyst."}, {"role": "user", "content": analysis_prompt}]
    )
    
    final_report = {
        "raw_data": results,
        "ai_judgment": judge_response.choices[0].message.content
    }

    with open("final2_benchmark_2026.json", "w") as f:
        json.dump(final_report, f, indent=4)
    
    print("\n" + "="*30)
    print("🏆 FINAL AI JUDGMENT:")
    print(final_report["ai_judgment"])
    print("="*30)

if __name__ == "__main__":
    run_benchmark()