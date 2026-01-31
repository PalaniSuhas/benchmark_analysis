import os
import time
import json
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
from tavily import TavilyClient

load_dotenv()

# ============================================================================
# BRAVE SEARCH WRAPPER
# ============================================================================
class BraveSearch:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key
        }
        self.client = httpx.Client(headers=self.headers)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def search(self, query: str, count: int = 5):
        params = {"q": query, "count": count, "extra_snippets": "true"}
        resp = self.client.get(self.base_url, params=params)
        resp.raise_for_status()
        return resp.json()


# ============================================================================
# CONFIGURATION
# ============================================================================
OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GEMINI_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
TAVILY_CLIENT = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

# Test queries across different domains
TEST_QUERIES = {
    "geopolitics": "latest global geopolitical developments January 2026",
    "sports": "major sports events and highlights January 2026",
    "tech": "breakthrough AI and technology developments 2026",
    "business": "Tesla vs BYD electric vehicle market performance 2025 2026",
    "science": "recent scientific discoveries and breakthroughs 2026"
}

# LLM Models to test
EVAL_MODELS = [
    # OpenAI
    {"name": "gpt-4o", "provider": "openai", "label": "GPT-4o"},
    {"name": "gpt-4o-mini", "provider": "openai", "label": "GPT-4o Mini"},
    {"name": "gpt-4.1", "provider": "openai", "label": "GPT-4.1"},

    {"name": "gpt-5", "provider": "openai", "label": "GPT-5"},
    {"name": "gpt-5-mini", "provider": "openai", "label": "GPT-5 Mini"},
    {"name": "gpt-5-nano", "provider": "openai", "label": "GPT-5 Nano"},
    {"name": "gpt-5.2", "provider": "openai", "label": "GPT-5.2"},
    {"name": "gpt-5.2-pro", "provider": "openai", "label": "GPT-5.2 Pro"},

    # Gemini
    {"name": "gemini-2.0-flash-exp", "provider": "gemini", "label": "Gemini 2.0 Flash"},
    {"name": "gemini-2.5-flash", "provider": "gemini", "label": "Gemini 2.5 Flash"},
    {"name": "gemini-2.5-pro", "provider": "gemini", "label": "Gemini 2.5 Pro"},
]


# ============================================================================
# SEARCH PROVIDER FUNCTIONS
# ============================================================================
def search_tavily(query):
    """Search using Tavily"""
    start = time.perf_counter()
    try:
        response = TAVILY_CLIENT.search(
            query=query,
            topic="news",
            search_depth="advanced",
            max_results=5
        )
        results = response.get("results", [])
        context_items = [
            {
                "title": r.get("title", ""),
                "snippet": r.get("content", "")[:300],
                "url": r.get("url", ""),
                "score": r.get("score", 0)
            }
            for r in results[:3]  # Top 3 results
        ]
        search_time = time.perf_counter() - start
        
        # Combine into searchable context
        context = "\n\n".join([
            f"Title: {item['title']}\nSnippet: {item['snippet']}\nSource: {item['url']}"
            for item in context_items
        ])
        
        return {
            "provider": "Tavily",
            "context": context,
            "num_results": len(context_items),
            "search_time": round(search_time, 3),
            "sources": [item['url'] for item in context_items]
        }
    except Exception as e:
        print(f"Tavily error: {e}")
        return {
            "provider": "Tavily",
            "context": None,
            "num_results": 0,
            "search_time": 0,
            "error": str(e)
        }


def search_brave(query):
    """Search using Brave"""
    start = time.perf_counter()
    try:
        with BraveSearch(BRAVE_API_KEY) as brave:
            data = brave.search(query, count=5)
            results = data.get("web", {}).get("results", [])
            
            context_items = []
            for r in results[:3]:  # Top 3 results
                main_desc = r.get("description", "")
                extra = " ".join(r.get("extra_snippets", []))
                combined = f"{main_desc} {extra}".strip()
                context_items.append({
                    "title": r.get("title", ""),
                    "snippet": combined[:300],
                    "url": r.get("url", "")
                })
            
            search_time = time.perf_counter() - start
            
            # Combine into searchable context
            context = "\n\n".join([
                f"Title: {item['title']}\nSnippet: {item['snippet']}\nSource: {item['url']}"
                for item in context_items
            ])
            
            return {
                "provider": "Brave",
                "context": context,
                "num_results": len(context_items),
                "search_time": round(search_time, 3),
                "sources": [item['url'] for item in context_items]
            }
    except Exception as e:
        print(f"Brave error: {e}")
        return {
            "provider": "Brave",
            "context": None,
            "num_results": 0,
            "search_time": 0,
            "error": str(e)
        }


# ============================================================================
# LLM EVALUATION FUNCTIONS
# ============================================================================
def evaluate_with_llm(model_config, query, search_result):
    """Evaluate search results using specified LLM"""
    if not search_result.get("context"):
        return None, 0
    
    eval_prompt = f"""You are evaluating search engine quality for RAG applications.

ORIGINAL QUERY: "{query}"

SEARCH PROVIDER: {search_result['provider']}
SEARCH TIME: {search_result['search_time']}s
NUMBER OF RESULTS: {search_result['num_results']}

SEARCH CONTEXT:
{search_result['context']}

Evaluate this search result on the following criteria:
1. Relevance: How well do results match the query intent? (0-10)
2. Freshness: How recent and up-to-date is the information? (0-10)
3. Quality: How authoritative and comprehensive are the sources? (0-10)
4. Usefulness: How useful would this be for answering the query? (0-10)
5. Coverage: How well does this cover different aspects of the topic? (0-10)

RETURN STRICT JSON ONLY:
{{
  "relevance_score": 0,
  "freshness_score": 0,
  "quality_score": 0,
  "usefulness_score": 0,
  "coverage_score": 0,
  "overall_score": 0.0,
  "reasoning": "Brief explanation of scores"
}}
"""
    
    start = time.perf_counter()
    
    try:
        if model_config['provider'] == "openai":
            response = OPENAI_CLIENT.chat.completions.create(
                model=model_config['name'],
                messages=[
                    {"role": "system", "content": "You are a search quality evaluator. Return only valid JSON."},
                    {"role": "user", "content": eval_prompt}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
        else:  # gemini
            response = GEMINI_CLIENT.models.generate_content(
                model=model_config['name'],
                contents=eval_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            result_text = response.text.strip()
            if result_text.startswith("```"):
                lines = result_text.split("\n")
                result_text = "\n".join(lines[1:-1])
            result = json.loads(result_text.strip())
        
        eval_time = time.perf_counter() - start
        return result, round(eval_time, 3)
        
    except Exception as e:
        print(f"Evaluation error with {model_config['label']}: {e}")
        return None, 0


# ============================================================================
# MAIN BENCHMARK ORCHESTRATOR
# ============================================================================
class UltimateSearchBenchmark:
    def __init__(self):
        self.results = []
    
    def run_single_query_benchmark(self, category, query):
        """Run benchmark for a single query across all providers and models"""
        print(f"\n{'='*70}")
        print(f"CATEGORY: {category.upper()}")
        print(f"QUERY: {query}")
        print(f"{'='*70}")
        
        query_results = {
            "category": category,
            "query": query,
            "providers": []
        }
        
        # Test each search provider
        search_providers = [
            ("Tavily", search_tavily),
            ("Brave", search_brave),
        ]
        
        for provider_name, search_func in search_providers:
            print(f"\n--- Testing {provider_name} ---")
            search_result = search_func(query)
            
            if not search_result.get("context"):
                print(f"⚠️  {provider_name} failed to return results")
                continue
            
            print(f"Search time: {search_result['search_time']}s")
            print(f"Results found: {search_result['num_results']}")
            
            provider_data = {
                "provider": provider_name,
                "search_time": search_result['search_time'],
                "num_results": search_result['num_results'],
                "sources": search_result.get('sources', []),
                "evaluations": []
            }
            
            # Evaluate with each LLM
            for model in EVAL_MODELS:
                print(f"  ⚡ Evaluating with {model['label']}...")
                eval_result, eval_time = evaluate_with_llm(model, query, search_result)
                
                if eval_result:
                    provider_data['evaluations'].append({
                        "model": model['label'],
                        "eval_time": eval_time,
                        "scores": eval_result
                    })
                    print(f"     Overall score: {eval_result.get('overall_score', 'N/A')}")
            
            query_results['providers'].append(provider_data)
        
        self.results.append(query_results)
        return query_results
    
    def run_full_benchmark(self):
        """Run complete benchmark across all queries"""
        print("="*70)
        print("ULTIMATE SEARCH PROVIDER BENCHMARK")
        print("Providers: Tavily, Brave")
        print(f"Evaluators: {', '.join([m['label'] for m in EVAL_MODELS])}")
        print("="*70)
        
        # Run benchmarks for all queries
        for category, query in TEST_QUERIES.items():
            self.run_single_query_benchmark(category, query)
        
        # Generate final analysis
        self.generate_final_verdict()
    
    def generate_final_verdict(self):
        """Generate comprehensive final verdict using LLM"""
        print(f"\n\n{'#'*70}")
        print("GENERATING FINAL VERDICT")
        print(f"{'#'*70}")
        
        # Prepare comprehensive summary for LLM judge
        summary_data = {
            "total_queries": len(TEST_QUERIES),
            "providers_tested": ["Tavily", "Brave"],
            "models_used": [m['label'] for m in EVAL_MODELS],
            "detailed_results": self.results
        }
        
        verdict_prompt = f"""You are an expert analyst evaluating search providers for RAG applications.

BENCHMARK DATA:
{json.dumps(summary_data, indent=2)}

Analyze the complete benchmark results and provide a comprehensive verdict.

Calculate for each provider:
1. Average search time across all queries
2. Average evaluation scores across all models and queries
3. Consistency of results (standard deviation)
4. Best use cases

Determine:
- Overall winner for SPEED
- Overall winner for QUALITY
- Overall winner for FRESHNESS
- Overall winner for RELIABILITY
- Best provider for each category (geopolitics, sports, tech, business, science)

RETURN STRICT JSON:
{{
  "overall_winner": {{
    "provider": "...",
    "reasoning": "..."
  }},
  "speed_winner": {{
    "provider": "...",
    "avg_search_time": 0.0,
    "reasoning": "..."
  }},
  "quality_winner": {{
    "provider": "...",
    "avg_overall_score": 0.0,
    "reasoning": "..."
  }},
  "freshness_winner": {{
    "provider": "...",
    "avg_freshness_score": 0.0,
    "reasoning": "..."
  }},
  "category_recommendations": {{
    "geopolitics": "...",
    "sports": "...",
    "tech": "...",
    "business": "...",
    "science": "..."
  }},
  "provider_rankings": [
    {{"rank": 1, "provider": "...", "total_score": 0.0}},
    {{"rank": 2, "provider": "...", "total_score": 0.0}}
  ],
  "key_findings": ["finding1", "finding2", "finding3"],
  "final_recommendation": "Comprehensive recommendation for which provider to use and when"
}}
"""
        
        print("\n🤖 AI Judge (GPT-4o) is analyzing all results...")
        
        try:
            judge_response = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert search quality analyst with deep expertise in RAG systems."},
                    {"role": "user", "content": verdict_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            verdict = json.loads(judge_response.choices[0].message.content)
            
            # Display verdict
            print(f"\n{'='*70}")
            print("🏆 FINAL VERDICT")
            print(f"{'='*70}")
            
            print(f"\n🥇 OVERALL WINNER: {verdict['overall_winner']['provider']}")
            print(f"   {verdict['overall_winner']['reasoning']}")
            
            print(f"\n⚡ SPEED WINNER: {verdict['speed_winner']['provider']}")
            print(f"   Average search time: {verdict['speed_winner']['avg_search_time']}s")
            print(f"   {verdict['speed_winner']['reasoning']}")
            
            print(f"\n⭐ QUALITY WINNER: {verdict['quality_winner']['provider']}")
            print(f"   Average quality score: {verdict['quality_winner']['avg_overall_score']}")
            print(f"   {verdict['quality_winner']['reasoning']}")
            
            print(f"\n🆕 FRESHNESS WINNER: {verdict['freshness_winner']['provider']}")
            print(f"   Average freshness score: {verdict['freshness_winner']['avg_freshness_score']}")
            print(f"   {verdict['freshness_winner']['reasoning']}")
            
            print(f"\n📊 PROVIDER RANKINGS:")
            for ranking in verdict['provider_rankings']:
                print(f"   {ranking['rank']}. {ranking['provider']} (Score: {ranking['total_score']})")
            
            print(f"\n🎯 CATEGORY RECOMMENDATIONS:")
            for cat, rec in verdict['category_recommendations'].items():
                print(f"   {cat.title()}: {rec}")
            
            print(f"\n💡 KEY FINDINGS:")
            for i, finding in enumerate(verdict['key_findings'], 1):
                print(f"   {i}. {finding}")
            
            print(f"\n📝 FINAL RECOMMENDATION:")
            print(f"   {verdict['final_recommendation']}")
            
            # Save complete results
            final_output = {
                "benchmark_metadata": {
                    "date": "2026-01-30",
                    "total_queries": len(TEST_QUERIES),
                    "providers": ["Tavily", "Brave"],
                    "evaluators": [m['label'] for m in EVAL_MODELS]
                },
                "detailed_results": self.results,
                "verdict": verdict
            }
            
            output_file = 'ultimate_search_benchmark_results.json'
            with open(output_file, 'w') as f:
                json.dump(final_output, f, indent=2)
            
            print(f"\n\n📁 Complete results saved to: {output_file}")
            print("="*70)
            
            return final_output
            
        except Exception as e:
            print(f"Error generating verdict: {e}")
            return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    benchmark = UltimateSearchBenchmark()
    results = benchmark.run_full_benchmark()