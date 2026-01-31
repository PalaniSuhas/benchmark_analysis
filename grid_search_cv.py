import os
import time
import json
import httpx
import itertools
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
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

    def search(self, query: str, count: int = 5, **kwargs):
        params = {
            "q": query, 
            "count": count, 
            "extra_snippets": kwargs.get("extra_snippets", "true")
        }
        resp = self.client.get(self.base_url, params=params)
        resp.raise_for_status()
        return resp.json()


# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclass
class SearchConfig:
    """Configuration for search provider hyperparameters"""
    provider: str
    max_results: int
    search_depth: str = "basic"
    extra_snippets: bool = True
    topic: str = "general"


@dataclass
class EvaluatorConfig:
    """Configuration for LLM evaluator"""
    name: str
    provider: str
    label: str
    temperature: float = 0.1
    max_tokens: int = 1000


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    search_provider: str
    search_config: Dict[str, Any]
    evaluator: str
    evaluator_config: Dict[str, Any]
    query: str
    category: str
    search_time: float
    eval_time: float
    num_results: int
    scores: Dict[str, float]
    sources: List[str]
    

# ============================================================================
# CONFIGURATION - GRID SEARCH PARAMETERS
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

# GRID SEARCH: Search Provider Configurations
SEARCH_PROVIDER_GRID = {
    "Tavily": [
        {"max_results": 3, "search_depth": "basic", "topic": "general"},
        {"max_results": 5, "search_depth": "advanced", "topic": "news"},
        {"max_results": 3, "search_depth": "advanced", "topic": "news"},
    ],
    "Brave": [
        {"max_results": 3, "extra_snippets": True},
        {"max_results": 5, "extra_snippets": True},
        {"max_results": 3, "extra_snippets": False},
    ]
}

# GRID SEARCH: LLM Evaluator Configurations (FIXED)
EVALUATOR_GRID = [
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
# SEARCH PROVIDER FUNCTIONS WITH HYPERPARAMETERS
# ============================================================================
def search_tavily(query: str, config: Dict[str, Any]) -> Dict:
    """Search using Tavily with configurable parameters"""
    start = time.perf_counter()
    try:
        response = TAVILY_CLIENT.search(
            query=query,
            topic=config.get("topic", "general"),
            search_depth=config.get("search_depth", "basic"),
            max_results=config.get("max_results", 5)
        )
        results = response.get("results", [])
        max_res = config.get("max_results", 5)
        
        context_items = [
            {
                "title": r.get("title", ""),
                "snippet": r.get("content", "")[:300],
                "url": r.get("url", ""),
                "score": r.get("score", 0)
            }
            for r in results[:max_res]
        ]
        search_time = time.perf_counter() - start
        
        context = "\n\n".join([
            f"Title: {item['title']}\nSnippet: {item['snippet']}\nSource: {item['url']}"
            for item in context_items
        ])
        
        return {
            "provider": "Tavily",
            "config": config,
            "context": context,
            "num_results": len(context_items),
            "search_time": round(search_time, 3),
            "sources": [item['url'] for item in context_items]
        }
    except Exception as e:
        print(f"Tavily error: {e}")
        return {
            "provider": "Tavily",
            "config": config,
            "context": None,
            "num_results": 0,
            "search_time": 0,
            "error": str(e)
        }


def search_brave(query: str, config: Dict[str, Any]) -> Dict:
    """Search using Brave with configurable parameters"""
    start = time.perf_counter()
    try:
        with BraveSearch(BRAVE_API_KEY) as brave:
            max_res = config.get("max_results", 5)
            data = brave.search(query, count=max_res, **config)
            results = data.get("web", {}).get("results", [])
            
            context_items = []
            for r in results[:max_res]:
                main_desc = r.get("description", "")
                if config.get("extra_snippets", True):
                    extra = " ".join(r.get("extra_snippets", []))
                    combined = f"{main_desc} {extra}".strip()
                else:
                    combined = main_desc
                context_items.append({
                    "title": r.get("title", ""),
                    "snippet": combined[:300],
                    "url": r.get("url", "")
                })
            
            search_time = time.perf_counter() - start
            
            context = "\n\n".join([
                f"Title: {item['title']}\nSnippet: {item['snippet']}\nSource: {item['url']}"
                for item in context_items
            ])
            
            return {
                "provider": "Brave",
                "config": config,
                "context": context,
                "num_results": len(context_items),
                "search_time": round(search_time, 3),
                "sources": [item['url'] for item in context_items]
            }
    except Exception as e:
        print(f"Brave error: {e}")
        return {
            "provider": "Brave",
            "config": config,
            "context": None,
            "num_results": 0,
            "search_time": 0,
            "error": str(e)
        }


# ============================================================================
# LLM EVALUATION WITH HYPERPARAMETERS (FIXED)
# ============================================================================
def evaluate_with_llm(evaluator_config: Dict, query: str, search_result: Dict) -> Tuple[Dict, float]:
    """Evaluate search results using specified LLM with temperature control"""
    if not search_result.get("context"):
        return None, 0
    
    eval_prompt = f"""You are evaluating search engine quality for RAG applications.

ORIGINAL QUERY: "{query}"

SEARCH PROVIDER: {search_result['provider']}
SEARCH CONFIG: {json.dumps(search_result.get('config', {}))}
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
        if evaluator_config['provider'] == "openai":
            # Handle o1 models differently (they don't support system messages or response_format)
            if evaluator_config.get('uses_reasoning', False):
                response = OPENAI_CLIENT.chat.completions.create(
                    model=evaluator_config['name'],
                    messages=[
                        {"role": "user", "content": eval_prompt + "\n\nIMPORTANT: Return ONLY valid JSON, no other text."}
                    ],
                    # o1 models don't support temperature or response_format
                )
                result_text = response.choices[0].message.content.strip()
                # Extract JSON from potential markdown
                if result_text.startswith("```"):
                    lines = result_text.split("\n")
                    result_text = "\n".join([l for l in lines if not l.strip().startswith("```")])
                result = json.loads(result_text)
            else:
                # Try with max_completion_tokens first (for newer models like GPT-5)
                # Fall back to max_tokens if that fails
                try:
                    response = OPENAI_CLIENT.chat.completions.create(
                        model=evaluator_config['name'],
                        messages=[
                            {"role": "system", "content": "You are a search quality evaluator. Return only valid JSON."},
                            {"role": "user", "content": eval_prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=evaluator_config.get('temperature', 0.1),
                        max_completion_tokens=evaluator_config.get('max_tokens', 1000)
                    )
                except Exception as param_error:
                    # If max_completion_tokens fails, try max_tokens (for older models)
                    if "max_completion_tokens" in str(param_error) or "max_tokens" in str(param_error):
                        response = OPENAI_CLIENT.chat.completions.create(
                            model=evaluator_config['name'],
                            messages=[
                                {"role": "system", "content": "You are a search quality evaluator. Return only valid JSON."},
                                {"role": "user", "content": eval_prompt}
                            ],
                            response_format={"type": "json_object"},
                            temperature=evaluator_config.get('temperature', 0.1),
                            max_tokens=evaluator_config.get('max_tokens', 1000)
                        )
                    else:
                        raise param_error
                
                result = json.loads(response.choices[0].message.content)
        
        else:  # gemini
            try:
                response = GEMINI_CLIENT.models.generate_content(
                    model=evaluator_config['name'],
                    contents=eval_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=evaluator_config.get('temperature', 0.1),
                        max_output_tokens=evaluator_config.get('max_tokens', 1000)
                    )
                )
                
                # Handle Gemini response safely
                if hasattr(response, 'text') and response.text:
                    result_text = response.text.strip()
                elif hasattr(response, 'candidates') and response.candidates and len(response.candidates) > 0:
                    if hasattr(response.candidates[0], 'content') and hasattr(response.candidates[0].content, 'parts'):
                        if len(response.candidates[0].content.parts) > 0:
                            result_text = response.candidates[0].content.parts[0].text.strip()
                        else:
                            raise ValueError("No parts in Gemini response content")
                    else:
                        raise ValueError("Invalid Gemini response structure")
                else:
                    raise ValueError("No text content in Gemini response")
                
                # Clean markdown if present
                if result_text.startswith("```"):
                    lines = result_text.split("\n")
                    result_text = "\n".join([l for l in lines if not l.strip().startswith("```") and l.strip() != "json"])
                
                result = json.loads(result_text.strip())
                
            except Exception as gemini_error:
                # Silent fail for Gemini errors to allow benchmark to continue
                return None, 0
        
        eval_time = time.perf_counter() - start
        return result, round(eval_time, 3)
        
    except Exception as e:
        # Silent fail to allow benchmark to continue with other models
        return None, 0


# ============================================================================
# GRID SEARCH CV BENCHMARK
# ============================================================================
class GridSearchCVBenchmark:
    """
    GridSearchCV-style benchmark for search providers and LLM evaluators.
    Tests all combinations of hyperparameters to find optimal configurations.
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.cv_scores = defaultdict(list)
        self.failed_evaluators = defaultdict(int)  # Track failures per model
        self.successful_evaluators = defaultdict(int)  # Track successes per model
        
    def run_grid_search(self):
        """Run exhaustive grid search across all parameter combinations"""
        print("="*70)
        print("GRID SEARCH CV BENCHMARK - SEARCH PROVIDERS × LLM EVALUATORS")
        print("="*70)
        
        total_combinations = self._count_combinations()
        print(f"\nTotal parameter combinations to test: {total_combinations}")
        print(f"Categories: {len(TEST_QUERIES)}")
        print(f"Total benchmark runs: {total_combinations * len(TEST_QUERIES)}")
        
        current_run = 0
        
        # Iterate through all queries
        for category, query in TEST_QUERIES.items():
            print(f"\n{'='*70}")
            print(f"CATEGORY: {category.upper()}")
            print(f"QUERY: {query}")
            print(f"{'='*70}")
            
            # Iterate through all search providers and their configs
            for provider_name, config_grid in SEARCH_PROVIDER_GRID.items():
                for search_config in config_grid:
                    
                    # Execute search with this configuration
                    print(f"\n--- {provider_name} | Config: {search_config} ---")
                    
                    if provider_name == "Tavily":
                        search_result = search_tavily(query, search_config)
                    else:  # Brave
                        search_result = search_brave(query, search_config)
                    
                    if not search_result.get("context"):
                        print(f"⚠️  Search failed")
                        continue
                    
                    print(f"Search time: {search_result['search_time']}s | Results: {search_result['num_results']}")
                    
                    # Evaluate with all LLM evaluators
                    for eval_config in EVALUATOR_GRID:
                        current_run += 1
                        progress = (current_run / (total_combinations * len(TEST_QUERIES))) * 100
                        
                        print(f"  [{progress:5.1f}%] Evaluating with {eval_config['label']}...", end=" ")
                        
                        eval_result, eval_time = evaluate_with_llm(eval_config, query, search_result)
                        
                        if eval_result:
                            # Store result
                            benchmark_result = BenchmarkResult(
                                search_provider=provider_name,
                                search_config=search_config,
                                evaluator=eval_config['label'],
                                evaluator_config={k: v for k, v in eval_config.items() if k != 'label'},
                                query=query,
                                category=category,
                                search_time=search_result['search_time'],
                                eval_time=eval_time,
                                num_results=search_result['num_results'],
                                scores=eval_result,
                                sources=search_result.get('sources', [])
                            )
                            
                            self.results.append(benchmark_result)
                            
                            # Track CV scores for each configuration
                            config_key = f"{provider_name}_{json.dumps(search_config, sort_keys=True)}"
                            self.cv_scores[config_key].append(eval_result.get('overall_score', 0))
                            
                            # Track success
                            self.successful_evaluators[eval_config['label']] += 1
                            
                            print(f"Score: {eval_result.get('overall_score', 0):.2f}")
                        else:
                            # Track failure
                            self.failed_evaluators[eval_config['label']] += 1
                            print("FAILED")
        
        print(f"\n{'='*70}")
        print("GRID SEARCH COMPLETE - ANALYZING RESULTS")
        print(f"{'='*70}")
        
        # Report model availability
        self._report_model_availability()
        
        self.analyze_results()
    
    def _count_combinations(self) -> int:
        """Count total parameter combinations"""
        total = 0
        for provider, configs in SEARCH_PROVIDER_GRID.items():
            total += len(configs) * len(EVALUATOR_GRID)
        return total
    
    def _report_model_availability(self):
        """Report which models worked and which failed"""
        print("\n" + "="*70)
        print("MODEL AVAILABILITY REPORT")
        print("="*70)
        
        all_evaluators = set(self.successful_evaluators.keys()) | set(self.failed_evaluators.keys())
        
        working_models = []
        failed_models = []
        
        for evaluator in sorted(all_evaluators):
            success_count = self.successful_evaluators.get(evaluator, 0)
            fail_count = self.failed_evaluators.get(evaluator, 0)
            total = success_count + fail_count
            success_rate = (success_count / total * 100) if total > 0 else 0
            
            if success_count > 0:
                working_models.append((evaluator, success_count, fail_count, success_rate))
            else:
                failed_models.append((evaluator, fail_count))
        
        if working_models:
            print(f"\n✅ WORKING MODELS ({len(working_models)}):")
            print(f"{'Model':<30} | {'Success':<8} | {'Failed':<8} | {'Success Rate':<12}")
            print("-" * 70)
            for model, success, failed, rate in sorted(working_models, key=lambda x: x[1], reverse=True):
                print(f"{model:<30} | {success:<8} | {failed:<8} | {rate:>11.1f}%")
        
        if failed_models:
            print(f"\n❌ FAILED MODELS ({len(failed_models)}):")
            print(f"{'Model':<30} | {'Attempts':<10} | {'Reason':<30}")
            print("-" * 70)
            for model, attempts in sorted(failed_models, key=lambda x: x[1], reverse=True):
                reason = "API Error / Model Not Available"
                if "GPT-5" in model or "5.2" in model:
                    reason = "Model not yet released"
                elif "Gemini 2.0 Flash" in model:
                    reason = "Model name/API issue"
                elif "Gemini 2.5" in model:
                    reason = "Model not yet released"
                print(f"{model:<30} | {attempts:<10} | {reason:<30}")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total Results Collected: {len(self.results)}")
        print(f"   Working Models: {len(working_models)}")
        print(f"   Failed Models: {len(failed_models)}")
        print(f"   Overall Success Rate: {(len(self.results) / (len(self.results) + sum(f[1] for f in failed_models)) * 100):.1f}%")
        print("="*70)
    
    def analyze_results(self):
        """Perform comprehensive analysis of grid search results"""
        
        # 1. Best configuration per search provider
        print("\n" + "="*70)
        print("BEST CONFIGURATIONS BY SEARCH PROVIDER")
        print("="*70)
        
        provider_best = {}
        for provider in SEARCH_PROVIDER_GRID.keys():
            provider_results = [r for r in self.results if r.search_provider == provider]
            
            if not provider_results:
                continue
            
            # Group by configuration
            config_scores = defaultdict(list)
            for result in provider_results:
                config_key = json.dumps(result.search_config, sort_keys=True)
                config_scores[config_key].append(result.scores.get('overall_score', 0))
            
            # Find best configuration
            best_config = max(config_scores.items(), key=lambda x: np.mean(x[1]))
            provider_best[provider] = {
                "config": json.loads(best_config[0]),
                "mean_score": np.mean(best_config[1]),
                "std_score": np.std(best_config[1]),
                "num_evals": len(best_config[1])
            }
            
            print(f"\n{provider}:")
            print(f"  Best Config: {provider_best[provider]['config']}")
            print(f"  Mean Score: {provider_best[provider]['mean_score']:.3f} ± {provider_best[provider]['std_score']:.3f}")
            print(f"  Evaluations: {provider_best[provider]['num_evals']}")
        
        # 2. Best evaluator model
        print("\n" + "="*70)
        print("EVALUATOR MODEL PERFORMANCE")
        print("="*70)
        
        evaluator_stats = defaultdict(lambda: {"scores": [], "times": []})
        for result in self.results:
            evaluator_stats[result.evaluator]["scores"].append(result.scores.get('overall_score', 0))
            evaluator_stats[result.evaluator]["times"].append(result.eval_time)
        
        print(f"\n{'Evaluator':<30} | {'Mean Score':<12} | {'Std Dev':<10} | {'Avg Time':<10} | {'N':<5}")
        print("-" * 75)
        
        for evaluator, stats in sorted(evaluator_stats.items()):
            mean_score = np.mean(stats["scores"])
            std_score = np.std(stats["scores"])
            avg_time = np.mean(stats["times"])
            n = len(stats["scores"])
            print(f"{evaluator:<30} | {mean_score:<12.3f} | {std_score:<10.3f} | {avg_time:<10.3f}s | {n:<5}")
        
        # 3. Best provider × evaluator combination
        print("\n" + "="*70)
        print("TOP 10 PROVIDER × EVALUATOR × CONFIG COMBINATIONS")
        print("="*70)
        
        combo_scores = defaultdict(list)
        for result in self.results:
            combo_key = (result.search_provider, result.evaluator, json.dumps(result.search_config, sort_keys=True))
            combo_scores[combo_key].append(result.scores.get('overall_score', 0))
        
        combo_rankings = []
        for combo, scores in combo_scores.items():
            combo_rankings.append({
                "provider": combo[0],
                "evaluator": combo[1],
                "config": json.loads(combo[2]),
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "n": len(scores)
            })
        
        combo_rankings.sort(key=lambda x: x['mean_score'], reverse=True)
        
        for i, combo in enumerate(combo_rankings[:10], 1):
            print(f"\n{i}. Score: {combo['mean_score']:.3f} ± {combo['std_score']:.3f}")
            print(f"   Provider: {combo['provider']} | Config: {combo['config']}")
            print(f"   Evaluator: {combo['evaluator']} | N={combo['n']}")
        
        # 4. Category-specific best configurations
        print("\n" + "="*70)
        print("BEST CONFIGURATION PER CATEGORY")
        print("="*70)
        
        for category in TEST_QUERIES.keys():
            cat_results = [r for r in self.results if r.category == category]
            
            if not cat_results:
                continue
            
            best_result = max(cat_results, key=lambda x: x.scores.get('overall_score', 0))
            
            print(f"\n{category.upper()}:")
            print(f"  Provider: {best_result.search_provider}")
            print(f"  Config: {best_result.search_config}")
            print(f"  Evaluator: {best_result.evaluator}")
            print(f"  Score: {best_result.scores.get('overall_score', 0):.3f}")
        
        # 5. Generate visualizations
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        self.create_visualizations()
        
        # 6. Generate final recommendation
        self.generate_final_recommendation(provider_best, combo_rankings)
    
    def create_visualizations(self):
        """Create comprehensive visualizations of grid search results"""
        
        # Convert results to DataFrame
        df_data = []
        for result in self.results:
            row = {
                'provider': result.search_provider,
                'evaluator': result.evaluator,
                'category': result.category,
                'search_time': result.search_time,
                'eval_time': result.eval_time,
                'num_results': result.num_results,
                'overall_score': result.scores.get('overall_score', 0),
                'relevance_score': result.scores.get('relevance_score', 0),
                'freshness_score': result.scores.get('freshness_score', 0),
                'quality_score': result.scores.get('quality_score', 0),
                'usefulness_score': result.scores.get('usefulness_score', 0),
                'coverage_score': result.scores.get('coverage_score', 0),
                'config': json.dumps(result.search_config, sort_keys=True)
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save DataFrame
        df.to_csv('gridsearch_results.csv', index=False)
        print(f"✅ Results DataFrame saved to: gridsearch_results.csv")
        
        # Create visualization dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Overall Score by Provider',
                'Overall Score by Evaluator',
                'Score Distribution by Category',
                'Search Time vs Eval Time',
                'Score Components Breakdown',
                'Provider × Evaluator Heatmap'
            ),
            specs=[
                [{'type': 'box'}, {'type': 'box'}],
                [{'type': 'violin'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'heatmap'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        # 1. Box plot: Overall Score by Provider
        for provider in df['provider'].unique():
            provider_data = df[df['provider'] == provider]['overall_score']
            fig.add_trace(
                go.Box(y=provider_data, name=provider, showlegend=False),
                row=1, col=1
            )
        
        # 2. Box plot: Overall Score by Evaluator
        for evaluator in df['evaluator'].unique():
            eval_data = df[df['evaluator'] == evaluator]['overall_score']
            fig.add_trace(
                go.Box(y=eval_data, name=evaluator[:15], showlegend=False),
                row=1, col=2
            )
        
        # 3. Violin plot: Score Distribution by Category
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]['overall_score']
            fig.add_trace(
                go.Violin(y=cat_data, name=category, showlegend=False),
                row=2, col=1
            )
        
        # 4. Scatter: Search Time vs Eval Time
        fig.add_trace(
            go.Scatter(
                x=df['search_time'],
                y=df['eval_time'],
                mode='markers',
                marker=dict(
                    size=df['overall_score'] * 2,
                    color=df['overall_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(x=1.15, len=0.3, y=0.5)
                ),
                text=df['provider'] + ' - ' + df['evaluator'],
                hovertemplate='<b>%{text}</b><br>Search: %{x:.3f}s<br>Eval: %{y:.3f}s<br>Score: %{marker.color:.2f}<extra></extra>',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 5. Bar chart: Score Components Breakdown
        score_components = ['relevance_score', 'freshness_score', 'quality_score', 
                          'usefulness_score', 'coverage_score']
        avg_scores = [df[comp].mean() for comp in score_components]
        component_names = ['Relevance', 'Freshness', 'Quality', 'Usefulness', 'Coverage']
        
        fig.add_trace(
            go.Bar(
                x=component_names,
                y=avg_scores,
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
                showlegend=False
            ),
            row=3, col=1
        )
        
        # 6. Heatmap: Provider × Evaluator
        pivot_table = df.pivot_table(
            values='overall_score',
            index='provider',
            columns='evaluator',
            aggfunc='mean'
        )
        
        fig.add_trace(
            go.Heatmap(
                z=pivot_table.values,
                x=[col[:15] for col in pivot_table.columns],
                y=pivot_table.index,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(x=1.15, len=0.3, y=0.17),
                hovertemplate='Provider: %{y}<br>Evaluator: %{x}<br>Score: %{z:.2f}<extra></extra>'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Grid Search CV - Comprehensive Results Dashboard",
            title_font_size=20,
            height=1400,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Provider", row=1, col=1)
        fig.update_xaxes(title_text="Evaluator", row=1, col=2, tickangle=-45)
        fig.update_xaxes(title_text="Category", row=2, col=1)
        fig.update_xaxes(title_text="Search Time (s)", row=2, col=2)
        fig.update_xaxes(title_text="Score Component", row=3, col=1)
        fig.update_xaxes(title_text="Evaluator", row=3, col=2, tickangle=-45)
        
        fig.update_yaxes(title_text="Overall Score", row=1, col=1)
        fig.update_yaxes(title_text="Overall Score", row=1, col=2)
        fig.update_yaxes(title_text="Overall Score", row=2, col=1)
        fig.update_yaxes(title_text="Eval Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Average Score", row=3, col=1)
        fig.update_yaxes(title_text="Provider", row=3, col=2)
        
        fig.write_html('gridsearch_dashboard.html')
        print(f"✅ Interactive dashboard saved to: gridsearch_dashboard.html")
        
        # Create individual detailed visualizations
        self._create_detailed_visualizations(df)
    
    def _create_detailed_visualizations(self, df):
        """Create additional detailed visualizations"""
        
        # 1. Provider Performance Comparison
        fig1 = go.Figure()
        
        for provider in df['provider'].unique():
            provider_df = df[df['provider'] == provider]
            
            fig1.add_trace(go.Box(
                y=provider_df['overall_score'],
                name=provider,
                boxmean='sd'
            ))
        
        fig1.update_layout(
            title="Search Provider Performance Comparison",
            yaxis_title="Overall Score",
            xaxis_title="Search Provider",
            height=500
        )
        fig1.write_html('provider_comparison.html')
        print(f"✅ Provider comparison saved to: provider_comparison.html")
        
        # 2. Evaluator Performance Ranking
        evaluator_stats = df.groupby('evaluator').agg({
            'overall_score': ['mean', 'std', 'count'],
            'eval_time': 'mean'
        }).round(3)
        
        evaluator_stats.columns = ['_'.join(col).strip() for col in evaluator_stats.columns.values]
        evaluator_stats = evaluator_stats.sort_values('overall_score_mean', ascending=False)
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            x=evaluator_stats.index,
            y=evaluator_stats['overall_score_mean'],
            error_y=dict(type='data', array=evaluator_stats['overall_score_std']),
            marker_color='lightblue',
            name='Mean Score'
        ))
        
        fig2.update_layout(
            title="Evaluator Model Performance Ranking",
            xaxis_title="Evaluator Model",
            yaxis_title="Mean Overall Score (± std)",
            xaxis_tickangle=-45,
            height=600
        )
        fig2.write_html('evaluator_ranking.html')
        print(f"✅ Evaluator ranking saved to: evaluator_ranking.html")
        
        # 3. Category Performance Breakdown
        category_scores = df.groupby('category')[
            ['relevance_score', 'freshness_score', 'quality_score', 
             'usefulness_score', 'coverage_score']
        ].mean()
        
        fig3 = go.Figure()
        
        score_types = ['relevance_score', 'freshness_score', 'quality_score', 
                      'usefulness_score', 'coverage_score']
        score_names = ['Relevance', 'Freshness', 'Quality', 'Usefulness', 'Coverage']
        
        for i, (score_type, score_name) in enumerate(zip(score_types, score_names)):
            fig3.add_trace(go.Bar(
                name=score_name,
                x=category_scores.index,
                y=category_scores[score_type]
            ))
        
        fig3.update_layout(
            title="Score Components by Category",
            xaxis_title="Category",
            yaxis_title="Score",
            barmode='group',
            height=600
        )
        fig3.write_html('category_breakdown.html')
        print(f"✅ Category breakdown saved to: category_breakdown.html")
        
        # 4. Time Performance Analysis
        time_analysis = df.groupby('provider').agg({
            'search_time': ['mean', 'std'],
            'eval_time': ['mean', 'std']
        }).round(3)
        
        fig4 = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Search Time by Provider', 'Evaluation Time by Provider')
        )
        
        providers = time_analysis.index
        
        fig4.add_trace(
            go.Bar(
                x=providers,
                y=time_analysis[('search_time', 'mean')],
                error_y=dict(type='data', array=time_analysis[('search_time', 'std')]),
                marker_color='coral',
                name='Search Time'
            ),
            row=1, col=1
        )
        
        fig4.add_trace(
            go.Bar(
                x=providers,
                y=time_analysis[('eval_time', 'mean')],
                error_y=dict(type='data', array=time_analysis[('eval_time', 'std')]),
                marker_color='lightgreen',
                name='Eval Time'
            ),
            row=1, col=2
        )
        
        fig4.update_xaxes(title_text="Provider", row=1, col=1)
        fig4.update_xaxes(title_text="Provider", row=1, col=2)
        fig4.update_yaxes(title_text="Time (seconds)", row=1, col=1)
        fig4.update_yaxes(title_text="Time (seconds)", row=1, col=2)
        
        fig4.update_layout(
            title_text="Performance Time Analysis",
            showlegend=False,
            height=500
        )
        fig4.write_html('time_analysis.html')
        print(f"✅ Time analysis saved to: time_analysis.html")
        
        # 5. Configuration Impact Analysis
        config_impact = df.groupby(['provider', 'config']).agg({
            'overall_score': ['mean', 'std', 'count']
        }).round(3)
        
        config_impact.columns = ['_'.join(col).strip() for col in config_impact.columns.values]
        config_impact = config_impact.sort_values('overall_score_mean', ascending=False)
        
        fig5 = go.Figure()
        
        config_labels = [f"{prov} - {conf[:30]}..." for prov, conf in config_impact.index]
        
        fig5.add_trace(go.Bar(
            y=config_labels[:15],  # Top 15 configurations
            x=config_impact['overall_score_mean'][:15],
            orientation='h',
            error_x=dict(type='data', array=config_impact['overall_score_std'][:15]),
            marker_color='mediumpurple'
        ))
        
        fig5.update_layout(
            title="Top 15 Configuration Performance",
            xaxis_title="Mean Overall Score",
            yaxis_title="Provider - Configuration",
            height=700
        )
        fig5.write_html('config_impact.html')
        print(f"✅ Configuration impact saved to: config_impact.html")
        
        # 6. Correlation Heatmap
        correlation_cols = ['overall_score', 'relevance_score', 'freshness_score', 
                          'quality_score', 'usefulness_score', 'coverage_score',
                          'search_time', 'eval_time', 'num_results']
        
        corr_matrix = df[correlation_cols].corr()
        
        fig6 = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig6.update_layout(
            title="Feature Correlation Matrix",
            height=700,
            xaxis_tickangle=-45
        )
        fig6.write_html('correlation_heatmap.html')
        print(f"✅ Correlation heatmap saved to: correlation_heatmap.html")
        
        print(f"\n{'='*70}")
        print("📊 ALL VISUALIZATIONS CREATED SUCCESSFULLY")
        print("="*70)
        print("Files generated:")
        print("  1. gridsearch_results.csv - Raw results data")
        print("  2. gridsearch_dashboard.html - Main interactive dashboard")
        print("  3. provider_comparison.html - Provider performance")
        print("  4. evaluator_ranking.html - Evaluator model ranking")
        print("  5. category_breakdown.html - Category-wise score breakdown")
        print("  6. time_analysis.html - Performance timing analysis")
        print("  7. config_impact.html - Configuration impact ranking")
        print("  8. correlation_heatmap.html - Feature correlations")
        print("="*70)
    
    def generate_final_recommendation(self, provider_best, combo_rankings):
        """Generate AI-powered final recommendation"""
        print(f"\n\n{'#'*70}")
        print("GENERATING FINAL AI RECOMMENDATION")
        print(f"{'#'*70}")
        
        summary_data = {
            "total_runs": len(self.results),
            "provider_best_configs": provider_best,
            "top_combinations": combo_rankings[:5],
            "detailed_results_sample": [asdict(r) for r in self.results[:10]]
        }
        
        recommendation_prompt = f"""You are an expert ML engineer analyzing GridSearchCV results for RAG search providers.

GRID SEARCH RESULTS:
{json.dumps(summary_data, indent=2)}

Based on this exhaustive grid search across multiple search providers, configurations, and LLM evaluators, provide:

1. BEST OVERALL CONFIGURATION (provider + hyperparameters)
2. BEST EVALUATOR MODEL (for assessing search quality)
3. PRODUCTION RECOMMENDATIONS (speed vs quality tradeoffs)
4. CATEGORY-SPECIFIC RECOMMENDATIONS
5. HYPERPARAMETER INSIGHTS (which parameters matter most)

RETURN STRICT JSON:
{{
  "best_overall": {{
    "provider": "...",
    "config": {{}},
    "evaluator": "...",
    "mean_score": 0.0,
    "reasoning": "..."
  }},
  "best_evaluator": {{
    "model": "...",
    "reasoning": "..."
  }},
  "production_recommendations": {{
    "high_quality": {{"provider": "...", "config": {{}}}},
    "balanced": {{"provider": "...", "config": {{}}}},
    "fast": {{"provider": "...", "config": {{}}}}
  }},
  "hyperparameter_insights": {{
    "max_results": "...",
    "search_depth": "...",
    "temperature": "...",
    "key_finding": "..."
  }},
  "category_recommendations": {{}},
  "final_summary": "Comprehensive recommendation based on grid search"
}}
"""
        
        try:
            judge_response = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert ML engineer specializing in hyperparameter optimization."},
                    {"role": "user", "content": recommendation_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            recommendation = json.loads(judge_response.choices[0].message.content)
            
            # Display recommendation
            print(f"\n{'='*70}")
            print("🏆 FINAL GRID SEARCH RECOMMENDATION")
            print(f"{'='*70}")
            
            print(f"\n🥇 BEST OVERALL CONFIGURATION:")
            print(f"   Provider: {recommendation['best_overall']['provider']}")
            print(f"   Config: {recommendation['best_overall']['config']}")
            print(f"   Evaluator: {recommendation['best_overall']['evaluator']}")
            print(f"   Mean Score: {recommendation['best_overall']['mean_score']}")
            print(f"   Reasoning: {recommendation['best_overall']['reasoning']}")
            
            print(f"\n🤖 BEST EVALUATOR MODEL:")
            print(f"   Model: {recommendation['best_evaluator']['model']}")
            print(f"   Reasoning: {recommendation['best_evaluator']['reasoning']}")
            
            print(f"\n🚀 PRODUCTION RECOMMENDATIONS:")
            for tier, config in recommendation['production_recommendations'].items():
                print(f"   {tier.upper()}: {config['provider']} - {config['config']}")
            
            print(f"\n🔬 HYPERPARAMETER INSIGHTS:")
            for param, insight in recommendation['hyperparameter_insights'].items():
                print(f"   {param}: {insight}")
            
            print(f"\n📝 FINAL SUMMARY:")
            print(f"   {recommendation['final_summary']}")
            
            # Save complete results
            output_data = {
                "metadata": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_runs": len(self.results),
                    "providers": list(SEARCH_PROVIDER_GRID.keys()),
                    "evaluators": [e['label'] for e in EVALUATOR_GRID],
                    "categories": list(TEST_QUERIES.keys())
                },
                "grid_search_results": [asdict(r) for r in self.results],
                "analysis": {
                    "provider_best_configs": provider_best,
                    "top_combinations": combo_rankings[:10]
                },
                "ai_recommendation": recommendation
            }
            
            output_file = 'gridsearch_cv_results.json'
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\n\n📁 Complete GridSearchCV results saved to: {output_file}")
            print("="*70)
            
            return output_data
            
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    benchmark = GridSearchCVBenchmark()
    results = benchmark.run_grid_search()