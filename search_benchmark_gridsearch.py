import os
import time
import json
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
from tavily import TavilyClient
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from itertools import product
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

load_dotenv()

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================
@dataclass
class SearchParams:
    """Parameters for search provider configuration"""
    max_results: int
    search_depth: str = "basic"  # basic, advanced
    topic: str = "general"  # general, news
    
    def to_dict(self):
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Structured result for each benchmark run"""
    category: str
    query: str
    provider: str
    params: Dict[str, Any]
    search_time: float
    num_results: int
    eval_model: str
    relevance_score: float
    freshness_score: float
    quality_score: float
    usefulness_score: float
    coverage_score: float
    overall_score: float
    sources: List[str]
    
    def to_dict(self):
        result = asdict(self)
        result['params'] = self.params
        return result


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
# GLOBAL CONFIGURATION
# ============================================================================
OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GEMINI_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
TAVILY_CLIENT = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

# Test queries across different domains
TEST_QUERIES = {
    "tech": "breakthrough AI and technology developments 2026"
}

# GridSearch parameter grid
PARAM_GRID = {
    "Tavily": {
        "max_results": [5, 7],
        "search_depth": ["basic", "advanced"],
        "topic": ["general"]
    },
    "Brave": {
        "max_results": [5, 7]
    }
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
    {"name": "gemini-2.5-flash", "provider": "gemini", "label": "Gemini 2.5 Flash"},
    {"name": "gemini-2.5-pro", "provider": "gemini", "label": "Gemini 2.5 Pro"},
]

# ============================================================================
# SEARCH PROVIDER FUNCTIONS WITH PARAMETERS
# ============================================================================
def search_tavily(query: str, params: SearchParams):
    """Search using Tavily with configurable parameters"""
    start = time.perf_counter()
    try:
        response = TAVILY_CLIENT.search(
            query=query,
            topic=params.topic,
            search_depth=params.search_depth,
            max_results=params.max_results
        )
        results = response.get("results", [])
        context_items = [
            {
                "title": r.get("title", ""),
                "snippet": r.get("content", "")[:300],
                "url": r.get("url", ""),
                "score": r.get("score", 0)
            }
            for r in results[:params.max_results]
        ]
        search_time = time.perf_counter() - start
        
        context = "\n\n".join([
            f"Title: {item['title']}\nSnippet: {item['snippet']}\nSource: {item['url']}"
            for item in context_items
        ])
        
        return {
            "provider": "Tavily",
            "context": context,
            "num_results": len(context_items),
            "search_time": round(search_time, 3),
            "sources": [item['url'] for item in context_items],
            "params": params.to_dict()
        }
    except Exception as e:
        print(f"Tavily error: {e}")
        return None


def search_brave(query: str, params: SearchParams):
    """Search using Brave with configurable parameters"""
    start = time.perf_counter()
    try:
        with BraveSearch(BRAVE_API_KEY) as brave:
            data = brave.search(query, count=params.max_results)
            results = data.get("web", {}).get("results", [])
            
            context_items = []
            for r in results[:params.max_results]:
                main_desc = r.get("description", "")
                extra = " ".join(r.get("extra_snippets", []))
                combined = f"{main_desc} {extra}".strip()
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
                "context": context,
                "num_results": len(context_items),
                "search_time": round(search_time, 3),
                "sources": [item['url'] for item in context_items],
                "params": params.to_dict()
            }
    except Exception as e:
        print(f"Brave error: {e}")
        return None


# ============================================================================
# LLM EVALUATION FUNCTIONS
# ============================================================================
def evaluate_with_llm(model_config, query, search_result):
    """Evaluate search results using specified LLM"""
    if not search_result or not search_result.get("context"):
        return None
    
    eval_prompt = f"""You are evaluating search engine quality for RAG applications.

ORIGINAL QUERY: "{query}"

SEARCH PROVIDER: {search_result['provider']}
PARAMETERS: {json.dumps(search_result['params'], indent=2)}
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
        
        return result
        
    except Exception as e:
        print(f"Evaluation error with {model_config['label']}: {e}")
        return None


# ============================================================================
# GRIDSEARCH BENCHMARK CLASS
# ============================================================================
class GridSearchBenchmark:
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.df = None
    
    def generate_param_combinations(self, provider: str) -> List[SearchParams]:
        """Generate all parameter combinations for a provider (GridSearchCV style)"""
        if provider not in PARAM_GRID:
            return [SearchParams(max_results=5)]
        
        grid = PARAM_GRID[provider]
        param_names = list(grid.keys())
        param_values = [grid[name] for name in param_names]
        
        combinations = []
        for combo in product(*param_values):
            param_dict = dict(zip(param_names, combo))
            # Ensure all required fields are present
            if 'search_depth' not in param_dict:
                param_dict['search_depth'] = 'basic'
            if 'topic' not in param_dict:
                param_dict['topic'] = 'general'
            combinations.append(SearchParams(**param_dict))
        
        return combinations
    
    def run_gridsearch(self):
        """Run complete GridSearch benchmark"""
        print("="*80)
        print("🔍 GRIDSEARCH BENCHMARK FOR SEARCH PROVIDERS")
        print("="*80)
        print(f"Queries: {len(TEST_QUERIES)}")
        print(f"Providers: Tavily, Brave")
        print(f"Evaluators: {', '.join([m['label'] for m in EVAL_MODELS])}")
        print(f"Parameter Grid:")
        for provider, params in PARAM_GRID.items():
            print(f"  {provider}: {params}")
        print("="*80)
        
        total_runs = 0
        for category, query in TEST_QUERIES.items():
            for provider in ["Tavily", "Brave"]:
                param_combinations = self.generate_param_combinations(provider)
                total_runs += len(param_combinations) * len(EVAL_MODELS)
        
        print(f"\nTotal benchmark runs: {total_runs}")
        print(f"\nStarting benchmark...\n")
        
        run_count = 0
        
        for category, query in TEST_QUERIES.items():
            print(f"\n{'='*80}")
            print(f"📂 CATEGORY: {category.upper()}")
            print(f"❓ QUERY: {query}")
            print(f"{'='*80}")
            
            # Test Tavily with all parameter combinations
            print(f"\n🔎 Testing Tavily...")
            tavily_params = self.generate_param_combinations("Tavily")
            for params in tavily_params:
                run_count += 1
                print(f"\n  Run {run_count}/{total_runs} - Params: {params.to_dict()}")
                
                search_result = search_tavily(query, params)
                if not search_result:
                    continue
                
                print(f"  ⏱️  Search time: {search_result['search_time']}s")
                print(f"  📊 Results: {search_result['num_results']}")
                
                # Evaluate with each model
                for model in EVAL_MODELS:
                    print(f"    🤖 Evaluating with {model['label']}...", end=" ")
                    eval_result = evaluate_with_llm(model, query, search_result)
                    
                    if eval_result:
                        benchmark_result = BenchmarkResult(
                            category=category,
                            query=query,
                            provider="Tavily",
                            params=search_result['params'],
                            search_time=search_result['search_time'],
                            num_results=search_result['num_results'],
                            eval_model=model['label'],
                            relevance_score=eval_result['relevance_score'],
                            freshness_score=eval_result['freshness_score'],
                            quality_score=eval_result['quality_score'],
                            usefulness_score=eval_result['usefulness_score'],
                            coverage_score=eval_result['coverage_score'],
                            overall_score=eval_result['overall_score'],
                            sources=search_result['sources']
                        )
                        self.results.append(benchmark_result)
                        print(f"✅ Score: {eval_result['overall_score']}")
                    else:
                        print("❌ Failed")
            
            # Test Brave with all parameter combinations
            print(f"\n🦁 Testing Brave...")
            brave_params = self.generate_param_combinations("Brave")
            for params in brave_params:
                run_count += 1
                print(f"\n  Run {run_count}/{total_runs} - Params: {params.to_dict()}")
                
                search_result = search_brave(query, params)
                if not search_result:
                    continue
                
                print(f"  ⏱️  Search time: {search_result['search_time']}s")
                print(f"  📊 Results: {search_result['num_results']}")
                
                # Evaluate with each model
                for model in EVAL_MODELS:
                    print(f"    🤖 Evaluating with {model['label']}...", end=" ")
                    eval_result = evaluate_with_llm(model, query, search_result)
                    
                    if eval_result:
                        benchmark_result = BenchmarkResult(
                            category=category,
                            query=query,
                            provider="Brave",
                            params=search_result['params'],
                            search_time=search_result['search_time'],
                            num_results=search_result['num_results'],
                            eval_model=model['label'],
                            relevance_score=eval_result['relevance_score'],
                            freshness_score=eval_result['freshness_score'],
                            quality_score=eval_result['quality_score'],
                            usefulness_score=eval_result['usefulness_score'],
                            coverage_score=eval_result['coverage_score'],
                            overall_score=eval_result['overall_score'],
                            sources=search_result['sources']
                        )
                        self.results.append(benchmark_result)
                        print(f"✅ Score: {eval_result['overall_score']}")
                    else:
                        print("❌ Failed")
        
        # Convert to DataFrame
        self.df = pd.DataFrame([r.to_dict() for r in self.results])
        
        print(f"\n\n{'='*80}")
        print(f"✅ Benchmark complete! Total results: {len(self.results)}")
        print(f"{'='*80}")
        
        return self.df
    
    def create_visualizations(self):
        """Create comprehensive Plotly visualizations"""
        if self.df is None or len(self.df) == 0:
            print("No data to visualize!")
            return
        
        print("\n📊 Generating visualizations...")
        
        # Create output directory
        os.makedirs('visualizations', exist_ok=True)
        
        # 1. Overall Performance Heatmap
        fig1 = self.create_performance_heatmap()
        fig1.write_html('visualizations/1_performance_heatmap.html')
        
        # 2. Provider Comparison Box Plots
        fig2 = self.create_provider_comparison()
        fig2.write_html('visualizations/2_provider_comparison.html')
        
        # 3. Parameter Impact Analysis
        fig3 = self.create_parameter_impact()
        fig3.write_html('visualizations/3_parameter_impact.html')
        
        # 4. Category Performance Radar
        fig4 = self.create_category_radar()
        fig4.write_html('visualizations/4_category_radar.html')
        
        # 5. Speed vs Quality Scatter
        fig5 = self.create_speed_quality_scatter()
        fig5.write_html('visualizations/5_speed_vs_quality.html')
        
        # 6. Time Series Performance
        fig6 = self.create_score_distribution()
        fig6.write_html('visualizations/6_score_distribution.html')
        
        # 7. Best Configuration Finder
        fig7 = self.create_best_config_table()
        fig7.write_html('visualizations/7_best_configurations.html')
        
        print("✅ All visualizations saved to 'visualizations/' directory")
        
        return {
            "heatmap": fig1,
            "comparison": fig2,
            "parameter_impact": fig3,
            "radar": fig4,
            "scatter": fig5,
            "distribution": fig6,
            "best_config": fig7
        }
    
    def create_performance_heatmap(self):
        """Create heatmap of average scores by provider and category"""
        pivot_data = self.df.groupby(['provider', 'category'])['overall_score'].mean().unstack()
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn',
            text=np.round(pivot_data.values, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Overall Score")
        ))
        
        fig.update_layout(
            title="Average Overall Score by Provider and Category",
            xaxis_title="Category",
            yaxis_title="Provider",
            height=400,
            font=dict(size=12)
        )
        
        return fig
    
    def create_provider_comparison(self):
        """Create box plots comparing providers across metrics"""
        metrics = ['relevance_score', 'freshness_score', 'quality_score', 
                   'usefulness_score', 'coverage_score', 'overall_score']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=metrics,
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        for idx, metric in enumerate(metrics):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            for provider in self.df['provider'].unique():
                provider_data = self.df[self.df['provider'] == provider][metric]
                fig.add_trace(
                    go.Box(y=provider_data, name=provider, showlegend=(idx == 0)),
                    row=row, col=col
                )
        
        fig.update_layout(
            title_text="Provider Performance Comparison Across All Metrics",
            height=700,
            showlegend=True
        )
        
        return fig
    
    def create_parameter_impact(self):
        """Analyze impact of different parameters on performance"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Impact of max_results", "Impact of search_depth (Tavily)"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Impact of max_results
        # Extract max_results from params first
        df_copy = self.df.copy()
        df_copy['max_results'] = df_copy['params'].apply(
            lambda x: x.get('max_results', 5) if isinstance(x, dict) else 5
        )
        
        max_results_impact = df_copy.groupby(['provider', 'max_results'])['overall_score'].mean().reset_index()
        
        for provider in df_copy['provider'].unique():
            provider_data = max_results_impact[max_results_impact['provider'] == provider]
            fig.add_trace(
                go.Bar(x=provider_data['max_results'], y=provider_data['overall_score'], name=provider),
                row=1, col=1
            )
        
        # Impact of search_depth (Tavily only)
        tavily_df = self.df[self.df['provider'] == 'Tavily'].copy()
        tavily_df['search_depth'] = tavily_df['params'].apply(
            lambda x: x.get('search_depth', 'basic') if isinstance(x, dict) else 'basic'
        )
        depth_impact = tavily_df.groupby('search_depth')['overall_score'].mean()
        
        fig.add_trace(
            go.Bar(x=depth_impact.index, y=depth_impact.values, name='Tavily', showlegend=False),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Parameter Impact on Overall Score",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_category_radar(self):
        """Create radar chart for category performance"""
        categories = self.df['category'].unique()
        providers = self.df['provider'].unique()
        
        fig = go.Figure()
        
        for provider in providers:
            provider_data = self.df[self.df['provider'] == provider]
            scores = []
            for category in categories:
                cat_score = provider_data[provider_data['category'] == category]['overall_score'].mean()
                scores.append(cat_score)
            
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=categories,
                fill='toself',
                name=provider
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            title="Provider Performance by Category (Radar Chart)",
            height=500
        )
        
        return fig
    
    def create_speed_quality_scatter(self):
        """Create scatter plot of speed vs quality"""
        fig = px.scatter(
            self.df,
            x='search_time',
            y='overall_score',
            color='provider',
            size='num_results',
            hover_data=['category', 'eval_model'],
            title='Search Speed vs Quality Trade-off',
            labels={
                'search_time': 'Search Time (seconds)',
                'overall_score': 'Overall Quality Score',
                'num_results': 'Number of Results'
            }
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_score_distribution(self):
        """Create distribution plots for all scores"""
        metrics = ['relevance_score', 'freshness_score', 'quality_score', 
                   'usefulness_score', 'coverage_score']
        
        fig = go.Figure()
        
        for metric in metrics:
            fig.add_trace(go.Box(
                y=self.df[metric],
                name=metric.replace('_score', '').title(),
                boxmean='sd'
            ))
        
        fig.update_layout(
            title="Score Distribution Across All Metrics",
            yaxis_title="Score (0-10)",
            height=500
        )
        
        return fig
    
    def create_best_config_table(self):
        """Create table showing best configurations for each category"""
        best_configs = []
        
        for category in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == category]
            best_idx = cat_data['overall_score'].idxmax()
            best_row = cat_data.loc[best_idx]
            
            best_configs.append({
                'Category': category,
                'Provider': best_row['provider'],
                'Parameters': json.dumps(best_row['params'], indent=2),
                'Overall Score': round(best_row['overall_score'], 2),
                'Search Time': round(best_row['search_time'], 3),
                'Eval Model': best_row['eval_model']
            })
        
        df_best = pd.DataFrame(best_configs)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df_best.columns),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[df_best[col] for col in df_best.columns],
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title="Best Configuration for Each Category",
            height=400
        )
        
        return fig
    
    def generate_summary_report(self):
        """Generate comprehensive summary statistics"""
        if self.df is None or len(self.df) == 0:
            return None
        
        summary = {
            "overall_stats": {
                "total_runs": len(self.df),
                "providers_tested": self.df['provider'].nunique(),
                "categories_tested": self.df['category'].nunique(),
                "eval_models_used": self.df['eval_model'].nunique()
            },
            "provider_rankings": {},
            "category_winners": {},
            "best_parameters": {},
            "speed_analysis": {},
            "quality_analysis": {}
        }
        
        # Provider rankings
        for provider in self.df['provider'].unique():
            provider_data = self.df[self.df['provider'] == provider]
            summary['provider_rankings'][provider] = {
                'avg_overall_score': round(provider_data['overall_score'].mean(), 2),
                'avg_search_time': round(provider_data['search_time'].mean(), 3),
                'avg_relevance': round(provider_data['relevance_score'].mean(), 2),
                'avg_freshness': round(provider_data['freshness_score'].mean(), 2),
                'avg_quality': round(provider_data['quality_score'].mean(), 2),
                'std_dev': round(provider_data['overall_score'].std(), 2)
            }
        
        # Category winners
        for category in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == category]
            best_idx = cat_data['overall_score'].idxmax()
            best = cat_data.loc[best_idx]
            summary['category_winners'][category] = {
                'provider': best['provider'],
                'score': round(best['overall_score'], 2),
                'params': best['params']
            }
        
        # Best parameters analysis
        for provider in self.df['provider'].unique():
            provider_data = self.df[self.df['provider'] == provider]
            best_idx = provider_data['overall_score'].idxmax()
            best = provider_data.loc[best_idx]
            summary['best_parameters'][provider] = best['params']
        
        # Speed analysis
        summary['speed_analysis'] = {
            'fastest_provider': self.df.groupby('provider')['search_time'].mean().idxmin(),
            'avg_times': self.df.groupby('provider')['search_time'].mean().to_dict()
        }
        
        # Quality analysis
        summary['quality_analysis'] = {
            'highest_quality_provider': self.df.groupby('provider')['overall_score'].mean().idxmax(),
            'avg_scores': self.df.groupby('provider')['overall_score'].mean().to_dict()
        }
        
        return summary
    
    def save_results(self):
        """Save all results and visualizations"""
        # Save DataFrame
        self.df.to_csv('search_benchmark_results.csv', index=False)
        self.df.to_json('search_benchmark_results.json', orient='records', indent=2)
        
        # Save summary
        summary = self.generate_summary_report()
        with open('benchmark_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create visualizations
        self.create_visualizations()
        
        print("\n📁 Results saved:")
        print("  - search_benchmark_results.csv")
        print("  - search_benchmark_results.json")
        print("  - benchmark_summary.json")
        print("  - visualizations/ (7 HTML files)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    benchmark = GridSearchBenchmark()
    
    # Run GridSearch
    df = benchmark.run_gridsearch()
    
    # Save results and create visualizations
    benchmark.save_results()
    
    # Print summary
    print("\n" + "="*80)
    print("📊 BENCHMARK SUMMARY")
    print("="*80)
    
    summary = benchmark.generate_summary_report()
    print(json.dumps(summary, indent=2))
    
    print("\n✅ Benchmark complete! Check the visualizations/ directory for charts.")