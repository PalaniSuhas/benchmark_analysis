"""
Quick Start Example - Search Provider Benchmark
================================================

This script demonstrates how to:
1. Run a minimal benchmark with reduced parameter space
2. Generate visualizations
3. Analyze results programmatically
"""

import os
import sys
from search_benchmark_gridsearch import GridSearchBenchmark, PARAM_GRID, TEST_QUERIES, EVAL_MODELS
from visualizer import BenchmarkVisualizer
import pandas as pd

# ============================================================================
# QUICK CONFIGURATION
# ============================================================================

# Override with minimal configuration for quick testing
QUICK_PARAM_GRID = {
    "Tavily": {
        "max_results": [5],  # Single value instead of [3, 5, 7]
        "search_depth": ["basic"],  # Only basic
        "topic": ["news"]  # Only news
    },
    "Brave": {
        "max_results": [5]  # Single value
    }
}

# Test only 2 categories instead of all 5
QUICK_QUERIES = {
    "tech": "breakthrough AI and technology developments 2026",
    "business": "Tesla vs BYD electric vehicle market performance 2025 2026"
}

# Use only fast, cheap models
QUICK_MODELS = [
    {"name": "gpt-4o-mini", "provider": "openai", "label": "GPT-4o Mini"},
]


def run_quick_benchmark():
    """Run a quick benchmark with reduced parameters"""
    
    print("="*80)
    print("🚀 QUICK START BENCHMARK")
    print("="*80)
    print("\nThis is a minimal benchmark for testing. For full evaluation, run:")
    print("  python search_benchmark_gridsearch.py")
    print("\n" + "="*80)
    
    # Temporarily override global configs
    import search_benchmark_gridsearch as sbg
    
    original_param_grid = sbg.PARAM_GRID.copy()
    original_queries = sbg.TEST_QUERIES.copy()
    original_models = sbg.EVAL_MODELS.copy()
    
    sbg.PARAM_GRID = QUICK_PARAM_GRID
    sbg.TEST_QUERIES = QUICK_QUERIES
    sbg.EVAL_MODELS = QUICK_MODELS
    
    try:
        # Run benchmark
        benchmark = GridSearchBenchmark()
        df = benchmark.run_gridsearch()
        
        # Save results
        benchmark.save_results()
        
        print("\n" + "="*80)
        print("📊 QUICK ANALYSIS")
        print("="*80)
        
        # Print quick insights
        if len(df) > 0:
            print("\n🏆 Quick Results:")
            
            # Best provider overall
            best_provider = df.groupby('provider')['overall_score'].mean().idxmax()
            best_score = df.groupby('provider')['overall_score'].mean().max()
            print(f"  Best Overall: {best_provider} (Score: {best_score:.2f})")
            
            # Speed comparison
            speed_stats = df.groupby('provider')['search_time'].agg(['mean', 'std'])
            print(f"\n⚡ Speed Comparison:")
            for provider, row in speed_stats.iterrows():
                print(f"  {provider}: {row['mean']:.3f}s ± {row['std']:.3f}s")
            
            # Score comparison
            print(f"\n⭐ Score Comparison:")
            score_stats = df.groupby('provider')['overall_score'].agg(['mean', 'min', 'max'])
            for provider, row in score_stats.iterrows():
                print(f"  {provider}: {row['mean']:.2f} (range: {row['min']:.2f}-{row['max']:.2f})")
            
            # Category breakdown
            print(f"\n📂 By Category:")
            cat_pivot = df.pivot_table(
                values='overall_score',
                index='provider',
                columns='category',
                aggfunc='mean'
            )
            print(cat_pivot.round(2))
            
        print("\n✅ Quick benchmark complete!")
        print(f"📁 Results saved to: search_benchmark_results.csv")
        print(f"📊 Visualizations saved to: visualizations/")
        
    finally:
        # Restore original configs
        sbg.PARAM_GRID = original_param_grid
        sbg.TEST_QUERIES = original_queries
        sbg.EVAL_MODELS = original_models


def analyze_existing_results(csv_path='search_benchmark_results.csv'):
    """Analyze existing benchmark results"""
    
    if not os.path.exists(csv_path):
        print(f"❌ No results found at {csv_path}")
        print("Run the benchmark first!")
        return
    
    print("\n" + "="*80)
    print("📊 ANALYZING EXISTING RESULTS")
    print("="*80)
    
    # Load and analyze
    visualizer = BenchmarkVisualizer(csv_path)
    visualizer.print_key_insights()
    
    # Generate visualizations
    print("\n📊 Generating enhanced visualizations...")
    visualizer.generate_all_visualizations()
    
    # Show DataFrame head
    print("\n📋 Sample Data:")
    print(visualizer.df.head())
    
    print("\n✅ Analysis complete!")


def compare_providers_programmatically():
    """Example: Programmatic comparison of providers"""
    
    csv_path = 'search_benchmark_results.csv'
    if not os.path.exists(csv_path):
        print(f"❌ No results found. Run benchmark first.")
        return
    
    print("\n" + "="*80)
    print("🔬 PROGRAMMATIC ANALYSIS EXAMPLE")
    print("="*80)
    
    df = pd.read_csv(csv_path)
    
    # Custom analysis examples
    print("\n1️⃣ Provider Comparison Table:")
    comparison = df.groupby('provider').agg({
        'overall_score': ['mean', 'std', 'min', 'max'],
        'search_time': ['mean', 'std'],
        'relevance_score': 'mean',
        'freshness_score': 'mean',
        'quality_score': 'mean'
    }).round(2)
    print(comparison)
    
    print("\n2️⃣ Best Configuration per Category:")
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        best_idx = cat_data['overall_score'].idxmax()
        best = cat_data.loc[best_idx]
        print(f"\n{category.upper()}:")
        print(f"  Provider: {best['provider']}")
        print(f"  Score: {best['overall_score']:.2f}")
        print(f"  Speed: {best['search_time']:.3f}s")
        if 'max_results' in best:
            print(f"  max_results: {best['max_results']}")
    
    print("\n3️⃣ Score Correlations:")
    numeric_cols = ['relevance_score', 'freshness_score', 'quality_score', 
                   'usefulness_score', 'coverage_score', 'overall_score']
    correlations = df[numeric_cols].corr()['overall_score'].sort_values(ascending=False)
    print(correlations)
    
    print("\n4️⃣ Statistical Significance Test:")
    try:
        from scipy import stats
        
        providers = df['provider'].unique()
        if len(providers) >= 2:
            p1_scores = df[df['provider'] == providers[0]]['overall_score']
            p2_scores = df[df['provider'] == providers[1]]['overall_score']
            
            t_stat, p_value = stats.ttest_ind(p1_scores, p2_scores)
            
            print(f"\nT-test: {providers[0]} vs {providers[1]}")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"  ✅ Statistically significant difference (p < 0.05)")
            else:
                print(f"  ℹ️  No statistically significant difference (p >= 0.05)")
    except ImportError:
        print("  (Install scipy for statistical tests)")


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║         Search Provider Benchmark - Quick Start Menu                 ║
╚═══════════════════════════════════════════════════════════════════════╝

Choose an option:

1. Run Quick Benchmark (minimal config, ~5 minutes)
2. Analyze Existing Results
3. Generate Visualizations Only
4. Programmatic Analysis Example
5. Run Full Benchmark (see search_benchmark_gridsearch.py)

0. Exit
""")
    
    choice = input("Enter your choice (0-5): ").strip()
    
    if choice == "1":
        run_quick_benchmark()
        
    elif choice == "2":
        analyze_existing_results()
        
    elif choice == "3":
        csv_path = input("Enter results CSV path (default: search_benchmark_results.csv): ").strip()
        if not csv_path:
            csv_path = 'search_benchmark_results.csv'
        
        if os.path.exists(csv_path):
            visualizer = BenchmarkVisualizer(csv_path)
            visualizer.generate_all_visualizations()
            print("✅ Visualizations generated!")
        else:
            print(f"❌ File not found: {csv_path}")
    
    elif choice == "4":
        compare_providers_programmatically()
    
    elif choice == "5":
        print("\nTo run the full benchmark, execute:")
        print("  python search_benchmark_gridsearch.py")
        print("\nThis will test all parameter combinations and may take 15-30 minutes.")
    
    elif choice == "0":
        print("👋 Goodbye!")
        return
    
    else:
        print("❌ Invalid choice")
    
    # Ask if user wants to continue
    print("\n" + "="*80)
    cont = input("\nWould you like to return to the menu? (y/n): ").strip().lower()
    if cont == 'y':
        main()


if __name__ == "__main__":
    main()