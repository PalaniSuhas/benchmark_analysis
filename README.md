# Search Provider Benchmark Analysis

A comprehensive benchmarking framework for evaluating web search APIs (Tavily, Brave) across multiple categories using various LLM evaluators.

## Overview

This project compares search provider performance across different domains (tech, geopolitics, sports, business, science) using grid search methodology to optimize parameters like max_results, search_depth, and topic selection.

## Key Features

- Grid search parameter optimization for search providers
- Multi-model LLM evaluation (GPT-4o, GPT-5, Gemini 2.5)
- Interactive visualizations with Plotly
- Comprehensive performance metrics (relevance, freshness, quality, speed)
- Statistical analysis and correlation studies

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys in `.env`:
```
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
BRAVE_API_KEY=your_key
TAVILY_API_KEY=your_key
```

3. Run benchmark:
```bash
python search_benchmark_gridsearch.py
```

4. View results:
- Open `visualizations/index.html` in browser
- Or run: `python visualizer.py` to generate visualizations

## Project Structure

- `search_benchmark_gridsearch.py` - Main grid search benchmark
- `visualizer.py` - Visualization generator
- `search_benchmark_results.csv` - Raw benchmark data
- `benchmark_summary.json` - Statistical summary
- `visualizations/` - Interactive HTML dashboards

## Results

View live interactive dashboard: https://palanisuhas.github.io/benchmark_analysis/visualizations/

Key findings:
- Brave: Faster average search time (0.5s vs 1.1s)
- Tavily: Higher quality scores with advanced search depth
- Optimal parameters vary by category

## Analysis Tools

- `grid_search_cv.py` - GridSearchCV-style parameter optimization
- `quickstart.py` - Quick benchmark with reduced parameters
- `visualizer.py` - Standalone visualization tool

## Citation

Results based on January 2026 benchmark runs across 5 categories, 2 search providers, and 9+ LLM evaluators.