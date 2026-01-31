import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import numpy as np
import ast

class BenchmarkVisualizer:
    """
    Interactive visualization dashboard for search benchmark results.
    Can be used standalone to analyze existing results.
    """
    
    def __init__(self, csv_path='search_benchmark_results.csv'):
        """Load benchmark results from CSV"""
        self.df = pd.read_csv(csv_path)
        
        # Parse params column if it's stored as string
        if 'params' in self.df.columns:
            def safe_parse_params(x):
                if pd.isna(x):
                    return {}
                if isinstance(x, dict):
                    return x
                try:
                    # Try parsing as JSON
                    return json.loads(x)
                except (json.JSONDecodeError, TypeError):
                    # If that fails, try using ast.literal_eval
                    try:
                        return ast.literal_eval(x)
                    except:
                        # Last resort: return empty dict
                        print(f"Warning: Could not parse params: {x}")
                        return {}
            
            self.df['params'] = self.df['params'].apply(safe_parse_params)
        
        # Extract parameter columns for easier analysis
        if 'params' in self.df.columns:
            self.df['max_results'] = self.df['params'].apply(
                lambda x: x.get('max_results', 5) if isinstance(x, dict) else 5
            )
            self.df['search_depth'] = self.df['params'].apply(
                lambda x: x.get('search_depth', 'N/A') if isinstance(x, dict) else 'N/A'
            )
            self.df['topic'] = self.df['params'].apply(
                lambda x: x.get('topic', 'N/A') if isinstance(x, dict) else 'N/A'
            )
    
    def create_comprehensive_dashboard(self):
        """Create a single comprehensive dashboard with all visualizations"""
        
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Average Score by Provider & Category',
                'Speed vs Quality Trade-off',
                'Score Distribution by Metric',
                'Parameter Impact: max_results',
                'Provider Performance Box Plots',
                'Category Performance Comparison',
                'Search Time Comparison',
                'Model Evaluator Agreement'
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "box"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "box"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.12,
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )
        
        # 1. Heatmap of scores by provider and category
        pivot_data = self.df.groupby(['provider', 'category'])['overall_score'].mean().unstack()
        fig.add_trace(
            go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='RdYlGn',
                showscale=True,
                text=np.round(pivot_data.values, 2),
                texttemplate='%{text}',
                colorbar=dict(x=0.46, len=0.2)
            ),
            row=1, col=1
        )
        
        # 2. Speed vs Quality scatter
        for provider in self.df['provider'].unique():
            provider_data = self.df[self.df['provider'] == provider]
            fig.add_trace(
                go.Scatter(
                    x=provider_data['search_time'],
                    y=provider_data['overall_score'],
                    mode='markers',
                    name=provider,
                    marker=dict(size=8),
                    showlegend=True
                ),
                row=1, col=2
            )
        
        # 3. Score distribution box plots
        metrics = ['relevance_score', 'freshness_score', 'quality_score']
        for metric in metrics:
            fig.add_trace(
                go.Box(y=self.df[metric], name=metric.replace('_score', '').title(), showlegend=False),
                row=2, col=1
            )
        
        # 4. Parameter impact on scores
        param_impact = self.df.groupby(['provider', 'max_results'])['overall_score'].mean().reset_index()
        for provider in self.df['provider'].unique():
            provider_data = param_impact[param_impact['provider'] == provider]
            fig.add_trace(
                go.Bar(x=provider_data['max_results'], y=provider_data['overall_score'], 
                       name=provider, showlegend=False),
                row=2, col=2
            )
        
        # 5. Provider performance box plots
        for provider in self.df['provider'].unique():
            provider_data = self.df[self.df['provider'] == provider]
            fig.add_trace(
                go.Box(y=provider_data['overall_score'], name=provider, showlegend=False),
                row=3, col=1
            )
        
        # 6. Category performance comparison
        cat_scores = self.df.groupby(['category', 'provider'])['overall_score'].mean().reset_index()
        for provider in self.df['provider'].unique():
            provider_data = cat_scores[cat_scores['provider'] == provider]
            fig.add_trace(
                go.Bar(x=provider_data['category'], y=provider_data['overall_score'],
                       name=provider, showlegend=False),
                row=3, col=2
            )
        
        # 7. Search time comparison
        time_data = self.df.groupby('provider')['search_time'].mean().reset_index()
        fig.add_trace(
            go.Bar(x=time_data['provider'], y=time_data['search_time'],
                   marker_color=['#FF6B6B', '#4ECDC4'], showlegend=False),
            row=4, col=1
        )
        
        # 8. Model evaluator agreement
        for model in self.df['eval_model'].unique():
            model_data = self.df[self.df['eval_model'] == model]
            fig.add_trace(
                go.Box(y=model_data['overall_score'], name=model, showlegend=False),
                row=4, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Search Provider Benchmark - Comprehensive Dashboard",
            height=1600,
            showlegend=True,
            font=dict(size=10)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Search Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Overall Score", row=1, col=2)
        
        fig.update_xaxes(title_text="max_results", row=2, col=2)
        fig.update_yaxes(title_text="Overall Score", row=2, col=2)
        
        fig.update_yaxes(title_text="Overall Score", row=3, col=1)
        
        fig.update_xaxes(title_text="Category", row=3, col=2)
        fig.update_yaxes(title_text="Overall Score", row=3, col=2)
        
        fig.update_xaxes(title_text="Provider", row=4, col=1)
        fig.update_yaxes(title_text="Avg Search Time (s)", row=4, col=1)
        
        fig.update_yaxes(title_text="Overall Score", row=4, col=2)
        
        return fig
    
    def create_interactive_explorer(self):
        """Create an interactive data explorer with filters"""
        
        # Create scatter plot with all dimensions
        fig = px.scatter(
            self.df,
            x='search_time',
            y='overall_score',
            color='provider',
            symbol='category',
            size='num_results',
            hover_data={
                'query': True,
                'eval_model': True,
                'relevance_score': ':.2f',
                'freshness_score': ':.2f',
                'quality_score': ':.2f',
                'max_results': True,
                'search_depth': True
            },
            facet_col='category',
            facet_col_wrap=3,
            title='Interactive Benchmark Explorer - Hover for Details',
            labels={
                'search_time': 'Search Time (seconds)',
                'overall_score': 'Overall Quality Score',
                'num_results': 'Results Count'
            }
        )
        
        fig.update_layout(height=800)
        
        return fig
    
    def create_parameter_tuning_guide(self):
        """Create visualization showing optimal parameters for each scenario"""
        
        # Find best parameters for each category
        best_configs = []
        for category in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == category]
            for provider in cat_data['provider'].unique():
                provider_data = cat_data[cat_data['provider'] == provider]
                best_idx = provider_data['overall_score'].idxmax()
                best_row = provider_data.loc[best_idx]
                
                best_configs.append({
                    'Category': category,
                    'Provider': provider,
                    'max_results': best_row['max_results'],
                    'search_depth': best_row['search_depth'],
                    'topic': best_row['topic'],
                    'Score': round(best_row['overall_score'], 2),
                    'Speed': round(best_row['search_time'], 3)
                })
        
        df_best = pd.DataFrame(best_configs)
        
        # Create grouped bar chart
        fig = px.bar(
            df_best,
            x='Category',
            y='Score',
            color='Provider',
            barmode='group',
            hover_data=['max_results', 'search_depth', 'Speed'],
            title='Best Configuration by Category and Provider',
            labels={'Score': 'Overall Score'}
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_correlation_matrix(self):
        """Create correlation matrix for all numeric metrics"""
        
        # Select numeric columns
        numeric_cols = ['search_time', 'num_results', 'relevance_score', 
                       'freshness_score', 'quality_score', 'usefulness_score',
                       'coverage_score', 'overall_score', 'max_results']
        
        corr_matrix = self.df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Matrix - All Metrics",
            height=600,
            width=700
        )
        
        return fig
    
    def create_time_series_comparison(self):
        """Create time series view if multiple runs over time"""
        
        # Group by provider and calculate cumulative stats
        fig = go.Figure()
        
        for provider in self.df['provider'].unique():
            provider_data = self.df[self.df['provider'] == provider].reset_index()
            cumulative_avg = provider_data['overall_score'].expanding().mean()
            
            fig.add_trace(go.Scatter(
                x=provider_data.index,
                y=cumulative_avg,
                mode='lines+markers',
                name=f'{provider} (cumulative avg)',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Cumulative Average Score Over Benchmark Runs",
            xaxis_title="Run Number",
            yaxis_title="Cumulative Average Score",
            height=500
        )
        
        return fig
    
    def create_summary_stats_table(self):
        """Create detailed summary statistics table"""
        
        summary_stats = []
        
        for provider in self.df['provider'].unique():
            provider_data = self.df[self.df['provider'] == provider]
            
            stats = {
                'Provider': provider,
                'Avg Score': round(provider_data['overall_score'].mean(), 2),
                'Std Dev': round(provider_data['overall_score'].std(), 2),
                'Min Score': round(provider_data['overall_score'].min(), 2),
                'Max Score': round(provider_data['overall_score'].max(), 2),
                'Avg Speed': round(provider_data['search_time'].mean(), 3),
                'Avg Relevance': round(provider_data['relevance_score'].mean(), 2),
                'Avg Freshness': round(provider_data['freshness_score'].mean(), 2),
                'Avg Quality': round(provider_data['quality_score'].mean(), 2),
                'Sample Size': len(provider_data)
            }
            summary_stats.append(stats)
        
        df_summary = pd.DataFrame(summary_stats)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df_summary.columns),
                fill_color='#3498db',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[df_summary[col] for col in df_summary.columns],
                fill_color=[['#ecf0f1', '#ffffff'] * len(df_summary)],
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title="Summary Statistics by Provider",
            height=300
        )
        
        return fig
    
    def generate_all_visualizations(self, output_dir='visualizations'):
        """Generate and save all visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        visualizations = {
            'comprehensive_dashboard': self.create_comprehensive_dashboard(),
            'interactive_explorer': self.create_interactive_explorer(),
            'parameter_tuning_guide': self.create_parameter_tuning_guide(),
            'correlation_matrix': self.create_correlation_matrix(),
            'time_series': self.create_time_series_comparison(),
            'summary_stats': self.create_summary_stats_table()
        }
        
        for name, fig in visualizations.items():
            filepath = f'{output_dir}/{name}.html'
            fig.write_html(filepath)
            print(f"✅ Saved: {filepath}")
        
        return visualizations
    
    def print_key_insights(self):
        """Print key insights from the benchmark"""
        print("\n" + "="*80)
        print("🔍 KEY INSIGHTS")
        print("="*80)
        
        # Overall winner
        overall_winner = self.df.groupby('provider')['overall_score'].mean().idxmax()
        overall_score = self.df.groupby('provider')['overall_score'].mean().max()
        print(f"\n🏆 Overall Winner: {overall_winner} (Avg Score: {overall_score:.2f})")
        
        # Speed winner
        speed_winner = self.df.groupby('provider')['search_time'].mean().idxmin()
        avg_speed = self.df.groupby('provider')['search_time'].mean().min()
        print(f"⚡ Fastest Provider: {speed_winner} (Avg Time: {avg_speed:.3f}s)")
        
        # Quality winner
        quality_winner = self.df.groupby('provider')['quality_score'].mean().idxmax()
        quality_score = self.df.groupby('provider')['quality_score'].mean().max()
        print(f"⭐ Highest Quality: {quality_winner} (Avg Quality: {quality_score:.2f})")
        
        # Freshness winner
        freshness_winner = self.df.groupby('provider')['freshness_score'].mean().idxmax()
        freshness_score = self.df.groupby('provider')['freshness_score'].mean().max()
        print(f"🆕 Freshest Results: {freshness_winner} (Avg Freshness: {freshness_score:.2f})")
        
        # Best category performance
        print(f"\n📊 Best Provider by Category:")
        for category in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == category]
            best_provider = cat_data.groupby('provider')['overall_score'].mean().idxmax()
            best_score = cat_data.groupby('provider')['overall_score'].mean().max()
            print(f"  {category}: {best_provider} ({best_score:.2f})")
        
        # Parameter insights
        print(f"\n⚙️  Optimal Parameters:")
        for provider in self.df['provider'].unique():
            provider_data = self.df[self.df['provider'] == provider]
            best_idx = provider_data['overall_score'].idxmax()
            best_params = provider_data.loc[best_idx, 'max_results']
            print(f"  {provider}: max_results={best_params}")
        
        # Statistical significance
        print(f"\n📈 Statistical Summary:")
        print(f"  Total Benchmark Runs: {len(self.df)}")
        print(f"  Categories Tested: {self.df['category'].nunique()}")
        print(f"  Evaluation Models: {self.df['eval_model'].nunique()}")
        if 'params' in self.df.columns:
            # Convert dicts to strings to count unique combinations
            unique_params = self.df['params'].apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, dict) else str(x)).nunique()
            print(f"  Parameter Combinations: {unique_params}")
        
        print("\n" + "="*80)


# ============================================================================
# STANDALONE USAGE
# ============================================================================
if __name__ == "__main__":
    import sys
    
    # Check if results file exists
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = 'search_benchmark_results.csv'
    
    print(f"Loading benchmark results from: {csv_path}")
    
    try:
        visualizer = BenchmarkVisualizer(csv_path)
        
        # Print insights
        visualizer.print_key_insights()
        
        # Generate all visualizations
        print("\n📊 Generating visualizations...")
        visualizer.generate_all_visualizations()
        
        print("\n✅ Complete! Open the HTML files in visualizations/ to explore results.")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {csv_path}")
        print("Please run the benchmark first or provide a valid CSV path.")
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()