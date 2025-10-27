from data_loader import DataLoader
from analysis.contributors_analyzer import ContributorsAnalyzer
from visualization.visualizer import Visualizer

class ContributorsController:
    def __init__(self):
        self.data_loader = DataLoader()
        self.analyzer = ContributorsAnalyzer()
        self.visualizer = Visualizer()

    def load_contributor_data(self):
        issues_df = self.data_loader.parse_issues()
        events_df = self.data_loader.parse_events(issues_df)
        return issues_df, events_df

    def plot_bug_closure_distribution(self, issues_df, events_df):
        """Controller method for Graph 1: Bug Closure Distribution. Analyzes yearly 
           bug closures and plots the share handled by the top 5 contributors 
           vs the rest of the community."""
        yearly_distribution = self.analyzer.analyze_bug_closure_distribution(issues_df, events_df)

        # Printing top 5 bug closers per year in CLI
        for _, row in yearly_distribution.iterrows():
            print(f"Year {int(row['year'])}: Top 5 bug closers -> {row['top5_users']}")

        fig = self.visualizer.create_bug_closure_distribution_chart(
            yearly_distribution,
            "Community Load Distribution: % of Bug Closures by Top 5 vs Rest"
        )
        return fig

    def plot_top_feature_requesters(self, issues_df, top_n=10):
        """Controller method for Graph 2: Top 10 Feature Requesters (open vs closed).
           Returns a figure showing stacked bars for top requesters."""
        top_requesters, feature_issues = self.analyzer.analyze_top_feature_requesters(issues_df, top_n=top_n)
        if top_requesters is None:
            return None

        fig = self.visualizer.create_top_feature_requesters_chart(
            top_requesters,
            feature_issues,
            "Top 10 Feature Requesters"
        )
        return fig


