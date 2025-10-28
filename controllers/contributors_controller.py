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

    def plot_docs_issues(self, issues_df, events_df):
        """Controller method for Graph 3: Documentation Issues (open vs closed per month).
            Returns a figure with stacked bars for issue counts and a line showing the
            average number of unique commenters per doc issue each month."""
        status_counts, avg_commenters = self.analyzer.analyze_docs_issues(issues_df, events_df, self.data_loader)
        if status_counts is None:
            return None
        return self.visualizer.create_docs_issues_chart(
            status_counts,
            avg_commenters,
            "Docs Issues: Open vs Closed per Month with Avg Commenters"
        )
    
    def plot_issues_created_per_user(self, issues_df, top_n=40):
        """Controller method for Graph 4: Top 40 Contributors by Issues Created.
           Returns a figure showing the top 40 users ranked by number of issues created."""
        issues_per_user, all_counts = self.analyzer.analyze_issues_created_per_user(issues_df, top_n=top_n)
        if issues_per_user is None:
            return None

        return self.visualizer.create_issues_created_per_user_chart(
            issues_per_user,
            all_counts,
            f"Top {top_n} Contributors by Issues Created"
        )

    def plot_top_active_users_per_year(self, contributors, top_n=10):
        """Controller method for Graph 5: Top Active Users per Year.
        """
        yearly_data = self.analyzer.analyze_top_active_users_per_year(contributors)

        return self.visualizer.create_top_active_users_per_year_chart(yearly_data, top_n)