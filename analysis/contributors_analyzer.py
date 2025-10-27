import pandas as pd
from model import Contributor

class ContributorsAnalyzer:
    def build_contributors(self, issues_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a contributors DataFrame from issues and events.
        Each row = one contributor with activity stats.
        """
        contributors = {}

        # Loop through issues and record who created them
        for _, row in issues_df.iterrows():
            creator = row["creator"]
            if creator not in contributors:
                contributors[creator] = Contributor(creator)
            contributors[creator].issues_created.append(row["number"])

        # Loop through events and record who commented/closed
        for _, row in events_df.iterrows():
            author = row["event_author"]
            if not author:
                continue
            if author not in contributors:
                contributors[author] = Contributor(author)
            contributors[author].comments.append(row["issue_number"])

        # Flatten Contributor objects into a DataFrame
        data = []
        for username, c in contributors.items():
            data.append({
                "username": username,
                "issues_created": len(c.issues_created),
                "comments": len(c.comments),
                "activity_count": c.get_activity_count(),
                "first_activity": c.first_activity,
                "last_activity": c.last_activity
            })

        return pd.DataFrame(data)

    def analyze_bug_closure_distribution(self, issues_df, events_df) -> pd.DataFrame:
        # Filtering only issues labeled as bugs
        bug_issues = issues_df[
            issues_df['labels'].apply(lambda L: any('bug' in l.lower() for l in L))
        ].copy()

        # Find closure events for those bug issues
        bug_closures = events_df[
            (events_df['issue_number'].isin(bug_issues['number'])) &
            (events_df['event_type'] == 'closed')
        ].copy()
        bug_closures['year'] = bug_closures['event_date'].dt.year
        
        # Counting how many bugs each contributor closed per year
        closer_counts = (
            bug_closures.groupby(['year', 'event_author'])
            .size()
            .reset_index(name='n_closed')
        )

        # A helper function to split yearly totals into top 5 vs the rest
        def top5_vs_rest(df):
            df = df.sort_values('n_closed', ascending=False)
            top5 = df.head(5)
            rest = df['n_closed'].sum() - top5['n_closed'].sum()
            total = df['n_closed'].sum()
            return pd.Series({
                'top5_pct': (top5['n_closed'].sum() / total) * 100 if total > 0 else 0,
                'rest_pct': (rest / total) * 100 if total > 0 else 0,
                'top5_users': ", ".join(top5['event_author'].tolist())
            })

        # Apply the split per year
        yearly_distribution = (
            closer_counts.groupby('year')
            .apply(top5_vs_rest)
            .reset_index()
        )

        return yearly_distribution
    
    
    def analyze_top_feature_requesters(self, issues_df, top_n=10):
        # Filtering only issues labeled as features
        feature_issues = issues_df[
            issues_df['labels'].apply(lambda L: any('feature' in str(l).lower() for l in L))
        ]
        if feature_issues.empty:
            return None, None
        
        # Ranking creators by number of feature requests
        top_requesters = feature_issues['creator'].value_counts().head(top_n)
        return top_requesters, feature_issues
    