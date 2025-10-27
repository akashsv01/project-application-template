import matplotlib.pyplot as plt

class Visualizer:
    def create_bug_closure_distribution_chart(self, yearly_distribution, title: str):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plotting stacked bars: top 5 bug closers vs the rest of the contributors
        ax.bar(yearly_distribution["year"], yearly_distribution["top5_pct"],
            label="Top 5 Closers", color="steelblue")
        ax.bar(yearly_distribution["year"], yearly_distribution["rest_pct"],
            bottom=yearly_distribution["top5_pct"], label="Rest", color="lightgray")

        ax.set_ylabel("Percentage of Bug Closures (%)")
        ax.set_xlabel("Year")
        ax.set_title(title)
        ax.legend()

        # Adding percentage labels inside the bars
        for i, year in enumerate(yearly_distribution["year"]):
            top5_val = yearly_distribution.loc[i, "top5_pct"]
            rest_val = yearly_distribution.loc[i, "rest_pct"]

            ax.text(year, top5_val / 2, f"{top5_val:.1f}%", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
            ax.text(year, top5_val + rest_val / 2, f"{rest_val:.1f}%", ha="center", va="center",
                    color="black", fontsize=9)

        plt.tight_layout()
        return fig
    

    def create_top_feature_requesters_chart(self, top_requesters, feature_issues, title="Top 10 Feature Requesters"):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filtering only issues from the top requesters
        top_users = top_requesters.index.tolist()
        subset = feature_issues[feature_issues['creator'].isin(top_users)].copy()
        subset['state'] = subset['state'].astype(str).str.replace("State.", "", regex=False)
        
        # Counting open vs closed per contributor
        status_counts = subset.groupby(['creator', 'state']).size().unstack(fill_value=0)
        status_counts = status_counts.loc[top_requesters.index]

        colors = {"open": "#1f77b4", "closed": "#2ca02c"}
        status_counts.plot(kind='barh', stacked=True, color=colors, ax=ax)

        # Annotate total feature requests per contributor at the end of each bar
        for i, (user, row) in enumerate(status_counts.iterrows()):
            total = row.sum()
            ax.text(total + 0.5, i, str(total), va='center', fontsize=10, fontweight='bold')

        ax.invert_yaxis() # Done so that we keep highest requester on top

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Number of Feature Requests")
        ax.set_ylabel("Contributor")
        ax.legend(title="State", loc="lower right")
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()
        return fig


    def save_figure(self, fig, filename):
        # A wrapper to save a matplotlib figure
        fig.savefig(filename)
        