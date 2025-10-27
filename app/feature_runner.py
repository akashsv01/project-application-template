from config import ConfigManager
from controllers.contributors_controller import ContributorsController

class FeatureRunner:
    def __init__(self):
        self.config = None
        self.contributors_controller = None

    def initialize_components(self):
        # Initialize config and the ContributorsController
        self.config = ConfigManager("config.json")
        self.contributors_controller = ContributorsController()

    def run_feature(self, feature_number: int, user: str = None, label: str = None):
        """
        Dispatch to the appropriate controller based on feature number.
        1 = Lifecycle Analysis
        2 = Contributors Dashboard
        3 = Priority Analysis
        """
        if feature_number == 1:
            print("▶ Running Lifecycle Analysis...")
            pass

        elif feature_number == 2:
            print("▶ Running Contributors Dashboard...")
            data_path = self.config.get_data_path()
            output_path = self.config.get_output_path()

            issues_df, events_df = self.contributors_controller.load_contributor_data()
            contributors_df = self.contributors_controller.analyzer.build_contributors(issues_df, events_df)

            figs = {}
            
            # ---------------- Graph 1: Bug Closure Distribution ----------------
            figs["graph1_bug_closures"] = self.contributors_controller.plot_bug_closure_distribution(
                issues_df, events_df
            )

            # ---------------- Graph 2: Top Feature Requesters ----------------
            fig2 = self.contributors_controller.plot_top_feature_requesters(issues_df, top_n=10)
            if fig2 is not None:
                figs["graph2_top_feature_requesters"] = fig2

            # ---------------- Graph 3: Docs Issues Analysis ----------------
            fig3 = self.contributors_controller.plot_docs_issues(issues_df, events_df)
            if fig3 is not None:
                figs["graph3_docs_issues"] = fig3

            # Saving the figures in output path
            for name, fig in figs.items():
                self.contributors_controller.visualizer.save_figure(fig, f"{output_path}/{name}.png")
                print(f"Saved {name} → {output_path}/{name}.png")

            # Displaying the figures
            import matplotlib.pyplot as plt
            plt.show()


        elif feature_number == 3:
            print("▶ Running Priority Analysis...")
            pass

        else:
            print("❌ Oops, this is an unknown feature number! Use --feature 1, 2, or 3.")