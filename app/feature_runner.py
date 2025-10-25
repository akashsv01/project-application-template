from config import ConfigManager

class FeatureRunner:

    def __init__(self):
        self.config = None
        self.contributors_controller = None

    def initialize_components(self):
        """Initialize config and the ContributorsController."""
        self.config = ConfigManager("config.json")

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
            pass

        elif feature_number == 3:
            print("▶ Running Priority Analysis...")
            pass

        else:
            print("❌Oops, this is an unknown feature number! Use --feature 1, 2, or 3.")