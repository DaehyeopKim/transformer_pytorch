import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualization argument parser")
    parser.add_argument("--visualize",
                        type=str,
                        required=True,
                        choices = ["attention", "loss", "gradient_flow"],
                        help="Visualization type (e.g., 'attention', 'loss')")