import argparse
import os
from visualization_utils import positional_encoder_visualization

def main():
    parser = argparse.ArgumentParser(description="Visualization argument parser")


    pe_classas = list(positional_encoder_visualization.pe_classas.keys())
    parser.add_argument("--pe",
                        type=str,
                        required=False,
                        choices = pe_classas,
                        help="Visualization type for positional encoders")

    parser.add_argument("--imgpth",
                        type=str,
                        default="visualization_outputs",
                        help="Path to save the visualization images")
    
    args = parser.parse_args()

    path = args.imgpth
    print(f"Output path: {path}")
    print(f"Current working directory: {os.getcwd()}")

    import matplotlib.pyplot as plt
    if args.pe is not None:
        print(f"Visualizing positional encoding for {args.pe}")
        try:
            positional_encoder_visualization.visualize_positional_encoding(path, args.pe, dim=64, max_len=5000)
        except Exception as e:
            print(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()



    

if __name__ == "__main__":
    main()