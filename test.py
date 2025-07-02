import argparse

def main():
    parser = argparse.ArgumentParser(description="Test argument parser")
    parser.add_argument("--train", type=str, required=True, help="Input file path")
    parser.add_argument("--inference", type=str, required=True, help="Output file path")
    
    args = parser.parse_args()
    
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")

if __name__ == "__main__":
    main()