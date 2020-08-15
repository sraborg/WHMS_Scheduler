import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("echo", help="echo the string you use here")
    parser.add_argument("-v", "--verbosity", help="increase output verbosity", action="store_true")
    #parser.add_argument("-a", "--algorithm", help="selects the scheduling algorithm", choices={"genetic", "model"}, default="genetic")


    # Dummy Tasks
    args = parser.parse_args()


    if args.verbosity:
        print("verbosity turned on")
        print(args.scheduler)