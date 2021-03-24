import argparse
from datetime import datetime
from task import UserTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--quantity', type=int, help="", required=True)
    parser.add_argument('-e', '--export-tasklist',  type=str, help='export tasklist', default="generated_tasks.json")
    parser.add_argument('-d', "--dependencies", type=bool, help="", default=True)
    parser.add_argument('--start-time', type=float, help="", required=True)
    parser.add_argument('--end-time', type=float, help="True")
    parser.add_argument('--max-value', type=int, help="", default=10)

    args = parser.parse_args()

    print("Generating " + str(args.quantity) + " tasks.")
    tasks = UserTask.generate_random_tasks(
        quantity=args.quantity,
        start=datetime.fromtimestamp(args.start_time),
        end=datetime.fromtimestamp(args.end_time),
        dependencies=args.dependencies,
        max_value=args.max_value
    )

    UserTask.save_tasks(args.export_tasklist, tasks)


if __name__ == "__main__":
    main()