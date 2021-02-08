import argparse
from datetime import datetime
from task import UserTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--quantity', type=int, help="", required=True)
    parser.add_argument('-e', '--export_tasklist',  type=str, help='export tasklist', default="generated_tasks.json")
    parser.add_argument('-d', "--dependencies", type=bool, help="", default=True)
    parser.add_argument('--start_time', type=float, help="", required=True)
    parser.add_argument('--end_time', type=float, help="True")
    parser.add_argument('--max_value', type=int, help="")

    args = parser.parse_args()

    print("Generating " + str(args.quantity) + " tasks.")
    tasks = UserTask.generate_random_tasks(
        args.quantity,
        args.dependencies,
        datetime.fromtimestamp(args.start_time),
        datetime.fromtimestamp(args.end_time),
        args.max_value
    )

    UserTask.save_tasks(args.export_tasklist, tasks)


if __name__ == "__main__":
    main()