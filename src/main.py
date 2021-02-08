from system import System
from scheduler import SchedulerFactory, AbstractScheduler
import argparse
from analysis import AnalysisFactory
from nu import NuFactory
from datetime import datetime, timedelta
import random
from task import UserTask


def parent(args):
    print(args)
    pass


# Sub-command functions
def genetic_sch(args):
    sys = System()
    start_time = get_start_time(args)

    sys.set_scheduler("genetic")
    sys.scheduler = SchedulerFactory.genetic_scheduler(
        start_time=start_time,
        population_size=args.population_size,
        breeding_percentage=args.breeding_percentage,
        mutation_rate=args.mutation_rate,
        elitism=args.elitism,
        max_iterations=args.max_iterations,
        threshold=args.threshold,
        generational_threshold=args.generational_threshold,
        # start_time=None,
        verbose=args.verbose,
        invalid_schedule_value=args.invalid_schedule_value,
    )

    # sys.scheduler.start_time = start
    sys._tasks = get_tasks(args)
    sys.schedule_tasks()
    gen_sch = sys._schedule
    total_gen_value = sys.simulate_schedule()  # start_time=start.timestamp())
    weighted_gen_value = sys.scheduler.weighted_schedule_value(gen_sch, total_gen_value)

    print("Genetic Scheduler Value: " + str(total_gen_value))
    print("Genetic Weighted Scheduler Value: " + str(weighted_gen_value))

    after_parse(args, sys._tasks)


def ant_sch(args):
    sys = System()
    start = earliest_start = datetime.now() + timedelta(minutes=5)

    sys.scheduler = SchedulerFactory.ant_scheduler(
        start_time=start_time,
        colony_size=args.colony_size,
        alpha=args.alpha,
        beta=args.beta,
        epsilon=args.epsilon,
        max_iterations=args.max_iterations,
        threshold=args.threshold,
        generational_threshold=5
    )
    sys.scheduler.start_time = start
    sys._tasks = get_tasks(args)
    sys.schedule_tasks()
    ant_sch = sys._schedule
    total_ant_value = sys.simulate_schedule()  # start_time=start.timestamp())
    weighted_ant_value = sys.scheduler.weighted_schedule_value(ant_sch, total_ant_value)

    print("Ant Scheduler Value: " + str(total_ant_value))
    print("Ant Weighted Scheduler Value: " + str(weighted_ant_value))

    after_parse(args, sys._tasks)


def annealing_sch(args):

    sys = System()
    start_time = get_start_time(args)

    sys._tasks = get_tasks(args)

    sys.scheduler = SchedulerFactory.simulated_annealing(
        start_time=start_time,
        max_iterations=args.max_iterations,
        generational_threshold=args.generational_threshold,
    )

    # Check for end_time
    if args.end_time is not None and args.end_time > args.start_time:
        sys.scheduler.end_time = datetime.fromtimestamp(args.end_time)

    sys.schedule_tasks()
    anneal_sch = sys._schedule
    total_anneal_value = sys.simulate_schedule()
    weighted_anneal_value = sys.scheduler.weighted_schedule_value(anneal_sch, total_anneal_value)
    print("Simulated Annealing Scheduler Value " + str(total_anneal_value))
    print("Simulated Weighted Annealing Scheduler Value " + str(weighted_anneal_value))

    after_parse(args, sys._tasks)


def random_sch(args):
    sys = System()
    start = earliest_start = datetime.now() + timedelta(minutes=5)
    sys._tasks = get_tasks(args)
    sys.scheduler = SchedulerFactory.random_scheduler(sample_size=args.sample_size)
    sys.schedule_tasks()
    random_sch = sys._schedule
    total_random_value = sys.simulate_schedule()  # start_time=start.timestamp())
    weighted_random_value = sys.scheduler.weighted_schedule_value(random_sch, total_random_value)

    print("Random Scheduler Value: " + str(total_random_value))
    print("Random Weighted Scheduler Value: " + str(weighted_random_value))

    after_parse(args, sys._tasks)


def get_tasks(args):

    if args.load_tasklist is not None:
        return UserTask.load_tasks(args.load_tasklist)
    else:
        return UserTask.generate_random_tasks(args.generate_tasks)


def get_start_time(args):
    """

    :param args:
    :return:
    """
    if args.start_time is None:
        return datetime.now() + timedelta(minutes=5)
    else:
        return datetime.fromtimestamp(args.start_time)


def get_end_time(args):
    """

    :param args:
    :return:
    """
    if args.end_time is None:
        return datetime.now() + timedelta(minutes=5)
    else:
        return datetime.now() + timedelta(seconds=args.end_time)

def after_parse(args, tasklist=None):
    if args.export_tasklist is not None:
        UserTask.save_tasks(args.export_tasklist, tasklist)


# Main Parser
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true', help='foo help')
parser.add_argument('-e', '--export_tasklist', type=str, help='export tasklist')
tasks = parser.add_mutually_exclusive_group(required=True)
tasks.add_argument('-l', '--load_tasklist', type=str, help="")
tasks.add_argument('-g', '--generate_tasks', type=int, help="")
subparsers = parser.add_subparsers(help='sub-command help')
parser.add_argument('--start_time', type=int, help="")
parser.add_argument('--end_time', type=float, help="")

#parser.add_argument("-g", "--generate_tasks", type=int, help="Generates Random Tasks")

# create the parser for the "genetic" command
parser_genetic = subparsers.add_parser('genetic', help='genetic help')
parser_genetic.add_argument('--population_size', type=int, help='bar help', default=500)
parser_genetic.add_argument('--breeding_percentage', type=float, help='bar help', default=0.2)
parser_genetic.add_argument('--mutation_rate', type=float, help='bar help', default=0.05)
parser_genetic.add_argument('--max_iterations', type=int, help='bar help', default=10)
parser_genetic.add_argument('--threshold', type=float, help='bar help', default=0.01)
parser_genetic.add_argument('--generational_threshold', type=int, help='bar help', default=20)
parser_genetic.add_argument('--invalid_schedule_value', type=int, help='bar help', default=-100)
parser_genetic.add_argument('--elitism', type=bool, help='bar help', default=True)
parser_genetic.set_defaults(func=genetic_sch)


# create the parser for the "ant" command
parser_ant = subparsers.add_parser('ant', help='ant help')
parser_ant.add_argument('--colony_size', type=int, help='bar help', default=30)
parser_ant.add_argument('--alpha', type=int, help='bar help', default=1)
parser_ant.add_argument('--beta', type=int, help='bar help', default=-1)
parser_ant.add_argument('--epsilon', type=float, help='bar help', default=-0.5)
parser_ant.add_argument('--max_iterations', type=int, help='bar help', default=10)
parser_ant.add_argument('--threshold', type=float, help='bar help', default=0.01)
parser_ant.add_argument('--generational_threshold', type=int, help='bar help', default=5)
parser_ant.add_argument('--invalid_schedule_value', type=int, help='bar help', default=-100)
parser_ant.add_argument('--elitism', type=bool, help='bar help', default=True)
parser_ant.set_defaults(func=ant_sch)


# create the parser for the "annealing" command
parser_annealing = subparsers.add_parser('annealing', help='Simulated Annealing help')
parser_annealing.add_argument('--max_iterations', type=int, help='bar help', default=10000)
parser_annealing.add_argument('--generational_threshold', type=int, help='bar help', default=50)
parser_annealing.set_defaults(func=annealing_sch)

# create the parser for the "random" command
parser_random = subparsers.add_parser('random', help='Simulated Annealing help')
parser_random.add_argument('--sample_size', type=int, help='bar help', default=1000)
parser_random.set_defaults(func=random_sch)

args = parser.parse_args()
args.func(args)


