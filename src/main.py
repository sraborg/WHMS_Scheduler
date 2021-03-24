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
    unscheduled_tasks = get_tasks(args)
    sys._tasks = unscheduled_tasks
    sys.sleep_interval = timedelta(minutes=args.sleep_interval)

    method: str = args.method
    if method == "ga":
        sys.scheduler = SchedulerFactory.genetic_scheduler(
            start_time=datetime.fromtimestamp(args.start_time),
            end_time=datetime.fromtimestamp(args.end_time),
            population_size=args.population_size,
            breeding_percentage=args.breeding_percentage,
            mutation_rate=args.mutation_rate,
            elitism=args.elitism,
            max_iterations=args.max_iterations,
            threshold=args.threshold,
            generational_threshold=args.generational_threshold,
            verbose=args.verbose,
            invalid_schedule_value=args.invalid_schedule_value,
            learning_duration=args.learning_duration
        )
    elif method == "nga":
        sys.scheduler = SchedulerFactory.new_genetic_scheduler(
            start_time=datetime.fromtimestamp(args.start_time),
            end_time=datetime.fromtimestamp(args.end_time),
            population_size=args.population_size,
            breeding_percentage=args.breeding_percentage,
            verbose=args.verbose,
            invalid_schedule_value=args.invalid_schedule_value,
            learning_duration=args.learning_duration
        )

    sys.schedule_tasks()
    gen_sch = sys._schedule
    utopian = sys.scheduler.utopian_schedule_value(unscheduled_tasks)
    raw_value = sys.simulate_schedule()  # start_time=start.timestamp())
    weighted_value = sys.scheduler.weighted_schedule_value(unscheduled_tasks, raw_value)

    #print("Genetic Scheduler Value: " + str(total_gen_value))
    #print("Genetic Weighted Scheduler Value: " + str(weighted_gen_value))
    print_results(sys.scheduler.algorithm_name(), raw_value, utopian, weighted_value)
    after_parse(args, sys._tasks)


def ant_sch(args):
    sys = System()
    unscheduled_tasks = get_tasks(args)
    sys._tasks = unscheduled_tasks
    sys.sleep_interval = timedelta(minutes=args.sleep_interval)
    method: str = args.method
    if method.upper() == "AS":
        sys.scheduler = SchedulerFactory.ant_scheduler(
            start_time=datetime.fromtimestamp(args.start_time),
            end_time=datetime.fromtimestamp(args.end_time),
            colony_size=args.colony_size,
            alpha=args.alpha,
            beta=args.beta,
            epsilon=args.epsilon,
            max_iterations=args.max_iterations,
            threshold=args.threshold,
            generational_threshold=5,
            learning_duration=args.learning_duration
        )
    if method.upper() == "ACO":
        sys.scheduler = SchedulerFactory.ant_colony_scheduler(
            start_time=datetime.fromtimestamp(args.start_time),
            end_time = datetime.fromtimestamp(args.end_time),
            colony_size=args.colony_size,
            alpha=args.alpha,
            beta=args.beta,
            epsilon=args.epsilon,
            max_iterations=args.max_iterations,
            threshold=args.threshold,
            generational_threshold=5,
            learning_duration=args.learning_duration
        )

    if method.upper() == "ELITE":
        sys.scheduler = SchedulerFactory.ElitistAntScheduler(
            start_time=datetime.fromtimestamp(args.start_time),
            end_time=datetime.fromtimestamp(args.end_time),
            colony_size=args.colony_size,
            alpha=args.alpha,
            beta=args.beta,
            epsilon=args.epsilon,
            max_iterations=args.max_iterations,
            threshold=args.threshold,
            generational_threshold=5,
            learning_duration=args.learning_duration
        )

    sys.schedule_tasks()
    ant_sch = sys._schedule
    raw_value = sys.simulate_schedule()  # start_time=start.timestamp())
    utopian = sys.scheduler.utopian_schedule_value(unscheduled_tasks)
    weighted_value = sys.scheduler.weighted_schedule_value(unscheduled_tasks, raw_value)

    print_results(sys.scheduler.algorithm_name(), raw_value, utopian, weighted_value)
    after_parse(args, sys._tasks)


def annealing_sch(args):

    sys = System()
    start_time = get_start_time(args)
    unscheduled_tasks = get_tasks(args)
    sys._tasks = unscheduled_tasks
    sys.sleep_interval = timedelta(minutes=args.sleep_interval)

    method: str = args.method
    if method == "sa":
        sys.scheduler = SchedulerFactory.simulated_annealing(
            start_time=datetime.fromtimestamp(args.start_time),
            end_time=datetime.fromtimestamp(args.end_time),
            max_iterations=args.max_iterations,
            generational_threshold=args.generational_threshold,
            learning_duration=args.learning_duration
        )
    elif method == "elbsa":
        sys.scheduler = SchedulerFactory.enhanced_list_based_simulated_annealing(
            start_time=datetime.fromtimestamp(args.start_time),
            end_time=datetime.fromtimestamp(args.end_time),
            max_iterations=args.max_iterations,
            generational_threshold=args.generational_threshold,
            learning_duration=args.learning_duration,
        )

    sys.schedule_tasks()
    #sys.scheduler._tasks[0].nu.shift_deadlines(10)
    anneal_sch = sys._schedule
    #total_anneal_value = sys.simulate_schedule()
    #utopian = sys.scheduler.utopian_schedule_value(tasks)
    #weighted_anneal_value = sys.scheduler.weighted_schedule_value(tasks, total_anneal_value)
    #print("Simulated Annealing Scheduler Value " + str(total_anneal_value))
    #print("Simulated Weighted Annealing Scheduler Value " + str(weighted_anneal_value))
    raw_value = sys.simulate_schedule()  # start_time=start.timestamp())
    utopian = sys.scheduler.utopian_schedule_value(unscheduled_tasks)
    weighted_value = sys.scheduler.weighted_schedule_value(unscheduled_tasks, raw_value)

    print_results(sys.scheduler.algorithm_name(), raw_value, utopian, weighted_value)
    after_parse(args, sys._tasks)


def random_sch(args):
    sys = System()
    sys._tasks = get_tasks(args)
    start_time = get_start_time(args)

    sys.scheduler = SchedulerFactory.random_scheduler(
        start_time=start_time,
        sample_size=args.sample_size
    )

    # Check for end_time
    if args.end_time is not None:
        if args.end_time <= args.start_time:
            raise Exception("Endtime must be later than start time")
        sys.scheduler.end_time = datetime.fromtimestamp(args.end_time)

    sys.schedule_tasks()
    random_sch = sys._schedule
    total_random_value = sys.simulate_schedule()  # start_time=start.timestamp())
    weighted_random_value = sys.scheduler.weighted_schedule_value(random_sch, total_random_value)

    print("Random Scheduler Value: " + str(total_random_value))
    print("Random Weighted Scheduler Value: " + str(weighted_random_value))

    after_parse(args, sys._tasks)


def print_results(algorithm: str, raw: float, utopian: float, weighted: float):
    print(algorithm)
    print("Raw Value: " + str(raw))
    print("Utopian Value: " + str(utopian))
    print("Weighted Value: " + str(weighted))

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
parser.add_argument('--learning-duration', type=int, help='How many minutes to run the algorithm', default=1)
parser.add_argument('--sleep_interval', type=int, help='Duration for each sleep task', default=1)
parser.add_argument('-e', '--export-tasklist', type=str, help='export tasklist')
tasks = parser.add_mutually_exclusive_group(required=True)
tasks.add_argument('-l', '--load-tasklist', type=str, help="")
tasks.add_argument('-g', '--generate-tasks', type=int, help="")
subparsers = parser.add_subparsers(help='sub-command help')
parser.add_argument('--start-time', type=int, help="")
parser.add_argument('--end-time', type=float, help="")

#parser.add_argument("-g", "--generate_tasks", type=int, help="Generates Random Tasks")

# create the parser for the "genetic" command
gen_arg_choices = ["ga", "nga"]
parser_genetic = subparsers.add_parser('genetic', help='genetic help')
parser_genetic.add_argument("-m", "--method", type=str, choices=gen_arg_choices)
parser_genetic.add_argument('--population_size', type=int, help='bar help', default=20)
parser_genetic.add_argument('--breeding_percentage', type=float, help='bar help', default=0.2)
parser_genetic.add_argument('--mutation_rate', type=float, help='bar help', default=0.05)
parser_genetic.add_argument('--max_iterations', type=int, help='bar help', default=10)
parser_genetic.add_argument('--threshold', type=float, help='bar help', default=0.01)
parser_genetic.add_argument('--generational_threshold', type=int, help='bar help', default=20)
parser_genetic.add_argument('--invalid_schedule_value', type=int, help='bar help', default=-100)
parser_genetic.add_argument('--elitism', type=bool, help='bar help', default=True)
parser_genetic.set_defaults(func=genetic_sch)


# create the parser for the "ant" command
ant_arg_choices = ["as", "aco", "elite"]
parser_ant = subparsers.add_parser('ant', help='ant help')
parser_ant.add_argument("-m", "--method", type=str, choices=ant_arg_choices)
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
anneal_arg_choices = ["sa", "elbsa"]
parser_annealing = subparsers.add_parser('annealing', help='Simulated Annealing help')
parser_annealing.add_argument("-m", "--method", type=str, choices=anneal_arg_choices)
parser_annealing.add_argument('--max_iterations', type=int, help='bar help', default=10000)
parser_annealing.add_argument('--generational_threshold', type=int, help='bar help', default=50)
parser_annealing.set_defaults(func=annealing_sch)

# create the parser for the "random" command
parser_random = subparsers.add_parser('random', help='Simulated Annealing help')
parser_random.add_argument('--sample_size', type=int, help='bar help', default=1000)
parser_random.set_defaults(func=random_sch)

args = parser.parse_args()
args.func(args)


