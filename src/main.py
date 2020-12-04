from system import System
from scheduler import SchedulerFactory, AbstractScheduler
import argparse
from analysis import AnalysisFactory
from nu import NuFactory
from datetime import datetime, timedelta
import random
from task import UserTask


# Sub-command functions
def genetic(args):
    sys = System()
    start = earliest_start = datetime.now() + timedelta(minutes=5)

    sys._tasks = UserTask.generate_random_tasks(200)

    sys.set_scheduler("genetic")
    sys.scheduler = SchedulerFactory.genetic_scheduler(
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
    sys.schedule_tasks()
    gen_sch = sys._schedule
    total_gen_value = sys.simulate_schedule()  # start_time=start.timestamp())
    weighted_gen_value = sys.scheduler.weighted_schedule_value(gen_sch, total_gen_value)

    print("Genetic Scheduler Value: " + str(total_gen_value))
    print("Genetic Weighted Scheduler Value: " + str(weighted_gen_value))


def ant(args):
    sys = System()
    start = earliest_start = datetime.now() + timedelta(minutes=5)

    sys.scheduler = SchedulerFactory.ant_scheduler(
        colony_size=args.colony_size,
        alpha=args.alpha,
        beta=args.beta,
        epsilon=args.epsilon,
        max_iterations=args.max_iterations,
        threshold=args.threshold,
        generational_threshold=5
    )
    sys.scheduler.start_time = start
    sys._tasks = UserTask.generate_random_tasks(200)
    sys.schedule_tasks()
    ant_sch = sys._schedule
    total_ant_value = sys.simulate_schedule()  # start_time=start.timestamp())
    weighted_ant_value = sys.scheduler.weighted_schedule_value(ant_sch, total_ant_value)

    print("Ant Scheduler Value: " + str(total_ant_value))
    print("Ant Weighted Scheduler Value: " + str(weighted_ant_value))

# Top-level Parser
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true', help='foo help')
subparsers = parser.add_subparsers(help='sub-command help')

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
parser_genetic.set_defaults(func=genetic)


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
parser_ant.set_defaults(func=ant)



# create the parser for the "annealing" command
parser_annealing = subparsers.add_parser('annealing', help='Simulated Annealing help')

# create the parser for the "random" command
parser_random = subparsers.add_parser('random', help='Simulated Annealing help')

args = parser.parse_args()
args.func(args)






'''


sys.scheduler = SchedulerFactory.random_scheduler(max_iterations=20000)
sys.schedule_tasks()
random_sch = sys._schedule
total_random_value = sys.simulate_schedule()#start_time=start.timestamp())
weighted_random_value = sys.scheduler.weighted_schedule_value(random_sch, total_random_value)

sys.scheduler = SchedulerFactory.simulated_annealing(
    max_iterations=100000,
    generational_threshold=10000,
)
sys.schedule_tasks()
anneal_sch = sys._schedule
total_anneal_value = sys.simulate_schedule()
weighted_anneal_value = sys.scheduler.weighted_schedule_value(anneal_sch, total_anneal_value)

'''

'''
print("Random Scheduler Value: " + str(total_random_value))
print("Random Weighted Scheduler Value: " + str(weighted_random_value))
print("Simulated Annealing Scheduler Value " + str(total_anneal_value))
print("Simulated Weighted Annealing Scheduler Value " + str(weighted_anneal_value))
'''
'''
sys.scheduler.save_tasklist("123xxx.csv", sys._tasks)

loaded_tasks = sys.scheduler.load_tasklist("123xxx.csv")
print("s")
#sys.execute_schedule()
'''