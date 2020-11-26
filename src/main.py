from system import System
from scheduler import SchedulerFactory, AbstractScheduler
from analysis import AnalysisFactory
from nu import NuFactory
from datetime import datetime, timedelta
import random
from task import TaskBuilder

sys = System()
start = earliest_start = datetime.now() + timedelta(minutes=5)
tb = TaskBuilder()

random.seed()


for i in range(10):
    earliest_start = datetime.now() + timedelta(minutes=20)
    soft_deadline = earliest_start + timedelta(minutes=random.randint(0, 10), seconds=random.randint(0, 30))
    hard_deadline = soft_deadline + timedelta(minutes=random.randint(0, 10), seconds=random.randint(0, 30))

    tb.set_earliest_start(earliest_start)
    tb.set_soft_deadline(soft_deadline)
    tb.set_hard_deadline(hard_deadline)
    tb._nu = NuFactory.regression()
    tb.fit_model([(earliest_start.timestamp(), 0), (soft_deadline.timestamp(), random.randint(0, 1000)), (hard_deadline.timestamp(), 0)])


    tb._analysis = AnalysisFactory.dummy_analysis()


    #tb.add_dependencies(["t1", "t2"])
    #tb.add_dynamic_tasks([("T1", lambda: True)])

    task = tb.build_task()
    if i > 0 and random.randint(0, 1) is 1:
        task.add_dependency(sys._tasks[i-1])
    sys.add_task(task)


sys.set_scheduler("genetic")
sys.scheduler = SchedulerFactory.genetic_scheduler(
    population_size=500,
    breeding_percentage=0.2,
    mutation_rate=0.05,
    elitism=True,
    max_iterations=10,
    threshold=0.01,
    generational_threshold=20,
    #start_time=None,
    verbose=False,
    invalid_schedule_value=-100.0,
)

#sys.scheduler.start_time = start
sys.schedule_tasks()
gen_sch = sys._schedule
total_gen_value = sys.simulate_schedule()#start_time=start.timestamp())
weighted_gen_value = sys.scheduler.weighted_schedule_value(gen_sch, total_gen_value)
'''

sys.scheduler = SchedulerFactory.ant_scheduler(
    colony_size=35,
    alpha=1,
    beta=1,
    epsilon=0.5,
    max_iterations=1,
    threshold=0.01,
    generational_threshold=5
)
sys.scheduler.start_time = start
sys.schedule_tasks()
ant_sch = sys._schedule
total_ant_value = sys.simulate_schedule()#start_time=start.timestamp())
weighted_ant_value = sys.scheduler.weighted_schedule_value(ant_sch, total_ant_value)

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

print("Ant Scheduler Value: " + str(total_ant_value))
print("Ant Weighted Scheduler Value: " + str(weighted_ant_value))
'''
print("Genetic Scheduler Value: " + str(total_gen_value))
print("Genetic Weighted Scheduler Value: " + str(weighted_gen_value))
'''
print("Random Scheduler Value: " + str(total_random_value))
print("Random Weighted Scheduler Value: " + str(weighted_random_value))
print("Simulated Annealing Scheduler Value " + str(total_anneal_value))
print("Simulated Weighted Annealing Scheduler Value " + str(weighted_anneal_value))
'''
sys.scheduler.save_tasklist("123seanl", sys._tasks)

loaded_tasks = sys.scheduler.load_tasklist("123xxx.csv")
print("s")
#sys.execute_schedule()
