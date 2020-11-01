from system import System
from scheduler import SchedulerFactory
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
    tb.set_nu("Regression")
    tb.fit_model([(earliest_start.timestamp(), 0), (soft_deadline.timestamp(), random.randint(0, 1000)), (hard_deadline.timestamp(), 0)])


    tb.set_analysis("duMmY")


    #tb.add_dependencies(["t1", "t2"])
    tb.add_dynamic_tasks([("T1", lambda: True)])

    task = tb.build_task()
    if i > 0 and random.randint(0, 1) is 1:
        task.add_dependency(sys._tasks[i-1])
    sys.add_task(task)

'''
sys.set_scheduler("random")
sys._scheduler.start_time = start
sys.schedule_tasks()
total_value = sys.simulate_schedule(start_time=start.timestamp())
print("Random Scheduler Value: " + str(total_value))
'''
sys.set_scheduler("genetic")
sys.scheduler = SchedulerFactory.genetic_scheduler(
    population_size=500,
    breeding_percentage=0.2,
    mutation_rate=0.05,
    elitism=True,
    max_iterations=1000,
    threshold=0.02,
    generational_threshold=20,
    start_time=None,
    verbose=False,
    invalid_schedule_value=-100.0,
)

sys.scheduler.start_time = start
sys.schedule_tasks()
gen_sch = sys._schedule
total_gen_value = sys.simulate_schedule(start_time=start.timestamp())


sys.scheduler = SchedulerFactory.ant_scheduler(
    colony_size=20,
    alpha=2,
    beta=1,
    epsilon=0.4,
    max_iterations=4,
    threshold=0.3,
    generational_threshold=3
)
sys.scheduler.start_time = start
sys.schedule_tasks()
ant_sch = sys._schedule
total_ant_value = sys.simulate_schedule(start_time=start.timestamp())


print("Genetic Scheduler Value: " + str(total_gen_value))
print("Ant Scheduler Value: " + str(total_ant_value))

#sys.execute_schedule()
