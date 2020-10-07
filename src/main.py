from system import System
from datetime import datetime, timedelta
import random
from task import TaskBuilder

sys = System()
sys.set_scheduler("genetic")
start = earliest_start = datetime.now() + timedelta(minutes=5)
sys._scheduler.start_time = start
tb = TaskBuilder()

random.seed()


for i in range(10):
    earliest_start = datetime.now() + timedelta(minutes=5)
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


#sys.schedule_tasks()
#sys.execute_schedule()


sys.schedule_tasks()
total_value = sys.simulate_schedule(start_time=start.timestamp())
print("Value: " + str(total_value))

#sys.execute_schedule()