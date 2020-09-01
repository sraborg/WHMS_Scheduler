from system import System
from datetime import datetime, timedelta
import random
from task import TaskBuilder

sys = System()

tb = TaskBuilder()

random.seed()

for i in range(10):
    earliest_start = datetime.now() + timedelta(seconds=random.randint(0, 10))
    soft_deadline = earliest_start + timedelta(seconds=random.randint(0, 10))
    hard_deadline = earliest_start + timedelta(seconds=random.randint(0, 10))

    tb.set_earliest_start(earliest_start)
    tb.set_soft_deadline(soft_deadline)
    tb.set_hard_deadline(hard_deadline)
    tb.set_nu("Regression")
    tb.fit_model([(earliest_start.timestamp(), 0), (soft_deadline.timestamp(), random.randint(0, 1000)), (hard_deadline.timestamp(), 0)])

    tb.set_analysis("duMmY")
    tb.add_dependencies(["t1", "t2"])
    tb.add_dynamic_tasks([("T1", lambda: True)])

    sys.add_task(tb.build_task())

sys.set_scheduler("DumMy")
sys.schedule_tasks()
sys.execute_schedule()