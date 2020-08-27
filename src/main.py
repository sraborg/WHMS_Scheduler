from system import System
from task_builder import TaskBuilder
from scheduled_task import ScheduledTask
import time

sys = System()

tb = TaskBuilder()
tasklist = []
for i in range(10):
    tb.set_soft_deadline("Some Deadline")
    tb.set_hard_deadline(123)
    tb.set_analysis("duMmY")
    tb.add_dependencies(["t1", "t2"])
    tb.add_dynamic_tasks([("T1", lambda: True)])

    sys.add_task(tb.get_task())

sys.set_scheduler("DumMy")
sys.schedule_tasks()
sys.execute_schedule()