from task_builder import TaskBuilder

tb = TaskBuilder()
tasklist = []
for i in range(10):
    tb.set_deadline("Some Deadline")
    tb.set_analysis("duMmY")
    tb.add_dependencies(["t1", "t2"])
    tb.add_dynamic_tasks([("T1", lambda: True)])
    task = tb.get_task()
    tasklist.append(task)

for task in tasklist:
    task.execute()
