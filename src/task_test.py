from task_builder import TaskBuilder

tb = TaskBuilder()
tb.set_deadline("Some Deadline")
tb.set_analysis("duMmY")
tb.add_dependencies(["t1", "t2"])
tb.add_dynamic_tasks(["t1", "t2"])
task = tb.get_task()

print(task)