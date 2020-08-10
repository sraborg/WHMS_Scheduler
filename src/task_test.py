from dummy_task import DummyTask
from task_with_dynamic_tasks import TaskWithDynamicTasks
from task_with_dependencies import TaskWithDependencies


def nu(x):
    return x+1

t1 = TaskWithDependencies(TaskWithDynamicTasks(TaskWithDynamicTasks(DummyTask())))

t1.dynamic_tasks.append(("object", "test"))
t1.dependent_tasks.append("junk")

print(t1)