from system import System
from dummy_task import DummyTask
from dummy_scheduler import DummyScheduler

system = System()

for i in range(10):
    system._taskset.add(DummyTask())

system._scheduler = DummyScheduler()

print('t')
