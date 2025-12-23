"""
Base mode of world model
"""
from abc import ABC, abstractmethod

class WorldModelBase(ABC):
    @abstractmethod
    def add_experiment(self, experiment):
        pass

    @abstractmethod
    def update_with_insights(self, insights):
        pass
