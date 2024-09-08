#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy
import random
from typing import Callable
from tqdm import tqdm

class Problem(ABC):

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def get_reward(self, action: int) -> tuple[float, float]:
        pass

@dataclass
class Kbandit(Problem):
    avg_rewards: list[float]
    std_reward: float
    
    def __len__(self) -> int:
        return len(self.avg_rewards)
    
    def get_reward(self, action: int) -> tuple[float, float]:
        reward = random.gauss(self.avg_rewards[action], self.std_reward)
        best_action = numpy.argmax(self.avg_rewards)
        return reward, int(best_action)
    
@dataclass
class KbanditNonStationary(Problem):
    rewards: list[float]
    mean_increment: float = 0.0
    std_increment: float = 0.01
    
    def __len__(self) -> int:
        return len(self.rewards)
    
    def get_reward(self, action: int) -> tuple[float, float]:
        reward = self.rewards[action]
        best_action = numpy.argmax(self.rewards)
        self.rewards = [reward + random.gauss(self.mean_increment, self.std_increment) for reward in self.rewards]
        return reward, int(best_action)
    
def generate_kbandit(n: int, avg_reward: float = 0.0, std_avg_reward: float = 1.0, std_reward: float = 1.0) -> Kbandit:
    average_rewards = [random.gauss(avg_reward, std_avg_reward) for i in range(n)]
    return Kbandit(average_rewards, std_reward)

@dataclass
class KbanditRun:
    k: int
    actions: list[int] = field(init=False, default_factory=list)
    rewards: list[float] = field(init=False, default_factory=list)
    best_actions: list[int] = field(init=False, default_factory=list)
    averages: list[float] = field(init=False)
    actions_count: list[int] = field(init=False)
    step_size: float | None = None

    def __post_init__(self) -> None:
        self.actions_count = [0] * self.k
        self.averages = [0.0] * self.k

    def action_taken(self, action: int, reward: float, best_action: float) -> None:
        self.actions_count[action] += 1
        step_size = self.step_size
        if step_size is None:
            step_size = 1/self.actions_count[action]
        self.averages[action] = self.averages[action] + step_size*(reward - self.averages[action])
        self.actions.append(action)
        self.rewards.append(reward)
        self.best_actions.append(int(best_action))
    
    def average_reward(self):
        return numpy.cumsum(self.rewards) / numpy.arange(1, len(self.rewards) + 1)
    
    def best_action_rate(self):
        is_best_action = numpy.array(self.actions) == numpy.array(self.best_actions)
        return numpy.cumsum(is_best_action)/numpy.arange(1, len(self.actions) + 1)

@dataclass
class Runner:
    steps: int
    epsilon: float = 0.0
    step_size: float | None = None
    
    def run(self, problem: Problem) -> KbanditRun:
        result = KbanditRun(len(problem), step_size=self.step_size)

        for _ in range(self.steps):
            dice = random.random()

            if dice > self.epsilon:
                action = numpy.argmax(result.averages)
            else:
                action = random.randrange(len(problem))
            reward, best_action = problem.get_reward(int(action))

            result.action_taken(int(action), reward, best_action)
        return result

def run_problem(steps: int, n: int, epsilon: float, problem_factory: Callable, step_size: float | None = None) -> tuple:
    print(f"n: {n}, steps: {steps}, epsilon: {epsilon}")
    problems = [problem_factory() for _ in range(n)]
    results = []
    for problem, _ in zip(problems, tqdm(range(n))):
        results.append(Runner(steps=steps, epsilon=epsilon, step_size=step_size).run(problem))

    average_reward = numpy.stack([result.average_reward() for result in results])
    average_reward = numpy.average(average_reward, axis=0)

    best_action_rates = [result.best_action_rate() for result in results]
    best_action_rates = numpy.stack(best_action_rates)
    average_best_action = numpy.average(best_action_rates, axis=0)

    return average_reward, average_best_action