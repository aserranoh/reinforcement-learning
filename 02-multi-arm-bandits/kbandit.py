#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy
import random
from typing import Any, Callable
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
    
def generate_kbandit(n: int,
                     avg_reward: float = 0.0,
                     std_avg_reward: float = 1.0,
                     std_reward: float = 1.0) -> Kbandit:
    average_rewards = [random.gauss(avg_reward, std_avg_reward) for i in range(n)]
    return Kbandit(average_rewards, std_reward)

@dataclass
class KbanditRun:
    actions: list[int] = field(init=False, default_factory=list)
    rewards: list[float] = field(init=False, default_factory=list)
    best_actions: list[int] = field(init=False, default_factory=list)

    def action_taken(self, action: int, reward: float, best_action: float) -> None:
        self.actions.append(action)
        self.rewards.append(reward)
        self.best_actions.append(int(best_action))
    
    def average_reward(self):
        return numpy.cumsum(self.rewards) / numpy.arange(1, len(self.rewards) + 1)
    
    def best_action_rate(self):
        is_best_action = numpy.array(self.actions) == numpy.array(self.best_actions)
        return numpy.cumsum(is_best_action)/numpy.arange(1, len(self.actions) + 1)

class ActionStrategy(ABC):

    @abstractmethod
    def select_next_action(self) -> int:
        pass

    @abstractmethod
    def action_taken(self, action: int, reward: float) -> None:
        pass

@dataclass
class SampleAveragesStrategy(ActionStrategy):
    k: int
    averages: list[float] = field(init=False, repr=False)
    actions_count: list[int] = field(init=False, repr=False)
    step_size: float | None = None
    initial_estimates: list[float] | None = None
    t: int = field(init=False, default=1, repr=False)
    
    def __post_init__(self) -> None:
        self.actions_count = [0] * self.k
        if self.initial_estimates is not None:
            self.averages = [x for x in self.initial_estimates]
        else:
            self.averages = [0.0] * self.k

    def action_taken(self, action: int, reward: float) -> None:
        self.actions_count[action] += 1
        step_size = self.step_size
        if step_size is None:
            step_size = 1/self.actions_count[action]
        self.averages[action] = self.averages[action] + step_size*(reward - self.averages[action])
        self.t += 1

@dataclass
class EpsilonStrategy(SampleAveragesStrategy):
    epsilon: float = 0.0

    def select_next_action(self) -> int:
        dice = random.random()
        if dice > self.epsilon:
            action = numpy.argmax(self.averages)
        else:
            action = random.randrange(self.k)
        return int(action)

@dataclass
class UCBStrategy(SampleAveragesStrategy):
    c: float = 0.0

    def select_next_action(self) -> int:
        try:
            action = self.actions_count.index(0)
        except ValueError:
            action = numpy.argmax(numpy.array(self.averages) + self.c * numpy.log(self.t) / numpy.array(self.actions_count))
        return int(action)

@dataclass
class GradientStrategy(ActionStrategy):
    k: int
    step_size: float
    baseline: float | None = None
    h: Any = field(init=False)
    pr: Any = field(init=False)
    average_reward: float = field(init=False, default=0.0)
    count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.h = numpy.zeros(self.k)
        self.pr = numpy.exp(self.h) / sum(numpy.exp(self.h))

    def select_next_action(self) -> int:
        return numpy.random.choice(self.k, p=self.pr)

    def action_taken(self, action: int, reward: float) -> None:
        baseline = self.baseline
        if baseline is None:
            baseline = self.average_reward
            
        for i in range(self.k):
            error = reward - baseline
            if i == action:
                self.h[i] += self.step_size * error * (1 - self.pr[i])
            else:
                self.h[i] -= self.step_size * error * self.pr[i]

        self.pr = numpy.exp(self.h) / sum(numpy.exp(self.h))
        
        self.count += 1
        self.average_reward += (1/self.count) * (reward - self.average_reward)

@dataclass
class Runner:
    steps: int
    action_strategy: ActionStrategy
    
    def run(self, problem: Problem) -> KbanditRun:
        result = KbanditRun()
        for _ in range(self.steps):
            action = self.action_strategy.select_next_action()
            reward, best_action = problem.get_reward(action)
            self.action_strategy.action_taken(action, reward)
            result.action_taken(action, reward, best_action)
        return result

def run_problem(steps: int, n: int, problem_factory: Callable, strategy_factory: Callable) -> tuple:
    print(f"n: {n}, steps: {steps}")
    results = []
    for _ in tqdm(range(n)):
        results.append(Runner(steps=steps, action_strategy=strategy_factory()).run(problem_factory()))

    average_reward = numpy.stack([result.average_reward() for result in results])
    average_reward = numpy.average(average_reward, axis=0)

    best_action_rates = [result.best_action_rate() for result in results]
    best_action_rates = numpy.stack(best_action_rates)
    average_best_action = numpy.average(best_action_rates, axis=0)

    return average_reward, average_best_action