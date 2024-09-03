#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy
import random
import sys
from typing import Callable

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

    def __post_init__(self) -> None:
        self.actions_count = [0] * self.k
        self.averages = [0.0] * self.k

    def action_taken(self, action: int, reward: float, best_action: float) -> None:
        self.actions_count[action] += 1
        self.averages[action] = self.averages[action] + 1/self.actions_count[action]*(reward - self.averages[action])
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
    
    def run(self, problem: Problem) -> KbanditRun:
        result = KbanditRun(len(problem))

        for _ in range(self.steps):
            dice = random.random()

            if dice > self.epsilon:
                action = numpy.argmax(result.averages)
            else:
                action = random.randrange(len(problem))
            reward, best_action = problem.get_reward(int(action))

            result.action_taken(int(action), reward, best_action)
        return result

def run_problem(steps: int, n: int, epsilon: float, problem_factory: Callable) -> tuple:
    print(f"Epsilon: {epsilon}")
    problems = [problem_factory() for _ in range(n)]
    results = [Runner(steps=steps, epsilon=epsilon).run(problem) for problem in problems]

    average_reward = numpy.stack([result.average_reward() for result in results])
    average_reward = numpy.average(average_reward, axis=0)

    best_action_rates = [result.best_action_rate() for result in results]
    best_action_rates = numpy.stack(best_action_rates)
    average_best_action = numpy.average(best_action_rates, axis=0)

    return average_reward, average_best_action

def plot(rewards: list[numpy.array], best_actions: list[numpy.array], epsilons: list[float]) -> None:
    plt.figure()
    plt.subplot(211)
    for reward, epsilon in zip(rewards, epsilons):
        plt.plot(reward, label=f"Epsilon = {epsilon}")
    plt.subplot(212)
    for best_action, epsilon in zip(best_actions, epsilons):
        plt.plot(best_action, label=f"Epsilon = {epsilon}")
    plt.legend()
    plt.show()

def kbandit_generic(factory: Callable) -> None:
    rewards, best_actions = [], []
    epsilons = (0.0, 0.1, 0.01)
    for epsilon in epsilons:
        reward, best_action = run_problem(1000, 2000, epsilon, factory)
        rewards.append(reward)
        best_actions.append(best_action)
    plot(rewards, best_actions, epsilons)

def kbandit() -> None:
    kbandit_generic(lambda: generate_kbandit(10))

def kbandit_ns() -> None:
    kbandit_generic(lambda: KbanditNonStationary([0.0]*10))

def main() -> None:
    if sys.argv[1] == "kbandit":
        kbandit()
    else:
        kbandit_ns()

if __name__ == "__main__":
    main()