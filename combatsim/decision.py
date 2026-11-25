"""Shared helpers for holding actions between decisions."""

def held_action(action):
    if isinstance(action,(tuple,list)):
        return (action[0],action[1],0)
    # Legacy packed action uses groups of three turns, then the fire flag.
    return action-3 if action%6 >= 3 else action


class DecisionScheduler:
    def __init__(self, interval):
        self.interval = interval
        self.actions = {}

    def action(self, ident, tick, decide):
        if ident not in self.actions or tick % self.interval == 0:
            self.actions[ident] = decide()
            return self.actions[ident]
        return held_action(self.actions[ident])

    def policy_action(self, policy, observation, tick):
        return self.action(observation['self']['id'],tick,lambda:policy.act(observation))
