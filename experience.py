import collections
import unittest

Experience = collections.namedtuple("Experience", \
    ["start_state", "action", "reward", "next_state", "done"])


class ExperienceSource:
    def __init__(self, env, agent, brain_name, steps, max_t):
        self.env = env
        self.agent = agent
        self.brain_name = brain_name
        self.steps = steps
        self.state = None
        self.memory = collections.deque(maxlen=steps)
        self.score = 0
        self.max_t = max_t

    def __iter__(self):
        self.score = 0
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.state = env_info.vector_observations[0]
        t = 0
        while t < self.max_t:
            start_state = self.state
            action = self.agent.act(self.state)
            env_info = self.env.step(action)[self.brain_name]
            reward = env_info.rewards[0]
            self.state = env_info.vector_observations[0]
            done = env_info.local_done[0]
            self.score += reward
            t += 1
            if done:
                self.state = None
            self.memory.append(Experience(start_state=start_state,\
                action=action,reward=reward,next_state=self.state,done=done))
            if done:
                yield tuple(self.memory)
                while len(self.memory) > 1:
                    self.memory.popleft()
                    yield tuple(self.memory)
                self.memory.popleft()
                break
            if len(self.memory) == self.steps:
                yield tuple(self.memory)

    def get_score(self):
        return self.score


class FirstAndLastExperienceSource:
    def __init__(self, underlying_source, gamma=1.0):
        self.underlying_source = underlying_source
        self.gamma = gamma

    def __iter__(self):
        for all_tuple in iter(self.underlying_source):
            reward_sum = 0
            for index, experience in enumerate(all_tuple):
                reward_sum += experience.reward * self.gamma ** index
            yield Experience(start_state=all_tuple[0].start_state,\
                action=all_tuple[0].action,reward=reward_sum,\
                next_state=all_tuple[-1].next_state, done=all_tuple[-1].done)

    def get_score(self):
        return self.underlying_source.score


class TestAgent:
    def act(self, state):
        return 0


class TestEnvInfo:
    def __init__(self, state, reward, done):
        self.vector_observations = [state]
        self.rewards = [reward]
        self.local_done = [done]


class TestEnv:
    def __init__(self, infos):
        self.infos = infos
        self.info_index = -1

    def step(self, action):
        self.info_index += 1
        return {"brain": self.infos[self.info_index]}

    def reset(self, train_mode=False):
        self.info_index = 0
        return {"brain": self.infos[self.info_index]}


class ExperienceSourceTests(unittest.TestCase):
    def test_it(self):
        env = TestEnv((TestEnvInfo(1, 0, False), TestEnvInfo(2, 3, False), TestEnvInfo(3, 1, False), TestEnvInfo(0, 0, True)))
        source = ExperienceSource(env, TestAgent(), "brain", 2)
        experience_count = 0
        expected_experience_map = {
            0: (Experience(start_state=1, action=0, reward=3, next_state=2, done=False), Experience(start_state=2, action=0, reward=1, next_state=3, done=False)),
            1: (Experience(start_state=2, action=0, reward=1, next_state=3, done=False), Experience(start_state=3, action=0, reward=0, next_state=None, done=True)),
            2: (Experience(start_state=3, action=0, reward=0, next_state=None, done=True),)
        }
        for experiences in source:
            expected_experiences = expected_experience_map[experience_count]
            self.assertEqual(experiences, expected_experiences)
            experience_count += 1

    def test_it_first_last(self):
        env = TestEnv(
            (TestEnvInfo(1, 0, False), TestEnvInfo(2, 3, False), TestEnvInfo(3, 1, False), TestEnvInfo(0, 0, True)))
        source = FirstAndLastExperienceSource(ExperienceSource(env, TestAgent(), "brain", 2), 0.99)
        experience_count = 0
        expected_experience_map = {
            0: Experience(start_state=1, action=0, reward=3.99, next_state=3, done=False),
            1: Experience(start_state=2, action=0, reward=1, next_state=None, done=True),
            2: Experience(start_state=3, action=0, reward=0, next_state=None, done=True)
        }
        for experiences in source:
            expected_experiences = expected_experience_map[experience_count]
            self.assertEqual(experiences, expected_experiences)
            experience_count += 1


if __name__ == '__main__':
    unittest.main()
