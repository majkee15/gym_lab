from collections import deque, namedtuple
import numpy as np
import itertools

from utils.segment_tree import SumSegmentTree, MinSegmentTree, SegmentTree

# This is the default buffer record nametuple type.
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_next', 'done'])


class ReplayMemory:
    def __init__(self, capacity=100000, replace=False, tuple_class=Transition):
        self.buffer = []
        self.capacity = capacity
        self.replace = replace
        self.tuple_class = tuple_class
        self.fields = tuple_class._fields

    def add(self, record):
        """Any named tuple item."""
        if isinstance(record, self.tuple_class):
            self.buffer.append(record)
        elif isinstance(record, list):
            self.buffer += record
        else:
            raise NotImplementedError('Has to be a tuple class.')

        while self.capacity and self.size > self.capacity:
            self.buffer.pop(0)

    def _reformat(self, indices):
        # Reformat a list of Transition tuples for training.
        # indices: list<int>
        return {
            field_name: np.array([getattr(self.buffer[i], field_name) for i in indices])
            for field_name in self.fields
        }

    def sample(self, batch_size):
        assert len(self.buffer) >= batch_size
        idxs = np.random.choice(range(len(self.buffer)), size=batch_size, replace=self.replace)
        return self._reformat(idxs)

    def pop(self, batch_size):
        # Pop the first `batch_size` Transition items out.
        i = min(self.size, batch_size)
        batch = self._reformat(range(i))
        self.buffer = self.buffer[i:]
        return batch

    def loop(self, batch_size, epoch=None):
        indices = []
        ep = None
        for i in itertools.cycle(range(len(self.buffer))):
            indices.append(i)
            if i == 0:
                ep = 0 if ep is None else ep + 1
            if epoch is not None and ep == epoch:
                break

            if len(indices) == batch_size:
                yield self._reformat(indices)
                indices = []

    @property
    def size(self):
        return len(self.buffer)

#
# class PrioritizedReplayMemory(ReplayMemory):
#
#     def __init__(self, alpha=0.6, capacity=100000, replace=False, tuple_class=Transition):
#         super().__init__(capacity, replace, tuple_class)
#         assert alpha >= 0
#         self.max_priority, self.tree_ptr = 1.0, 0
#         self.alpha = alpha
#         # capacity must be positive and a power of 2.
#         tree_capacity = 1
#         while tree_capacity < self.capacity:
#             tree_capacity *= 2
#
#         self.sum_tree = SumSegmentTree(tree_capacity)
#         self.min_tree = MinSegmentTree(tree_capacity)
#
#     def add(self, record):
#         super().add(record)
#         self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
#         self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
#         self.tree_ptr = (self.tree_ptr + 1) % self.capacity
#
#     def sample(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
#         """Sample a batch of experiences."""
#         assert len(self) >= self.batch_size
#         assert beta > 0
#
#         indices = self._sample_proportional()
#
#         obs = self.obs_buf[indices]
#         next_obs = self.next_obs_buf[indices]
#         acts = self.acts_buf[indices]
#         rews = self.rews_buf[indices]
#         done = self.done_buf[indices]
#         weights = np.array([self._calculate_weight(i, beta) for i in indices])
#
#         return dict(
#             obs=obs,
#             next_obs=next_obs,
#             acts=acts,
#             rews=rews,
#             done=done,
#             weights=weights,
#             indices=indices,
#         )
#
#     def update_priorities(self, indices: List[int], priorities: np.ndarray):
#         """Update priorities of sampled transitions."""
#         assert len(indices) == len(priorities)
#
#         for idx, priority in zip(indices, priorities):
#             assert priority > 0
#             assert 0 <= idx < len(self)
#
#             self.sum_tree[idx] = priority ** self.alpha
#             self.min_tree[idx] = priority ** self.alpha
#
#             self.max_priority = max(self.max_priority, priority)
#
#     def _sample_proportional(self) -> List[int]:
#         """Sample indices based on proportions."""
#         indices = []
#         p_total = self.sum_tree.sum(0, len(self) - 1)
#         segment = p_total / self.batch_size
#
#         for i in range(self.batch_size):
#             a = segment * i
#             b = segment * (i + 1)
#             upperbound = random.uniform(a, b)
#             idx = self.sum_tree.retrieve(upperbound)
#             indices.append(idx)
#
#         return indices
#
#     def _calculate_weight(self, idx: int, beta: float):
#         """Calculate the weight of the experience at idx."""
#         # get max weight
#         p_min = self.min_tree.min() / self.sum_tree.sum()
#         max_weight = (p_min * len(self)) ** (-beta)
#
#         # calculate weights
#         p_sample = self.sum_tree[idx] / self.sum_tree.sum()
#         weight = (p_sample * len(self)) ** (-beta)
#         weight = weight / max_weight
#
#         return weight


class ReplayTrajMemory:
    def __init__(self, capacity=100000, step_size=16):
        self.buffer = deque(maxlen=capacity)
        self.step_size = step_size

    def add(self, traj):
        # traj (list<Transition>)
        if len(traj) >= self.step_size:
            self.buffer.append(traj)

    def sample(self, batch_size):
        traj_idxs = np.random.choice(range(len(self.buffer)), size=batch_size, replace=True)
        batch_data = {field_name: [] for field_name in Transition._fields}

        for traj_idx in traj_idxs:
            i = np.random.randint(0, len(self.buffer[traj_idx]) + 1 - self.step_size)
            transitions = self.buffer[traj_idx][i: i + self.step_size]

            for field_name in Transition._fields:
                batch_data[field_name] += [getattr(t, field_name) for t in transitions]

        assert all(len(v) == batch_size * self.step_size for v in batch_data.values())
        return {k: np.array(v) for k, v in batch_data.items()}

    @property
    def size(self):
        return len(self.buffer)

    @property
    def transition_size(self):
        return sum(map(len, self.buffer))