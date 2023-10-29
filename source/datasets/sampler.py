import numpy as np


def legacy_sampler(data: np.ndarray, window_length: int, samples_per_window: int) -> np.ndarray:
    assert window_length / samples_per_window % 1 == 0

    _, T, V, C = data.shape
    if T < window_length:
        # data is shorter than window
        # position the data
        new_data = np.zeros((*data.shape[:-3], window_length, V, C), dtype=np.float32)
        t = window_length - T
        pos = np.random.randint(0, t)
        new_data[..., pos:pos + T, :, :] = data
        # Fill missing
        new_data[..., :pos, :, :] = data[..., 0, np.newaxis, :, :]
        new_data[..., pos + T:, :, :] = data[..., -1, np.newaxis, :, :]
        data = new_data
        T = window_length

    start_indice = np.random.randint(0, T - window_length + 1)
    end_indice = start_indice + window_length
    jump = window_length // samples_per_window
    return data[..., start_indice:end_indice:jump, :, :]


class Sampler:
    def __init__(self, window_length: int, samples_per_window: int, test_mode: bool = False, clips: int = 1,
                 seed: int = 255):
        self.window_length = window_length
        self.samples_per_window = samples_per_window
        self.clips = clips
        self.seed = seed
        self.test = test_mode
        assert window_length / samples_per_window % 1 == 0

        self.min_size_test = max(self.clips + self.window_length - 1, int(self.window_length * 1.5))

    def __call__(self, data: np.ndarray):
        if self.test:
            return self.test_sample(data)
        else:
            return self.sample(data)

    def test_sample(self, data: np.ndarray):
        # Multiple clips at predictable spots
        np.random.seed(self.seed)
        jump = self.window_length // self.samples_per_window
        outputs = []
        _, T, V, C = data.shape

        if T < self.min_size_test:
            le = left_extend = (self.min_size_test - T) // 2
            new_data = np.zeros((*data.shape[:-3], self.min_size_test, V, C), dtype=np.float32)
            # Copy data to new array
            new_data[..., le:le + T, :, :] = data
            # Fill left side with first value of original data
            new_data[..., :le, :, :] = data[..., 0, np.newaxis, :, :]
            # Fill right side with last value of original data
            new_data[..., le + T:, :, :] = data[..., -1, np.newaxis, :, :]
            data = new_data
            T = self.min_size_test

        avail_position_count = T - self.window_length + 1 + (jump - 1)
        positions = equally_spaced_positions(avail_position_count, self.clips)

        for start_index in positions:
            end_index = start_index + self.window_length
            outputs.append(data[..., start_index:end_index:jump, :, :])

        return outputs

    def sample(self, data: np.ndarray):
        # Single clip at random space
        jump = self.window_length // self.samples_per_window
        _, T, V, C = data.shape
        min_size = int(self.window_length * 1.5)
        if T < min_size:
            le = left_extend = (min_size - T) // 2
            new_data = np.zeros((*data.shape[:-3], min_size, V, C), dtype=np.float32)
            # Copy data to new array
            new_data[..., le:le + T, :, :] = data
            # Fill left side with first value of original data
            new_data[..., :le, :, :] = data[..., 0, np.newaxis, :, :]
            # Fill right side with last value of original data
            new_data[..., le + T:, :, :] = data[..., -1, np.newaxis, :, :]
            data = new_data
            T = min_size

        start_index = np.random.randint(0, T - self.window_length + 1 + (jump - 1))
        end_index = start_index + self.window_length

        return data[..., start_index:end_index:jump, :, :]


def equally_spaced_positions(n, v):
    if v == 1:
        return np.array([0], dtype=int)  # Special case when selecting only one position.

    step_size = (n - 1) / (v - 1)
    selected_positions = np.round(np.arange(v) * step_size).astype(int)

    return selected_positions
