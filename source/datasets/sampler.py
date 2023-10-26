import numpy as np


def sampler(data: np.ndarray, window_length: int, samples_per_window: int) -> np.ndarray:
    assert window_length / samples_per_window % 1 == 0

    _, T, V, C = data.shape
    if T < window_length:
        # data is shorter than window
        # position the data
        new_data = np.zeros((*data.shape[:-3], window_length, V, C), dtype=np.float32)
        t = window_length - T
        pos = np.random.randint(0, t)
        new_data[..., pos:pos + T, :, :] = data
        xd = new_data[0, :, 0, 0]
        # Fill missing
        new_data[..., :pos, :, :] = data[..., 0, np.newaxis, :, :]
        new_data[..., pos + T:, :, :] = data[..., -1, np.newaxis, :, :]
        data = new_data
        T = window_length

    start_indice = np.random.randint(0, T - window_length + 1)
    end_indice = start_indice + window_length
    jump = window_length // samples_per_window
    return data[..., start_indice:end_indice:jump, :, :]
