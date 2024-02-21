import numpy as np

class SimulationAnim:
    def __init__(self, animation_data):
        self.animation_data = animation_data
        self.chunk_size = 60  # Number of data points in each chunk
        self.num_chunks = len(animation_data) // self.chunk_size
        self.current_frame = 0
        self.data_generator = SimulationAnim.generate_data(self)

    def generate_data(self):
        for chunk_index in range(self.num_chunks):
            start_index = chunk_index * self.chunk_size
            end_index = (chunk_index + 1) * self.chunk_size
            chunk_data = self.animation_data.iloc[start_index:end_index]
            yield chunk_data

    def update(self):
        try:
            chunk_data = next(self.data_generator)
            self.current_frame += 1
            return chunk_data
        except StopIteration:
            return None
        