from enum import Enum
from samplelib import SampleGeneratorBase


class Paddle(Enum):
    PING = 'ping'  # Ascending
    PONG = 'pong'  # Descending


class PingPongOptions:
    def __init__(self, enabled=False, iterations=1000, model_iter=1, paddle=Paddle.PING, batch_cap=1):
        self.enabled = enabled
        self.iterations = iterations
        self.model_iter = model_iter
        self.paddle = paddle
        self.batch_cap = batch_cap


class SampleGeneratorPingPong(SampleGeneratorBase):
    def __init__(self, *args, batch_size, ping_pong=PingPongOptions()):
        self.ping_pong = ping_pong
        super().__init__(*args, batch_size)

    def __next__(self):
        if self.ping_pong.enabled and self.ping_pong.model_iter % self.ping_pong.iterations == 0 \
                and self.ping_pong.model_iter != 0:

            # If batch size is greater then batch cap, set it to batch cap
            if self.batch_size > self.ping_pong.batch_cap:
                self.batch_size = self.ping_pong.batch_cap

            # If we are at the batch cap, switch to PONG (descend)
            if self.batch_size == self.ping_pong.batch_cap:
                self.paddle = Paddle.PONG
            # Else if we are at 1, switch to PING (ascend)
            elif self.batch_size == 1:
                self.paddle = Paddle.PING

            # If PING (ascending) increase the batch size
            if self.paddle is Paddle.PING:
                self.batch_size += 1
            # Else decrease the batch size
            else:
                self.batch_size -= 1

        self.ping_pong.model_iter += 1
        super().__next__()



