from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage
from uuid import UUID


class ListSideChannel(SideChannel):
    def __init__(self):
        super().__init__(UUID('12345678-1234-5678-1234-567812345678'))

    def send_list(self, values):
        msg = OutgoingMessage()
        msg.write_int32(len(values))
        for val in values:
            msg.write_float32(val)
        self.queue_message_to_send(msg)

    def on_message_received(self, msg: IncomingMessage):
        list_len = msg.read_int32()
        values = [msg.read_float32() for _ in range(list_len)]