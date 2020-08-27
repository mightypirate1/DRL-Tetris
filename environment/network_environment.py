import time

import pySFMLnet as p
from environment.tetris_environment_vector import tetris_environment_vector

ip       = "127.0.0.1"
udp_port = 6666
tcp_port = 6667

client_send_code = {"ready" : 7, "new_room" : 11, "login" : 2, "verify_udp" : 99, "game_over" : 3,}

class network_environment(tetris_environment_vector):
    def __init__(self, n_envs, env_type, init_envs=None, settings=None):
        tetris_vector_environment.__init__(1, settings["env_type"], settings=settings)
        self.init_networking()
        self.env = self.envs[0]
        self.playing = False
        self.id = None #This should be filled in by login handshake
        self.opponent_bpm = 75
        self.piece_delay = 800 #60000 / 75
        self.last_piece_placed = 0
        self.done = False
    #Frontend
    def perform_action(self, *args, **kwargs):
        while time.time() < self.last_piece_placed + self.piece_delay:
            time.sleep(0.01)
        self.last_piece_placed = time.time()
        rs,ds = tetris_environment_vector.perform_action(*args,**kwargs)
        self.broadcast_state(self.env.get_state(), cleared=rs[0])
        return rs, [self.done, self.done]
    def get_state(self, *args, **kwargs):
        self.update_state(self.envs[0])
        return self.get_state(*args, **kwargs)
    def reset(self, *args, **kwargs):
        tetris_vector_environment.reset(*args, **kwargs)
        self.block_until_new_round()

    #Backend
    def init_networking(self):
        p.init(udp_port,tcp_port) #Setup ports
        self.send_signal("login")
        while self.id is None:
            #expecting a 0-code packet, which contains our id
            self.receive_all_tcp()
            time.sleep(0.1)
        self.send_signal("verify_udp")
        self.send_signal("new_room")

    def broadcast_state(self, state, cleared=0):
        s = state.backend_state.compress_state(0)
        p.send_state(self.id, s)
        self.send_linedata(state.backend_state.states[0].send_lines, cleared) #lines sent, lines cleared
        if state.backend_state.states[0].dead:
            self.send_signal("am_dead")

    def update_state(self, env):
        while p.receive_udp():
            newstate = p.read_state()
            self.env.backend.extract_state(1,newstate)
        self.receive_all_tcp()

    def block_until_new_round(self):
        print("dbg: blocking until new round")
        self.send_signal("ready")
        while not self.playing:
            self.receive_all_tcp()
            time.sleep(1)

    def receive_all_tcp(self):
        while p.receive_tcp():
            code = p.get_tcp_code()
            self.receive_tcp(code)

    def receive_tcp(self, code):
        print("dbg: received tcp-package", code)
        if code == 0:
            #Server says hi!
            self.id = p.read_uint16()
            some_string = p.read_string()
            print(some_string, " <- message from server during handshake")
        if code == 7:
            #End of round!
            print("server end of round signal", code)
            self.playing = False
        if code == 8:
            #I won!
            print("server i win signal", code)
            self.playing = False
            self.done = True
        if code == 9:
            print("incoming lines")
            #Incoming lines!
            pass #Do we need to do something
        if code == 10: #New round!
            print("reseeding env")
            self.playing = True
            seed1, seed2 = p.read_uint16(), p.read_uint16()
            self.env.backend.reset()
            self.env.backend.set_seed(seed1, seed2)
        if code == 23:
            # 23 - Sending players average bpm for round, id1=player id, id2=avg_bpm
            _, self.opponent_bpm = p.read_uint16(), p.read_uint16()
            self.piece_delay = 60000 / self.opponent_bpm
            print("setting bpm", self.opponent_bpm)

    def send_signal(self, key):
        code = client_send_code[key]
        p.clear()
        p.write_uint8(code)
        if key == "login":
            p.write_uint16(666) #version
            p.write_uint8(1)    #guest
            p.write_string(self.settings["run-id"]) #name
        if key == "verify_udp":
            p.write_uint16(self.id)
        if key == "new_room":
            p.write_string("bot_room") #name
            p.write_uint8(2) #n_players
        if key == "ready":
            pass #No info needed?
        if key == "game_over":
            p.write_uint8(0)
            p.write_uint16(0)
            p.write_uint16(0)
            p.write_uint16(0)
            p.write_uint16(0)#above and this line is "roundscore".. figure this out :)
            p.write_uint32(0)#duration
            p.write_uint16(0)#piece count
        p.sendTCP()

    def send_linedata(self, send_lines, cleared_lines):
        p.clear()
        p.write_uint8(2) #lines sent
        p.write_uint8(send_lines)
        p.sendTCP()
        p.clear()
        p.write_uint8(3) #lines cleared
        p.write_uint8(cleared_lines)
        p.sendTCP()
