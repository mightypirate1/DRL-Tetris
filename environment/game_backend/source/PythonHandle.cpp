#include "PythonHandle.h"
#include "packet_compress.hpp"

#include <iostream>

PythonHandle PythonHandle::init(int number_of_players, std::array<int, 2> field_size) {
    PythonHandle ph;
    FIELD_HEIGHT = field_size[0];
    FIELD_WIDTH = field_size[1];
    ph.players.resize(number_of_players);
    ph.masks.resize(number_of_players);
    ph.states.reserve(number_of_players);

    if (number_of_players > 1) ph.check_for_winner = true;

    for (auto& player : ph.players) {
        player.restartRound();
        ph.states.emplace_back(player);
    }
    ph.seed();

    // Pickle-proofed?
    ph.pickle_proof = false;

    return ph;
}

void PythonHandle::recreate_state() {
    if (pickle_proof) return;
    states.clear();
    for (uint8_t i = 0; i < players.size(); ++i) {
        states.emplace_back(players[i]);
    }
    pickle_proof = true;
}

PythonHandle PythonHandle::copy() {
    PythonHandle new_handle = init(players.size(), {FIELD_HEIGHT, FIELD_WIDTH});
    new_handle = *this;
    return new_handle;
}

void PythonHandle::set(const PythonHandle& other) { *this = other; }

// pybind11::tuple PythonHandle::seabass(int p){
// 		auto &player = players[p];
// 		return pybind11::make_tuple(player.field.square);
// }

void PythonHandle::reset() {
    round_over = false;
    int player_count = -1;
    int winner = -1;
    int alive_count = 0;
    for (auto& player : players) {
        ++player_count;
        if (!player.dead) {
            ++alive_count;
            winner = player_count;
        }
        player.restartRound();
    }
    last_winner = winner;
    if (!player_count) last_winner = 0;
    if (alive_count > 1) last_winner = -1;
    seed();
}

void PythonHandle::re_seed(unsigned long long seed1, unsigned long long seed2) {
    for (auto& player : players) player.seed(seed1, seed2);
}
void PythonHandle::seed() {
    unsigned long long seed1 = time(NULL), seed2 = time(NULL);
    for (auto& player : players) player.seed(seed1, seed2);
}

void PythonHandle::action_make(int player, int action) {
    GamePlay& game = players[player];
    switch (action) {
        case 1:
            game.field.mLeft();
            break;
        case 2:
            while (game.field.mLeft())
                ;
            break;
        case 3:
            game.field.mRight();
            break;
        case 4:
            while (game.field.mRight())
                ;
            break;
        case 5:
            game.mDown();
            break;
        case 6:
            while (game.mDown())
                ;
            break;
        case 7:
            game.hd_make();
            break;
        case 8:
            game.field.rcw();
            break;
        case 9:
            game.field.rccw();
            break;
        case 10:
            game.field.r180();
            break;
        default:
            break;
    }
}

bool PythonHandle::action_finish(int player) {
    int sent = players[player].hd_finish();
    if (sent == -1)  // Player died
        return true;
    else if (!sent)  // Nothing sent
        return false;
    players[player].send_lines += sent;
    distributeLines(player, sent);
}

void PythonHandle::distributeLines(int sender, int amount) {
    float number_of_players = players.size() - 1;
    if (number_of_players < 1) return;
    float amount_per_player = amount / number_of_players;

    int count = -1;
    for (auto& player : players) {
        ++count;
        if (count == sender) continue;

        player.incoming_lines += amount_per_player;
    }
}

bool PythonHandle::make_actions(std::vector<std::vector<int>> actions) {
    if (round_over) return true;
    for (unsigned i = 0; i < players.size(); ++i) {
        if (!players[i].dead) {
            for (auto p_action : actions[i]) {
                action_make(i, p_action);
            }
        }
    }
}

bool PythonHandle::finish_actions(int time_elapsed) {
    if (round_over) return true;
    for (unsigned i = 0; i < players.size(); ++i) {
        if (!players[i].dead) {
            if (action_finish(i)) {
                players[i].dead = true;
                break;
            }
        }
    }

    int player_count = -1;
    int alive_count = 0;
    for (auto& player : players) {
        ++player_count;

        if (player.dead) continue;

        int sent = player.delayCheck(time_elapsed);
        player.send_lines += sent;
        if (sent == -1) {
            player.dead = true;
            masks[player_count].clear();
            continue;
        } else if (sent)
            distributeLines(player_count, sent);
        if (!player.dead) alive_count++;

        player.reward = player.data.linesCleared - player.linesCleared;
        player.linesCleared = player.data.linesCleared;

        player.incoming_lines_count = player.garbage.count();
    }

    if ((check_for_winner && alive_count < 2) || !alive_count) {
        round_over = true;
        return true;
    }

    return false;
}

void PythonHandle::get_actions(int p) { masks[p] = players[p].getMask(2); }

std::vector<uint8_t> PythonHandle::compress_state(int p){
  PacketCompress pc;
  std::size_t m = pc.square.size();
  std::size_t n = pc.square[0].size();
  for (uint8_t i=0; i<m;i++){
    for (uint8_t j=0;j<n;j++){
      pc.square[i][j] = players[p].field.square[i*m+j];
    }
  }
  // pc.square = players[p].field.square;
  pc.posX = players[p].field.piece.posX;
  pc.posY = players[p].field.piece.posY;
  pc.piece = players[p].field.piece.piece;
  pc.color = players[p].field.piece.tile;
  pc.rotation = players[p].field.piece.current_rotation;
  pc.nextpiece = players[p].nextpiece;
  pc.npcol = players[p].nextpiece;
  pc.nprot = 0;
  pc.comboText = players[p].combo.comboCount;
  pc.pendingText = players[p].incoming_lines_count;
  pc.bpmText = 0;
  pc.comboTimerCount = players[p].combo.remaining;
  pc.countdown = 0;
  pc.time_val = 0;
  pc.compress();
  return pc.m_data;
}

void PythonHandle::extract_state(int p, std::vector<uint8_t> data){
  PacketCompress pc;
  pc.loadTmp(data);
  pc.extract();
  std::size_t m = pc.square.size();
  std::size_t n = pc.square[0].size();
  for (uint8_t i=0; i<m;i++){
    for (uint8_t j=0;j<n;j++){
      players[p].field.square[i*m+j] = pc.square[i][j];
    }
  }
  // players[p].field.square = pc.square;
  players[p].field.piece.posX = pc.posX;
  players[p].field.piece.posY = pc.posY;
  players[p].field.piece.piece = pc.piece;
  players[p].field.piece.tile = pc.color;
  players[p].field.piece.current_rotation = pc.rotation;
  players[p].nextpiece = pc.nextpiece;
  players[p].nextpiece = pc.npcol;
  players[p].combo.comboCount = pc.comboText;
  players[p].incoming_lines_count = pc.pendingText;
  players[p].combo.remaining = pc.comboTimerCount;

}
