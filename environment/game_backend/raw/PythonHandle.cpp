#include "PythonHandle.h"
#include <iostream>

PythonHandle PythonHandle::init(int number_of_players, std::array<int, 2> field_size, int mask) {
	PythonHandle ph;
	FIELD_HEIGHT = field_size[0];
	FIELD_WIDTH = field_size[1];
	ph.players.resize(number_of_players);
	ph.states.reserve(number_of_players);

	if (number_of_players > 1)
		ph.check_for_winner = true;

	ph.use_mask = mask;

	for (auto& player : ph.players) {
		player.restartRound();
		ph.states.emplace_back(player);
	}
	ph.seed();

	if (ph.use_mask)
		for (auto& player : ph.players)
			ph.masks.push_back(player.getMask(ph.use_mask));

	return ph;
}

PythonHandle PythonHandle::copy() {
	PythonHandle new_handle = init(players.size(), {FIELD_HEIGHT, FIELD_WIDTH}, 0);
	new_handle = *this;
	return new_handle;
}

void PythonHandle::set(const PythonHandle& other) {
	*this = other;
}

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
	if (!player_count)
		last_winner = 0;
	if (alive_count > 1)
		last_winner = -1;
	seed();

	if (use_mask)
		for (unsigned i=0; i<players.size(); ++i)
			masks[i] = players[i].getMask(use_mask);
}

void PythonHandle::seed() {
	unsigned long long seed1 = time(NULL), seed2 = time(NULL);
	for (auto& player : players)
		player.seed(seed1, seed2);
}

bool PythonHandle::action(int player, int action) {
	GamePlay& game = players[player];
	switch (action) {
		case 1:
			game.field.mLeft();
			break;
		case 2:
			while (game.field.mLeft());
			break;
		case 3:
			game.field.mRight();
			break;
		case 4:
			while (game.field.mRight());
			break;
		case 5:
			game.mDown();
			break;
		case 6:
			while (game.mDown());
			break;
		case 7:
			{
				int sent = game.hd();
				if (sent == -1) // Player died
					return true;
				else if (!sent) // Nothing sent
					return false;

				distributeLines(player, sent);
			}
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
	return false;
}

void PythonHandle::distributeLines(int sender, int amount) {
	float number_of_players = players.size()-1;
	if (number_of_players < 1)
		return;
	float amount_per_player = amount / number_of_players;

	int count = -1;
	for (auto& player : players) {
		++count;
		if (count == sender)
			continue;

		player.incoming_lines += amount_per_player;
	}
}

bool PythonHandle::make_actions(std::vector<std::vector<int>> actions, int time_elapsed) {
	if (round_over)
		return true;
	for (unsigned i = 0; i < players.size(); ++i)
    if (!players[i].dead)
      for (auto p_action : actions[i])
        if (action(i, p_action)) {
          players[i].dead = true;
          break;
        }

	int player_count = -1;
	int alive_count = 0;
	for (auto& player : players) {
		++player_count;

		if (player.dead)
			continue;

		int sent = player.delayCheck(time_elapsed);
		if (sent == -1) {
			player.dead = true;
			masks[player_count].clear();
			continue;
		}
		else if (sent)
			distributeLines(player_count, sent);
		if (!player.dead)
			alive_count++;

		player.reward = player.data.linesCleared - player.linesCleared;
		player.linesCleared = player.data.linesCleared;

		player.incoming_lines_count = player.garbage.count();
		if (use_mask)
			masks[player_count] = player.getMask(use_mask);
	}

	if ((check_for_winner && alive_count < 2) || !alive_count) {
		round_over = true;
		return true;
	}

	return false;
}
