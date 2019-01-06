// 			Usage:
//
// Build the c++ module by running:
// 		cmake .
//		make
//
// In python, use like this:
//		import tetris_env as t
//		env = t.PythonHandle(2, [22,10])  // 2 is number of players, 22, 10 is size of field
//
//		env.states	- An array of players state
//		env.masks		- An array of possible special moves for each player
//
//		env.action( {p1_action, p2_action }, time_elapsed)
//
//		p1_action is an array of actions for player1, p2_action same for player2
//		time elapsed (in milliseconds) since last action
//		return true if the round ended, and all player states will have been reset to starting state
//
//		List of actions:
//			0 - Do nothing
//			1 - Move 1 step to the left
//			2 - Move as far as possible to the left
//			3 - Move 1 step to the right
//			4 - Move as far as possible to the right
//			5 - Move 1 step down
//			6 - Move as far as possible down
//			7 - Hard Drop, like 6 but the piece is also locked down
//			8 - Rotate clock-wise
//			9 - Rotate counter clock-wise
//			10 - Flip piece, same as 2 rotations in the same direction
//
//		State struct has the following members
//			field - 2d array containing the playing field
//			piece - 2d array containing the piece
//			x & y - position of the piece
//			inc_lines - Lines that are waiting to be added to the playfield
//			combo_count - The current combo count
//			combo_time - How much time combo time that remains
//			nextpiece - The id-number of the next piece, 0-6
//			reward - reward value for latest action
//			dead - true if player is dead

#ifndef PYTHONHANDLE_H
#define PYTHONHANDLE_H

#include "gamePlay.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include <memory>

struct State {
	State() = default;
	State(GamePlay& gp) :
		field({FIELD_HEIGHT, FIELD_WIDTH}, {FIELD_WIDTH, 1}, gp.field.square.data(), pybind11::capsule(this, [](void*){})),
		piece({4,4}, {4,1}, gp.field.piece.grid[0].data(), pybind11::capsule(this, [](void*){})),
		x({1}, {1}, &gp.field.piece.posX, pybind11::capsule(this, [](void*){})),
		y({1}, {1}, &gp.field.piece.posY, pybind11::capsule(this, [](void*){})),
		incoming_lines({1}, {1}, &gp.incoming_lines_count, pybind11::capsule(this, [](void*){})),
		combo_time({1}, {1}, &gp.combo.remaining, pybind11::capsule(this, [](void*){})),
		combo_count({1}, {1}, &gp.combo.comboCount, pybind11::capsule(this, [](void*){})),
		nextpiece({1}, {1}, &gp.nextpiece, pybind11::capsule(this, [](void*){})),
		reward({1}, {1}, &gp.reward, pybind11::capsule(this, [](void*){})),
		dead({1}, {1}, &gp.dead, pybind11::capsule(this, [](void*){})) {}

	State(const State&) {} // Added no-op copy so env can be copied without ruining state
	State& operator=(const State&) { return *this; }

	pybind11::array_t<uint8_t> field;
	pybind11::array_t<uint8_t> piece;
	pybind11::array_t<int8_t> x; // position for current piece
	pybind11::array_t<int8_t> y;

	pybind11::array_t<uint8_t> incoming_lines;
	pybind11::array_t<uint16_t> combo_time;
	pybind11::array_t<uint8_t> combo_count;
	pybind11::array_t<uint8_t> nextpiece;
	pybind11::array_t<uint8_t> reward;
	pybind11::array_t<uint8_t> dead;
};

struct PythonHandle {
	static PythonHandle init(int number_of_players, std::array<int, 2> field_size);
	std::vector<GamePlay> players;
	int use_mask=0;
	bool check_for_winner=false, round_over=false;
	std::vector<State> states;
	std::vector<Mask> masks;
	int last_winner = -1;

	bool action(int player, int action);

	bool make_actions(std::vector<std::vector<int>> actions, int time_elapsed);

	void distributeLines(int sender, int amount);

	void reset();
	void seed();

	void get_actions(int player);

	PythonHandle copy();
	void set(const PythonHandle&);
};

PYBIND11_MODULE(TETRIS_MODULE_NAME, m) {
  m.doc() = "tetris environment for DQN learning";

	m.def("set_pieces", [&](std::array<uint8_t, 7> a){
		GamePlay::randomizer_map = a;
		GamePlay::only_zs = true;
		for (auto v : a)
			if (v != 2 && v != 3)
				GamePlay::only_zs = false;
	});

	pybind11::class_<PythonHandle>(m, "PythonHandle")
	.def(pybind11::init(&PythonHandle::init))
	.def("action", &PythonHandle::make_actions)
	.def("copy", &PythonHandle::copy)
	.def("set", &PythonHandle::set)
	.def("reset", &PythonHandle::reset)
	.def("get_actions", &PythonHandle::get_actions)
	.def_readwrite("use_mask", &PythonHandle::use_mask)
	.def_readonly("states", &PythonHandle::states)
	.def_readonly("masks", &PythonHandle::masks)
	.def_readonly("last_winner", &PythonHandle::last_winner);

	pybind11::class_<Mask>(m, "Mask")
	.def(pybind11::init<>())
	.def_readonly("mask", &Mask::mask)
	.def_readonly("action", &Mask::action);

  pybind11::class_<State>(m, "State")
	.def(pybind11::init<>())
	.def_readwrite("field", &State::field)
	.def_readwrite("piece", &State::piece)
	.def_readonly("inc_lines", &State::incoming_lines)
	.def_readonly("combo_time", &State::combo_time)
	.def_readonly("combo_count", &State::combo_count)
	.def_readwrite("nextpiece", &State::nextpiece)
	.def_readonly("reward", &State::reward)
	.def_readonly("dead", &State::dead)
	.def_readonly("x", &State::x)
	.def_readonly("y", &State::y);
}

#endif
