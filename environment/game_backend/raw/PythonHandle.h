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

	// pybind11::tuple seabass(int);

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

pybind11::class_<BasicField>(m, "BasicField")
.def(pybind11::pickle([](const BasicField& g){
	return pybind11::make_tuple(
															g.square,
															g.piece,
															g.width,
															g.height
															);
}, [](pybind11::tuple t){
	auto ret = BasicField();
	ret.square = t[0].cast<std::vector<uint8_t>>();
	ret.piece  = t[1].cast<basePieces>();
	ret.width  = t[2].cast<uint8_t>();
	ret.height = t[3].cast<uint8_t>();
	return ret;
}))
.def(pybind11::init<>());
// :)

pybind11::class_<TestField>(m, "TestField")
.def(pybind11::pickle([](const TestField& g){
	return pybind11::make_tuple(
															g.test_path,
															g.mask,
															g.current_column,
															g.finesseMove
															);
}, [](pybind11::tuple t){
	auto ret = TestField();
	ret.test_path      = t[0].cast<std::vector<uint8_t>>();
	ret.mask           = t[1].cast<Mask>();
	ret.current_column = t[2].cast<int>();
	ret.finesseMove    = t[3].cast<MoveInfo>();
	return ret;
}))
.def(pybind11::init<>());
// :)

pybind11::class_<MoveInfo>(m, "MoveInfo")
.def(pybind11::pickle([](const MoveInfo& g){
	return pybind11::make_tuple(
															g.posX,
															g.rot,
															g.path,
															g.use_path,
															g.score
															);
}, [](pybind11::tuple t){
	auto ret = MoveInfo();
	ret.posX      = t[0].cast<int8_t>();
	ret.rot       = t[1].cast<uint8_t>();
	ret.path      = t[2].cast<std::vector<uint8_t>>();
	ret.use_path  = t[3].cast<bool>();
	ret.score     = t[4].cast<double>();
	return ret;
}))
.def(pybind11::init<>());
// :)


pybind11::class_<basePieces>(m, "basePieces")
.def(pybind11::pickle([](const basePieces& g){
	return pybind11::make_tuple(
															g.grid,
															g.rotation,
															g.current_rotation,
															g.posX,
															g.posY,
															g.lpiece,
															g.tile,
															g.piece
															);
}, [](pybind11::tuple t){
	auto ret = basePieces();
	ret.grid             = t[0].cast<std::array<std::array<uint8_t, 4>, 4>>();
	ret.rotation         = t[1].cast<uint8_t>();
	ret.current_rotation = t[2].cast<uint8_t>();
	ret.posX             = t[3].cast<int8_t>();
	ret.posY             = t[4].cast<int8_t>();
	ret.lpiece           = t[5].cast<short>();
	ret.tile             = t[6].cast<uint8_t>();
	ret.piece            = t[7].cast<uint8_t>();
	return ret;
}))
.def(pybind11::init<>());
// :)

pybind11::class_<randomizer>(m, "randomizer")
.def(pybind11::pickle([](const randomizer&){
	return pybind11::make_tuple(0);
}, [](pybind11::tuple){
	return randomizer();
}))
.def(pybind11::init<>());
// :)

pybind11::class_<GameplayData>(m, "GameplayData")
.def(pybind11::pickle([](const GameplayData& g){
	return pybind11::make_tuple(
															g.linesSent,
															g.linesRecieved,
															g.bpm,
															g.garbageCleared,
															g.linesCleared,
															g.pieceCount,
															g.linesBlocked,
															g.maxCombo,
															g.roundDuration
															);
}, [](pybind11::tuple t){
	auto ret = GameplayData();
	ret.linesSent	      = t[0].cast<uint16_t>();
	ret.linesRecieved	  = t[1].cast<uint16_t>();
	ret.bpm	            = t[2].cast<uint16_t>();
	ret.garbageCleared	= t[3].cast<uint16_t>();
	ret.linesCleared	  = t[4].cast<uint16_t>();
	ret.pieceCount	    = t[5].cast<uint16_t>();
	ret.linesBlocked	  = t[6].cast<uint16_t>();
	ret.maxCombo	      = t[7].cast<uint16_t>();
	ret.roundDuration	  = t[8].cast<uint16_t>();
	return ret;
}))
.def(pybind11::init<>());
// :)

pybind11::class_<GarbageHandler>(m, "GarbageHandler")
.def(pybind11::pickle([](const GarbageHandler& g){
	return pybind11::make_tuple(
															g.garbage,
															g.minRemaining,
															g.addDelay
															);
}, [](pybind11::tuple t){
	auto ret = GarbageHandler();
	ret.garbage      = t[0].cast<std::vector<Garbage>>();
	ret.minRemaining = t[1].cast<int32_t>();
	ret.addDelay     = t[2].cast<int32_t>();
	return ret;
}))
.def(pybind11::init<>());
// :)

pybind11::class_<ComboCounter>(m, "ComboCounter")
.def(pybind11::pickle([](const ComboCounter& g){
	return pybind11::make_tuple(
															g.maxCombo,
															g.comboCount,
															g.remaining
															);
}, [](pybind11::tuple t){
	auto ret = ComboCounter();
	ret.maxCombo   = t[0].cast<uint16_t>();
	ret.comboCount = t[1].cast<uint8_t>();
	ret.remaining  = t[2].cast<uint16_t>();
	return ret;
}))
.def(pybind11::init<>());
// :)

	pybind11::class_<DropDelay>(m, "DropDelay")
	.def(pybind11::pickle([](const DropDelay& g){
		return pybind11::make_tuple(
																g.dropDelay,
																g.dropDelayTime,
																g.increaseDropDelay,
																g.increaseDropDelayTime,
																g.lockdownTime,
																g.lockdown
															  );
	}, [](pybind11::tuple t){
		auto ret = DropDelay();
		ret.dropDelay             = t[0].cast<int32_t>();
		ret.dropDelayTime         = t[1].cast<int32_t>();
		ret.increaseDropDelay     = t[2].cast<int32_t>();
		ret.increaseDropDelayTime = t[3].cast<int32_t>();
		ret.lockdownTime          = t[4].cast<int32_t>();
		ret.lockdown              = t[5].cast<bool>();
		return ret;
	}))
	.def(pybind11::init<>());
	// :)

	pybind11::class_<GamePlay>(m, "GamePlay")
	.def(pybind11::pickle([](const GamePlay& g){
		return pybind11::make_tuple(
																g.field,
																g.testField,
																g.basepiece,
																g.rander,
																g.data,
																g.garbage,
																g.combo,
																g.dropDelay,
																g.nextpiece,
																g.incoming_lines,
																g.incoming_lines_count,
																g.time_ms,
																g.linesCleared,
																g.reward,
																g.dead
															 );
	}, [](pybind11::tuple t){
		auto ret                 = GamePlay();
		ret.field                = t[0].cast<BasicField>();
		ret.testField            = t[1].cast<TestField>();
		ret.basepiece            = t[2].cast<std::array<basePieces, 7>>();
		ret.rander               = t[3].cast<randomizer>();
		ret.data                 = t[4].cast<GameplayData>();
		ret.garbage              = t[5].cast<GarbageHandler>();
		ret.combo                = t[6].cast<ComboCounter>();
		ret.dropDelay            = t[7].cast<DropDelay>();
		ret.nextpiece            = t[8].cast<uint8_t>();
		ret.incoming_lines       = t[9].cast<float>();
		ret.incoming_lines_count = t[10].cast<uint8_t>();
		ret.time_ms              = t[11].cast<int32_t>();
		ret.linesCleared         = t[12].cast<uint16_t>();
		ret.reward               = t[13].cast<uint8_t>();
		ret.dead                 = t[14].cast<uint8_t>();
		return ret;
	}))
	.def(pybind11::init<>());
	// :)

	pybind11::class_<PythonHandle>(m, "PythonHandle")
	.def(pybind11::pickle([](const PythonHandle& p){
		return pybind11::make_tuple(
																p.players,
																p.round_over,
																p.check_for_winner,
																p.players.size(),
																p.masks,
																p.last_winner,
																FIELD_WIDTH,
																FIELD_HEIGHT,
																GamePlay::randomizer_map,
																GamePlay::only_zs
															  );
	}, [](pybind11::tuple t){
		PythonHandle ret;
		ret.players              = t[0].cast<std::vector<GamePlay>>();
		ret.round_over           = t[1].cast<bool>();
		ret.check_for_winner     = t[2].cast<bool>();
		auto n_players           = t[3].cast<std::size_t>();
		ret.states.clear();
		ret.masks.clear();
		for(uint8_t i=0; i<n_players;++i){
			ret.states.emplace_back(ret.players[i]);
			ret.masks.emplace_back( );
		}
		ret.masks                = t[4].cast<std::vector<Mask>>();
		ret.last_winner          = t[5].cast<int>();
		FIELD_WIDTH              = t[6].cast<int>();
		FIELD_HEIGHT             = t[7].cast<int>();
		GamePlay::randomizer_map = t[8].cast<std::array<uint8_t, 7>>();
		GamePlay::only_zs        = t[9].cast<bool>();
		return ret;
	}))
	.def(pybind11::init(&PythonHandle::init))
	.def("action", &PythonHandle::make_actions)
	.def("copy", &PythonHandle::copy)
	.def("set", &PythonHandle::set)
	.def("reset", &PythonHandle::reset)
	// .def("seabass", &PythonHandle::seabass)
	.def("get_actions", &PythonHandle::get_actions)
	.def_readonly("states", &PythonHandle::states)
	.def_readonly("masks", &PythonHandle::masks)
	.def_readonly("last_winner", &PythonHandle::last_winner);

	pybind11::class_<Mask>(m, "Mask")
	.def(pybind11::init<>())
	.def_readonly("mask", &Mask::mask)
	.def_readonly("action", &Mask::action)
	.def(pybind11::pickle([](const Mask&){
		return pybind11::make_tuple(
																0
																);
	}, [](pybind11::tuple){
		return Mask();
	}));

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
	.def_readonly("y", &State::y)
	.def(pybind11::pickle([](const State&){
		return pybind11::make_tuple(
																0
																);
	}, [](pybind11::tuple){
		return State();
	}));
}

#endif
