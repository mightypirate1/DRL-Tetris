#ifndef GAMEPLAY_H
#define GAMEPLAY_H

#include <deque>
#include <array>
#include "pieces.h"
#include "randomizer.h"
#include "gameField.h"
#include "Garbage.h"
#include "Combo.h"
#include "DropDelay.h"
#include "TestField.h"

struct GameplayData {
	uint16_t linesSent=0;
	uint16_t linesRecieved=0;
	uint16_t bpm=0;
	uint16_t garbageCleared=0;
	uint16_t linesCleared=0;
	uint16_t pieceCount=0;
	uint16_t linesBlocked=0;
	uint16_t maxCombo=0;
	uint16_t roundDuration=0;
	void clear() {
		linesSent=0;
		linesRecieved=0;
		bpm=0;
		garbageCleared=0;
		linesCleared=0;
		pieceCount=0;
		linesBlocked=0;
		maxCombo=0;
		roundDuration=0;
	}
};

class GamePlay {
public:
	GamePlay();
	GamePlay& operator=(const GamePlay&);

	BasicField field;
	TestField firstMove, secondMove;
	std::array<basePieces, 7> basepiece;

	randomizer rander;

	GameplayData data;

	GarbageHandler garbage;
	ComboCounter combo;
	DropDelay dropDelay;

	static std::array<uint8_t, 7> randomizer_map;
	static bool only_zs;

	uint8_t nextpiece = 7;

	float incoming_lines = 0;

	uint8_t incoming_lines_count = 0;

	int32_t time_ms = 0;

	uint16_t linesCleared = 0;
	uint8_t reward = 0;

	uint8_t dead = 0;

	// Heuristic

	std::array<int16_t, 7> state_representation;

	int hd();

	bool mDown();

	void addPiece(int32_t);
	bool makeNewPiece();
	void copyPiece(uint8_t np);

	int delayCheck(int time_elapsed);

	void setPieceOrientation();
	void initBasePieces();

	int sendLines(Vector2i lines);

	void addGarbage(int amount);
	bool pushGarbage();
	void addGarbageLine();
	void addGarbageLine(uint8_t hole);

	void restartRound();
	void seed(uint64_t seed1, uint64_t seed2);
};

#endif
