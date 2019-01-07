#ifndef TESTFIELD_H
#define TESTFIELD_H

#include "gameField.h"
#include <array>
#include <vector>

struct Mask {
	std::vector<int> mask;
	std::vector<std::vector<int>> action;

	void clear_to_size() {
		mask = std::vector<int>(FIELD_WIDTH*4);
		action = std::vector<std::vector<int>>(FIELD_WIDTH*4);
	}

	void clear() {
		mask.clear();
		action.clear();
	}
};

struct MoveInfo {
	int8_t posX;
	uint8_t rot;
	std::vector<uint8_t> path;
	bool use_path;
	double score;

	void clear();
};

class TestField : public BasicField {
public:
	std::vector<uint8_t> test_path;
	Mask mask;
	int current_column;
	// int use_mask;

	MoveInfo finesseMove;

	void setPiece(basePieces& newpiece);

	Mask& getMask(int use_mask);

	void findNextMove();
	void useFinesseMove();
	bool setFinesseMove();
	void tryAllFinesseMoves();
	bool finesseIsPossible();
	bool tryLeft(bool clearTestPath=false);
	bool tryRight(bool clearTestPath=false);
	bool tryUp(int turnVal);
	bool reverseWallkick();
	bool doWallKick();
	void r180KeepPos();
	uint8_t moveUp();
	bool restorePiece(basePieces p, bool returnValue);
};

#endif
