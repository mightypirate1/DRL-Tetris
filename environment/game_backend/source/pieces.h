#ifndef PIECES_H
#define PIECES_H

#include <cstdint>
#include <array>

class basePieces {
public:
	basePieces();
	std::array<std::array<uint8_t, 4>, 4> grid;

	uint8_t rotation;
	uint8_t current_rotation;

	int8_t posX, posY;

	short lpiece;

	uint8_t tile, piece;

	void rcw();
	void rccw();

	void mleft() { posX--; }
	void mright() { posX++; }
	void mup() { posY--; }
	void mdown() { posY++; }
};

#endif
