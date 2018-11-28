#include "gamePlay.h"
#include "gameField.h"
#include "pieces.h"
#include "randomizer.h"

#define PI 3.14159265

std::array<uint8_t, 7> GamePlay::randomizer_map {0,1,2,3,4,5,6};
bool GamePlay::only_zs=false;

GamePlay::GamePlay() :
field(),
garbage(data.linesBlocked),
combo(data.maxCombo)
{
	nextpiece=0;
	initBasePieces();
}

GamePlay& GamePlay::operator=(const GamePlay& other) {
	field = other.field;
	firstMove = other.firstMove;
	secondMove = other.secondMove;
	basepiece = other.basepiece;
	rander = other.rander;
	data = other.data;
	dropDelay = other.dropDelay;
	nextpiece = other.nextpiece;
	incoming_lines = other.incoming_lines;
	incoming_lines_count = other.incoming_lines_count;
	time_ms = other.time_ms;
	linesCleared = other.linesCleared;
	reward = other.reward;
	dead = other.dead;

	garbage = other.garbage;
	combo = other.combo;

	return *this;
}

int GamePlay::hd() {
	field.hd();
	field.addPiece();
	dropDelay.reset(time_ms);
	int sent = sendLines(field.clearlines());
	if (makeNewPiece())
		return -1;

	return sent;
}

bool GamePlay::mDown() {
	if (field.mDown()) {
		dropDelay.reset(time_ms);
		return true;
	}

	dropDelay.set(time_ms);
	return false;
}

bool GamePlay::makeNewPiece() {
	copyPiece(nextpiece);

	nextpiece = randomizer_map[rander.getPiece()];

	if (!field.possible()) {
		field.addPiece();
		return true;
	}

	return false;
}

void GamePlay::copyPiece(uint8_t np) {
	field.piece = basepiece[np];
	field.piece.posX = (FIELD_WIDTH - 4) / 2;
	field.piece.posY = 0;
}

int GamePlay::delayCheck(int time_elapsed) {
	time_ms += time_elapsed;

	if (dropDelay.check(time_ms))
		mDown();

	if (dropDelay.checkLockdown(time_ms) && !mDown())
		return hd();

	int add_garbage = 0;
	while (incoming_lines >= 1) {
		add_garbage++;
		incoming_lines -= 1.f;
	}
	if (add_garbage)
		garbage.add(add_garbage, time_ms);

	uint16_t comboLinesSent = combo.check(time_ms);
	if (comboLinesSent) {
		comboLinesSent = garbage.block(comboLinesSent, time_ms, false);
		data.linesSent += comboLinesSent;
	}

	if (garbage.check(time_ms))
		if (pushGarbage())
			return -1;

	return comboLinesSent;
}

void GamePlay::setPieceOrientation() {
	std::array<int, 7> piecerotation = {3,1,3,1,1,2,0};
	for (int x=0; x<7; x++) {
		basepiece[x].rotation = piecerotation[x];
		while (basepiece[x].rotation != basepiece[x].current_rotation)
			basepiece[x].rcw();
	}
}

void GamePlay::initBasePieces() {
	std::vector<short> value = { 0, 4, 0, 0,
								 0, 4, 0, 0,
								 0, 4, 4, 0,
								 0, 0, 0, 0,

								 0, 3, 0, 0,
								 0, 3, 0, 0,
								 3, 3, 0, 0,
								 0, 0, 0, 0,

								 0, 5, 0, 0,
								 0, 5, 5, 0,
								 0, 0, 5, 0,
								 0, 0, 0, 0,

								 0, 7, 0, 0,
								 7, 7, 0, 0,
								 7, 0, 0, 0,
								 0, 0, 0, 0,

								 0, 2, 0, 0,
								 0, 2, 0, 0,
								 0, 2, 0, 0,
								 0, 2, 0, 0,

								 0, 0, 0, 0,
								 1, 1, 1, 0,
								 0, 1, 0, 0,
								 0, 0, 0, 0,

								 0, 0, 0, 0,
								 0, 6, 6, 0,
								 0, 6, 6, 0,
								 0, 0, 0, 0 };

	short vc=0;

	for (int p=0; p<7; p++) {
		basepiece[p].posX=0;
		basepiece[p].posY=0;
		basepiece[p].lpiece=false;
		basepiece[p].current_rotation=0;
		basepiece[p].tile=p+1;
		basepiece[p].piece=p;
		for (int y=0; y<4; y++)
			for (int x=0; x<4; x++) {
                basepiece[p].grid[y][x] = value[vc];
				vc++;
			}
	}
	basepiece[4].lpiece=true;
	basepiece[6].lpiece=true;

	setPieceOrientation();
}

int GamePlay::sendLines(Vector2i lines) {
	data.garbageCleared+=lines.y;
	data.linesCleared+=lines.x;
	if (lines.x==0) {
		combo.noClear();
		return 0;
	}
	int amount = garbage.block(lines.x-1, time_ms);
	data.linesSent += amount;
	combo.increase(time_ms, lines.x);
	return amount;
}

void GamePlay::addGarbage(int amount) {
	garbage.add(amount, time_ms);

	data.linesRecieved+=amount;
}

bool GamePlay::pushGarbage() {
	addGarbageLine(rander.getHole());

	if (field.piece.posY > 0)
		field.piece.mup();

	if (!field.possible()) {
		if (field.piece.posY > 0)
			field.piece.mup();
		else
			return true;
	}

	return false;
}

void GamePlay::addGarbageLine() {
	uint8_t hole = rander.getHole();
	addGarbageLine(hole);
}

void GamePlay::addGarbageLine(uint8_t hole) {
	for (int y=0; y<FIELD_HEIGHT-1; y++)
		for (int x=0; x<FIELD_WIDTH; x++)
			field.getSquare(y,x)=field.getSquare(y+1, x);
	for (int x=0; x<10; x++)
		field.getSquare(FIELD_HEIGHT-1, x)=8;
	field.getSquare(FIELD_HEIGHT-1, hole)=0;
}

void GamePlay::restartRound() {
	field.clear();
	garbage.clear();
	combo.clear();
	data.clear();
	dropDelay.clear();
	time_ms = 0;
	incoming_lines = 0;
	linesCleared = 0;
	dead = 0;
}

void GamePlay::seed(uint64_t seed1, uint64_t seed2) {
	rander.seedHole(seed1);
	rander.seedPiece(seed2);
	rander.reset();
	makeNewPiece();
	if (!only_zs) {
		while (nextpiece == 2 || nextpiece == 3) {
			rander.reset();
			makeNewPiece();
		}
	}
	makeNewPiece();
}
