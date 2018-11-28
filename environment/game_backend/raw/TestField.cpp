#include "TestField.h"

void convert(std::vector<int>& vec) {
	for (auto& m : vec) {
		switch (m) {
			case 255:
				m=3;
				break;

			case 254:
				m=1;
				break;

			case 253:
				m=6;
				break;

			case 252:
				m=5;
				break;

			case 241:
				m=8;
				break;

			case 242:
				m=10;
				break;

			case 243:
				m=9;
				break;
		}
	}
}

void addRotationValue(int rotationValue, std::vector<int>& vec) {
	if (rotationValue < 0)
		rotationValue += 4;
	if (rotationValue)
		vec.push_back(240+rotationValue);
}

void makeStartSequence(int posX, std::vector<int>& vec) {
	if (posX > (FIELD_WIDTH - 4) / 2) for (int i=0; i < posX - (FIELD_WIDTH - 4) / 2; i++)
		vec.push_back(255);
	else for (int i=0; i < (FIELD_WIDTH - 4) / 2 - posX; i++)
		vec.push_back(254);
}

void MoveInfo::clear() {
	posX=(FIELD_WIDTH - 4) / 2;
	rot=0;
	score=-10000;
	use_path=false;
}

void TestField::setPiece(basePieces& newpiece) {
	piece = newpiece;
	piece.posX = (FIELD_WIDTH - 4) / 2;;
	piece.posY = 0;
}

Mask& TestField::getMask(int use_mask_) {
	use_mask = use_mask_;
	if (use_mask == 2)
		mask.clear();
	else
		mask.clear_to_size();

	if (piece.piece == 6) {
		for (int x=-1; x<FIELD_WIDTH-1; x++) {
			piece.posX=x;
			piece.posY=0;
			if (!possible())
				continue;

			findNextMove();
		}
	}
	else if (piece.piece == 4 || piece.piece == 2 || piece.piece == 3) {
		for (int x=-1; x<FIELD_WIDTH-1; x++) {
			for (int r=0; r<2; r++) {
				piece.posX=x;
				piece.posY=0;
				while (piece.current_rotation != r)
					piece.rcw();
				if (!possible())
					continue;

				findNextMove();
			}
		}
	}
	else {
		for (int x=-1; x<FIELD_WIDTH-1; x++) {
			for (int r=0; r<4; r++) {
				piece.posX=x;
				piece.posY=0;
				while (piece.current_rotation != r)
					piece.rcw();
				if (!possible())
					continue;

				findNextMove();
			}
		}
	}

	return mask;
}

void TestField::findNextMove() {
	if (use_mask == 2) {
		mask.action.push_back({});
		mask.mask.push_back(1);
		addRotationValue(piece.current_rotation - piece.rotation, mask.action.back());
		makeStartSequence(piece.posX, mask.action.back());
		convert(mask.action.back());
		mask.action.back().push_back(7);
	}
	current_column = piece.posX+1;
	hd();
	tryAllFinesseMoves();
}

void TestField::useFinesseMove() {
	std::vector<int>* currentMove;
	if (use_mask == 2) {
		mask.action.push_back({});
		mask.mask.push_back(1);
		currentMove = &mask.action.back();
	}
	else {
		int cc = current_column*4;
		for (int i=0; i<3; i++)
			if (mask.mask[cc])
				++cc;

		if (mask.mask[cc])
			return;

		mask.mask[cc] = 1;

		currentMove = &mask.action[cc];
	}

	addRotationValue(finesseMove.rot - piece.rotation, *currentMove);

	if (finesseMove.use_path) {
		makeStartSequence(finesseMove.posX, *currentMove);
		for (auto it = finesseMove.path.rbegin(); it != finesseMove.path.rend(); it++) {
			if (*it < 240) {
				for (int i = 0; i < *it; i++)
					currentMove->push_back(252);
			}
			else currentMove->push_back(*it);
		}
	}

	convert(*currentMove);

	currentMove->push_back(7);
}

bool TestField::setFinesseMove() {
	finesseMove.path = test_path;
	finesseMove.posX = piece.posX;
	finesseMove.rot = piece.current_rotation;
	finesseMove.use_path=true;
	return true;
}

void TestField::tryAllFinesseMoves() {
	auto pieceBackup = piece;
	for (int y=piece.posY+2; y < FIELD_HEIGHT-1; y++) {
		piece = pieceBackup;
		piece.posY = y;
		if (possible()) {
			hd();
			y=piece.posY;
			if (finesseIsPossible())
					useFinesseMove();
		}
	}
	piece = pieceBackup;
}

bool TestField::finesseIsPossible() {
	auto pieceBackup = piece;

	if (reverseWallkick())
		return restorePiece(pieceBackup, true);

	piece = pieceBackup;
	if (tryLeft(true))
		return restorePiece(pieceBackup, true);

	return restorePiece(pieceBackup, tryRight(true));
}

bool TestField::tryLeft(bool clearTestPath) {
	if (!mLeft())
		return false;

	if (clearTestPath)
		test_path.clear();
	test_path.push_back(255);

	auto up = moveUp();
	if (up)
		test_path.push_back(up);

	if (piece.posY == 0) {
		setFinesseMove();
		return true;
	}

	return tryLeft();
}

bool TestField::tryRight(bool clearTestPath) {
	if (!mRight())
		return false;

	if (clearTestPath)
		test_path.clear();
	test_path.push_back(254);

	auto up = moveUp();
	if (up)
		test_path.push_back(up);

	if (piece.posY == 0)
		return setFinesseMove();

	return tryRight();
}

bool TestField::tryUp(int turnVal) {
	test_path.clear();
	auto up = moveUp();

	test_path.push_back(turnVal+240);

	if (up)
		test_path.push_back(up);

	if (piece.posY == 0)
		 return setFinesseMove();

	if (tryLeft())
		return true;
	test_path.clear();
	test_path.push_back(turnVal+240);
	if (up)
		test_path.push_back(up);
	return tryRight();
}

bool TestField::reverseWallkick() {
	if (piece.piece == 6)
		return false;

	if (mRight()) { mLeft(); return false; }
	if (mLeft()) { mRight(); return false; }

	auto pieceBackup = piece;
	if (doWallKick())
		return restorePiece(pieceBackup, true);

	if (piece.piece == 2 || piece.piece == 3 || piece.piece == 4) {
		r180KeepPos();
		return restorePiece(pieceBackup, doWallKick());
	}

	return restorePiece(pieceBackup, false);
}

bool TestField::doWallKick() {
	auto pieceBackup = piece;
	int r;
	bool working;
	for (r=0; r<4; r++) {
		working=true;
		if (r == pieceBackup.current_rotation) {
			working=false;
			continue;
		}

		while (piece.current_rotation != r)
			piece.rcw();

		if (possible()) {
			int turnVal = pieceBackup.current_rotation - r;
			if (turnVal < 0)
				turnVal+=4;
			if (tryUp(turnVal))
				return restorePiece(pieceBackup, true);
			else
				return restorePiece(pieceBackup, false);
		}

		piece.posY--; if (possible()) break;
		piece.posY++; piece.posX--; if (possible()) break;
		piece.posX+=2; if (possible()) break;
		piece.posY--; piece.posX-=2; if (possible()) break;
		piece.posX+=2; if (possible()) break;
		piece.posY++; piece.posX-=3; if (possible()) break;
		piece.posX+=4; if (possible()) break;

		piece.posX-=2;

		working=false;
	}

	if (working) {
		int turnVal = pieceBackup.current_rotation - r;
		if (turnVal < 0)
			turnVal+=4;

		auto pieceBackup2 = piece;
		if (turnVal == 1)
			rcw();
		else if (turnVal == 2)
			r180();
		else if (turnVal == 3)
			rccw();
		else
			return restorePiece(pieceBackup, false);

		if (piece.posX != pieceBackup.posX || piece.posY != pieceBackup.posY)
			return restorePiece(pieceBackup, false);

		piece = pieceBackup2;

		if (tryUp(turnVal))
			return restorePiece(pieceBackup, true);

		piece = pieceBackup2;
		test_path.clear();
		test_path.push_back(240+turnVal);
		if (tryLeft())
			return restorePiece(pieceBackup, true);

		piece = pieceBackup2;
		if (test_path.size() > 1) {
			test_path.clear();
			test_path.push_back(240+turnVal);
		}

		return restorePiece(pieceBackup, tryRight());
	}

	return restorePiece(pieceBackup, false);
}

void TestField::r180KeepPos() {
	piece.rcw(); piece.rcw();
	if (piece.piece == 4) {
		if (piece.current_rotation == 0)
			piece.posX++;
		else if (piece.current_rotation == 1)
			piece.posY++;
		else if (piece.current_rotation == 2)
			piece.posX--;
		else
			piece.posY--;
	}
	if (piece.piece == 3) {
		if (piece.current_rotation == 0)
			piece.posX++;
		else if (piece.current_rotation == 1)
			piece.posY++;
		else if (piece.current_rotation == 2)
			piece.posX--;
		else
			piece.posY--;
	}
	if (piece.piece == 2) {
		if (piece.current_rotation == 0)
			piece.posX--;
		else if (piece.current_rotation == 1)
			piece.posY--;
		else if (piece.current_rotation == 2)
			piece.posX++;
		else
			piece.posY++;
	}
}

uint8_t TestField::moveUp() {
	bool doHD = false;
	piece.mdown();
	if (!possible())
		doHD=true;
	piece.mup();

	uint8_t count=0;
	do {
	    piece.mup();
	    count++;
	} while (possible());

	count--;
    piece.mdown();
    if (count && doHD)
    	return 253;
    return count;
}

bool TestField::restorePiece(basePieces p, bool returnValue) {
	piece = p;
	return returnValue;
}
