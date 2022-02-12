#ifndef GAMEFIELD_H
#define GAMEFIELD_H

#include <cstdint>
#include "pieces.h"
#include <vector>

extern int FIELD_WIDTH, FIELD_HEIGHT;

class Resources;

struct Vector2i{
  int x,y;
};

class BasicField {
public:
  BasicField() : square(FIELD_WIDTH*FIELD_HEIGHT), width(FIELD_WIDTH), height(FIELD_HEIGHT) {}

  std::vector<uint8_t> square;
  basePieces piece;
  uint8_t width, height;

  uint8_t& getSquare(uint8_t y, uint8_t x) {
    return square[y*width + x];
  }

  bool possible();

  bool mRight();
  bool mLeft();
  bool mDown();
  void hd();
  bool rcw();
  bool rccw();
  bool r180();
  bool kickTest();

  void addPiece();

  void removeline(short y);
  Vector2i clearlines();

  void clear();
};

#endif
