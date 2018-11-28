#ifndef COMBO_H
#define COMBO_H

#include <cstdint>

class ComboCounter {
private:
	int32_t comboStart=0, comboTime=0;
	uint8_t lineCount=0;
public:
	ComboCounter(uint16_t& _maxCombo) : maxCombo(_maxCombo) {}
	ComboCounter& operator=(const ComboCounter& other) {
		comboStart = other.comboStart;
		comboTime = other.comboTime;
		lineCount = other.lineCount;
		comboCount = other.comboCount;
		remaining = other.remaining;

		return *this;
	}

	uint16_t & maxCombo;
	uint8_t comboCount=0;
	uint16_t remaining;

	void clear();
	void increase(int32_t t, uint8_t amount);
	uint16_t check(int32_t t);
	void noClear();
};

#endif
