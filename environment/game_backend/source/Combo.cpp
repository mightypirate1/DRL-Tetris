#include "Combo.h"
#include <cmath>

const int32_t lineMultiplyer = 1000;
const int32_t staticMultiplyer = 800;

void ComboCounter::clear() {
	comboStart=0;
	comboTime=0;
	maxCombo=0;
	comboCount=0;
	lineCount=0;
}

void ComboCounter::increase(int32_t t, uint8_t amount) {
	if (comboCount==0) {
		comboStart=t;
		comboTime=0;
	}
	comboCount++;
	float lineTime = 0;
	for (int i=0; i<amount; i++) {
		lineCount++;
		lineTime+=lineMultiplyer/lineCount;
	}
	comboTime+= staticMultiplyer/comboCount + lineTime;

	if (comboCount>maxCombo)
		maxCombo=comboCount;
}

uint16_t ComboCounter::check(int32_t t) {
	int32_t remaining_time = comboStart + comboTime - t;
	if (remaining_time < 0)
		remaining = 0;
	else
		remaining = remaining_time;

	if (t > comboStart+comboTime && comboCount!=0) {
		float durationMultiplyer = 1.f + t / 60000.f * 0.1f;
		uint16_t comboLinesSent = pow(comboCount, 1.4+comboCount*0.01) * durationMultiplyer;
		comboCount = 0;
		lineCount = 0;

		return comboLinesSent;
	}
	return 0;
}

void ComboCounter::noClear() {
	comboTime-=200;
}