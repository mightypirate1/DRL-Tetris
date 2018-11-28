#include "Garbage.h"

const int32_t initialDelay = 1000;
const int32_t freezeDelay = 450;

GarbageHandler::GarbageHandler(uint16_t& _linesBlocked) : linesBlocked(_linesBlocked), addDelay(450) {}

uint16_t GarbageHandler::count() {
	uint16_t total=0;
	for (auto& garb : garbage)
		total+=garb.count;
	return total;
}

void GarbageHandler::clear() {
	garbage.clear();
	linesBlocked=0;
	minRemaining=initialDelay;
}

void GarbageHandler::add(uint16_t amount, int32_t _time) {
	garbage.push_back(Garbage(amount, _time + initialDelay));
}

uint16_t GarbageHandler::block(uint16_t amount, int32_t _time, bool freeze_incoming) {
	if (garbage.empty())
		return amount;
	int32_t delay = garbage.front().delay;
	
	int blocked=0;
	while (amount && !garbage.empty()) {
		garbage.front().count--;
		amount--;
		blocked++;
		if (garbage.front().count == 0)
			garbage.pop_front();
	}

	linesBlocked+=blocked;

	if (!garbage.empty()) {
		garbage.front().delay = std::max(delay, garbage.front().delay);
		if (freeze_incoming)
			garbage.front().delay = std::min(garbage.front().delay+freezeDelay, _time+minRemaining+freezeDelay);
	}
	else minRemaining = initialDelay;

	return amount;
}

bool GarbageHandler::check(int32_t _time) {
	if (garbage.empty())
		return false;
	if (_time > garbage.front().delay) {
		int32_t delay = garbage.front().delay + addDelay;
		if (--garbage.front().count == 0)
			garbage.pop_front();
		if (!garbage.empty()) {
			garbage.front().delay = std::max(delay, garbage.front().delay);
			minRemaining = garbage.front().delay - _time;
		}
		else minRemaining = initialDelay;
		return true;
	}
	minRemaining = std::min(minRemaining, garbage.front().delay - _time);
	return false;
}

void GarbageHandler::setAddDelay(int32_t delay) {
	addDelay = delay;
}