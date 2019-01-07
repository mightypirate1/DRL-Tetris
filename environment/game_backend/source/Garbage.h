#ifndef GARBAGE_H
#define GARBAGE_H

#include <vector>
#include <cstdint>

struct Garbage {
	Garbage(short c, int32_t t) : count(c), delay(t) {}
	short count;
	int32_t delay;
};

inline uint16_t foo2;

class GarbageHandler {
public:
	GarbageHandler() : linesBlocked(foo2) {}
	GarbageHandler(uint16_t& _linesBlocked);
	GarbageHandler& operator=(const GarbageHandler& other) {
		garbage = other.garbage;
		minRemaining = other.minRemaining;
		addDelay = other.addDelay;

		return *this;
	}

	std::vector<Garbage> garbage;
	uint16_t & linesBlocked;
	int32_t minRemaining, addDelay;

	uint16_t count();
	void clear();
	void add(uint16_t amount, int32_t _time);
	uint16_t block(uint16_t amount, int32_t _time, bool freeze_incoming=true);
	bool check(int32_t _time);
	void setAddDelay(int32_t delay);
};

#endif
