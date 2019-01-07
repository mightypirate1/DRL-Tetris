#ifndef DROPDELAY_H
#define DROPDELAY_H

#include <cstdint>

class DropDelay {
public:
	int32_t dropDelay, dropDelayTime;
	int32_t increaseDropDelay, increaseDropDelayTime;
	int32_t lockdownTime;
	bool lockdown;
	bool check(int32_t t);
	void reset(int32_t t);
	void clear();

	void set(int32_t t);
	bool checkLockdown(int32_t t);
};

#endif
