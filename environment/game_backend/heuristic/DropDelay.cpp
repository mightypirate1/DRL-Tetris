#include "DropDelay.h"

bool DropDelay::check(int32_t t) {
	if (t - increaseDropDelayTime > increaseDropDelay) {
		if (dropDelay > 200)
			dropDelay-=10;
		else if (dropDelay > 100)
			dropDelay-=5;
		else if (dropDelay > 50)
			dropDelay-=2;
		else if (dropDelay > 10)
			dropDelay-=1;
		increaseDropDelayTime = t;
	}

	if (t - dropDelayTime > dropDelay) {
		dropDelayTime = t;
		return true;
	}
	return false;
}

void DropDelay::reset(int32_t t) {
	dropDelayTime = t;
	lockdown = false;
}

void DropDelay::clear() {
	increaseDropDelay=3000;
	increaseDropDelayTime=0;
	dropDelay=1000;
	dropDelayTime=0;
	lockdownTime=0;
	lockdown=false;
}

void DropDelay::set(int32_t t) {
	if (!lockdown)
		lockdownTime = t + 400;
	lockdown = true;
}

bool DropDelay::checkLockdown(int32_t t) {
	if (lockdown && t > lockdownTime)
		return true;

	return false;
}