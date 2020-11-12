#pragma once

#include "dtypes.h"


class Particles {
    public:
        uint count;
        ptype *r;
        ptype *v;
        ptype *f;
        ptype *dev_v;
        ptype *dev_r;
        ptype *dev_f;

        Particles(uint n);
	    ~Particles();

        void Zero();
        void ZeroHost();
        void ZeroDevice();

        void ToDevice(bool onlyParticles = true);
        void ToHost(bool onlyParticles = true);

    private:
        size_t alloc_size;
};