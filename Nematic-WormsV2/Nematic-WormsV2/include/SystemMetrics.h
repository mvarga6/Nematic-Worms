#pragma once

#include "../defs.h"
#include "typedefs.h"

namespace NW
{
	class SystemMetrics
	{
	public:
		virtual DisplacementMetric GetDisplacementMetric() = 0;
		virtual DistanceSquaredMetric GetDistanceMetric() = 0;
	};

	class SquareBoxMetrics : public SystemMetrics
	{
	public:
		virtual DisplacementMetric GetDisplacementMetric();
		virtual DistanceSquaredMetric GetDistanceMetric();
	};
}

