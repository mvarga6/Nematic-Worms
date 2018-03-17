#pragma once


namespace NW
{
	class NeighborsGraph
	{
	public:
		NeighborsGraph();
		~NeighborsGraph();

		// the neighbors list
		int *List;
		int Max;
	};
}

