#include "../../include/Math.cuh"





//// return magniture of a vector
//__inline__ __storage__
//real NW::Math::Mag(const float v[_D_])
//{
//	return sqrt(NW::Math::DotProd(v, v));
//}
//
//// returns dot product of two vectors
//__inline__ __storage__
//real NW::Math::DotProd(const real v1[_D_], const real v2[_D_])
//{
//	real result = 0.0f;
//	for_D_ result += (v1[d] * v2[d]);
//	return result;
//}
//
//// rotate a vector by angle
//__inline__ __storage__
//void NW::Math::Rotate2D(real v[2], const float theta)
//{
//	const real costh = cosf(theta), sinth = sinf(theta);
//	const real R[2][2] = {
//		{ costh, -sinth },
//		{ sinth, costh }
//	};
//	real _nv[2] = { 0, 0 };
//	for (int i = 0; i < 2; i++) // for x and y
//		for (int j = 0; j < 2; j++) // sum over j
//			_nv[i] += R[i][j] * v[j];
//
//	v[0] = _nv[0]; v[1] = _nv[1]; // assign
//}