#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <future>
#include <immintrin.h>
#include <wmmintrin.h>
#include <time.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//	将数据裁剪到unsigned char所能表示的有效范围内。
inline unsigned char IM_ClampToByte(int Value)
{
	if (Value > 255)
		return 255;
	else if (Value < 0)
		return 0;
	else
		return Value;
}

//	用于16位有符号和无符号整除的相关函数
inline unsigned int bit_scan_reverse(unsigned int  a)
{
	unsigned long r= 0;
	// _BitScanReverse(&r, a);            // defined in intrin.h for MS and Intel compilers
	return r;
}

inline void GetMSS_S(short d, short &Multiplier, int &Shift, int &Sign)
{
	const int d1 = abs(d);
	int sh, m;
	if (d1 > 1)
	{
		// sh = bit_scan_reverse(d1 - 1);                   // shift count = ceil(log2(d1))-1 = (bit_scan_reverse(d1-1)+1)-1
		// m = ((int(1) << (16 + sh)) / d1 - ((int(1) << 16) - 1)); // calculate multiplier
		sh = __builtin_ffs(d1 - 1);
		m = ((int(1) << (15 + sh)) / d1 - ((int(1) << 16) - 1)); // calculate multiplier
	}
	else
	{
		m = 1;                                        // for d1 = 1
		sh = 0;
		if (d == 0)
			m /= d;                            // provoke error here if d = 0
		if ((unsigned short)(d) == 0x8000u)
		{                  // fix overflow for this special case
			m = 0x8001;
			sh = 14;
		}
	}
	Multiplier = short(m);				// broadcast multiplier
	Shift = sh;						 // shift count
	Sign = d < 0 ? -1 : 0;					 // sign of divisor
}

//	浮点数据的绝对值计算
inline __m128 _mm_abs_ps(__m128 v)
{
	static const int i = 0x7fffffff;
	float mask = *(float*)&i;
	return _mm_and_ps(v, _mm_set_ps1(mask));
}

inline __m256 _mm256_abs_ps(__m256 v)
{
	static const int i = 0x7fffffff;
	float mask = *(float*)&i;
	return _mm256_and_ps(v, _mm256_set1_ps(mask));
}

//	http://www.agner.org/optimize/		vectorclass
//	The division of a vector of 16-bit integers is faster than division of a vector of other integer sizes.
//	当除数为正数,被除数为signed short时的16位除法, 注意这里除数必须时相同的
inline __m128i _mm_divp_epi16(__m128i Dividend, int Multiplier, int Shift)
{
	__m128i t1 = _mm_mulhi_epi16(Dividend, _mm_set1_epi16(Multiplier));				// multiply high signed words
	__m128i t2 = _mm_add_epi16(t1, Dividend);										// + a
	__m128i t3 = _mm_srai_epi16(t2, Shift);											// shift right arithmetic
	__m128i t4 = _mm_srai_epi16(Dividend, 15);										// sign of a
	return  _mm_sub_epi16(t3, t4);													// + 1 if a < 0, -1 if d < 0
}


inline __m256i _mm256_divp_epi16(__m256i Dividend, int Multiplier, int Shift)
{
	__m256i t1 = _mm256_mulhi_epi16(Dividend, _mm256_set1_epi16(Multiplier));				// multiply high signed words
	__m256i t2 = _mm256_add_epi16(t1, Dividend);										// + a
	__m256i t3 = _mm256_srai_epi16(t2, Shift);											// shift right arithmetic
	__m256i t4 = _mm256_srai_epi16(Dividend, 15);										// sign of a
	return  _mm256_sub_epi16(t3, t4);													// + 1 if a < 0, -1 if d < 0
}

#ifdef AVX512
inline __m512i _mm512_divp_epi16(__m512i Dividend, int Multiplier, int Shift)
{
	__m512i t1 = _mm512_mulhi_epi16(Dividend, _mm512_set1_epi16(Multiplier));				// multiply high signed words
	__m512i t2 = _mm512_add_epi16(t1, Dividend);										// + a
	__m512i t3 = _mm512_srai_epi16(t2, Shift);											// shift right arithmetic
	__m512i t4 = _mm512_srai_epi16(Dividend, 15);										// sign of a
	return  _mm512_sub_epi16(t3, t4);													// + 1 if a < 0, -1 if d < 0
}
#endif // AVX512

void IM_SplitRGB(unsigned char *Src, unsigned char *Blue, unsigned char *Green, unsigned char *Red, int Width, int Height, int Stride)
{
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePB = Blue + Y * Width;
		unsigned char *LinePG = Green + Y * Width;
		unsigned char *LinePR = Red + Y * Width;
		for (int X = 0; X < Width; X++, LinePS += 3)
		{
			LinePB[X] = LinePS[0], LinePG[X] = LinePS[1], LinePR[X] = LinePS[2];
		}
	}
}

void IM_CombineRGB(unsigned char *Blue, unsigned char *Green, unsigned char *Red, unsigned char *Dest, int Width, int Height, int Stride)
{
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *LinePD = Dest + Y * Stride;
		unsigned char *LinePB = Blue + Y * Width;
		unsigned char *LinePG = Green + Y * Width;
		unsigned char *LinePR = Red + Y * Width;
		for (int X = 0; X < Width; X++, LinePD += 3)
		{
			LinePD[0] = LinePB[X];
			LinePD[1] = LinePG[X];
			LinePD[2] = LinePR[X];
		}
	}
}


void TV_Curvature_Filter_Original(float *Src, float *Dest, int Width, int Height, int Stride, int StartRow, int StartCol)
{
	for (int Y = StartRow; Y < Height - 1; Y += 2)
	{
		float *LinePC = Src + Y       * Stride;
		float *LinePL = Src + (Y - 1) * Stride;
		float *LinePN = Src + (Y + 1) * Stride;
		float *LinePD = Dest + Y * Stride;
		for (int X = StartCol; X < Width - 1; X += 2)
		{
			float Dist[8] = { 0 };
			float C5 = 5 * LinePC[X];
			Dist[0] = (LinePL[X - 1] + LinePL[X] + LinePC[X - 1] + LinePN[X - 1] + LinePN[X]) - C5;
			Dist[1] = (LinePL[X] + LinePL[X + 1] + LinePC[X + 1] + LinePN[X] + LinePN[X + 1]) - C5;
			Dist[2] = (LinePL[X - 1] + LinePL[X] + LinePL[X + 1] + LinePC[X - 1] + LinePC[X + 1]) - C5;
			Dist[3] = (LinePN[X - 1] + LinePN[X] + LinePN[X + 1] + LinePC[X - 1] + LinePC[X + 1]) - C5;
			Dist[4] = (LinePL[X - 1] + LinePL[X] + LinePL[X + 1] + LinePC[X - 1] + LinePN[X - 1]) - C5;
			Dist[5] = (LinePL[X - 1] + LinePL[X + 1] + LinePL[X + 1] + LinePC[X + 1] + LinePN[X + 1]) - C5;
			Dist[6] = (LinePN[X - 1] + LinePN[X] + LinePN[X + 1] + LinePL[X - 1] + LinePC[X - 1]) - C5;
			Dist[7] = (LinePN[X - 1] + LinePN[X] + LinePN[X + 1] + LinePL[X + 1] + LinePC[X + 1]) - C5;

			float Min = abs(Dist[0]);
			int Index = 0;
			for (int Z = 1; Z < 8; Z++)
			{
				if (abs(Dist[Z]) < Min)
				{
					Min = abs(Dist[Z]);
					Index = Z;
				}
			}
			LinePD[X] = LinePC[X] + Dist[Index] * 0.2f;
		}
	}
}


void IM_TV_CurvatureFilter_Original(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride, int Iteration)
{
	int Channel = Stride / Width;
	if (Channel == 3)
	{
		unsigned char *Blue = (unsigned char *)malloc(Width * Height * sizeof(unsigned char));
		unsigned char *Green = (unsigned char *)malloc(Width * Height * sizeof(unsigned char));
		unsigned char *Red = (unsigned char *)malloc(Width * Height * sizeof(unsigned char));
		IM_SplitRGB(Src, Blue, Green, Red, Width, Height, Stride);
		IM_TV_CurvatureFilter_Original(Blue, Blue, Width, Height, Width, Iteration);
		IM_TV_CurvatureFilter_Original(Green, Green, Width, Height, Width, Iteration);
		IM_TV_CurvatureFilter_Original(Red, Red, Width, Height, Width, Iteration);
		IM_CombineRGB(Blue, Green, Red, Dest, Width, Height, Stride);
		free(Blue);
		free(Green);
		free(Red);
	}
	else
	{
		float *Temp1 = (float *)malloc(Height * Stride * sizeof(float));
		float *Temp2 = (float *)malloc(Height * Stride * sizeof(float));

		for (int Y = 0; Y < Height * Stride; Y++)
			Temp1[Y] = Src[Y];
		memcpy(Temp2, Temp1, Height * Stride * sizeof(float));

		for (int Y = 0; Y < Iteration; Y++)
		{
			TV_Curvature_Filter_Original(Temp1, Temp2, Width, Height, Stride, 1, 1);
			memcpy(Temp1, Temp2, Height * Stride * sizeof(float));

			TV_Curvature_Filter_Original(Temp1, Temp2, Width, Height, Stride, 2, 2);
			memcpy(Temp1, Temp2, Height * Stride * sizeof(float));

			TV_Curvature_Filter_Original(Temp1, Temp2, Width, Height, Stride, 1, 2);
			memcpy(Temp1, Temp2, Height * Stride * sizeof(float));

			TV_Curvature_Filter_Original(Temp1, Temp2, Width, Height, Stride, 2, 1);
			memcpy(Temp1, Temp2, Height * Stride * sizeof(float));
		}
		for (int Y = 0; Y < Height * Stride; Y++)
			Dest[Y] = IM_ClampToByte((int)(Temp2[Y] + 0.4999999f));
		free(Temp1);
		free(Temp2);
	}
}


void TV_Curvature_Filter_Float_SSE(float *Data, int Width, int Height, int Stride)
{
	int Channel = Stride / Width;
	int BlockSize = 4, Block = (Width * Channel) / BlockSize;

	float *RowCopy = (float *)malloc((Width + 2) * 3 * Channel * sizeof(float));

	float *First = RowCopy;
	float *Second = RowCopy + (Width + 2) * Channel;
	float *Third = RowCopy + (Width + 2) * 2 * Channel;
	
	memcpy(Second, Data, Channel * sizeof(float));
	memcpy(Second + Channel, Data, Width * Channel * sizeof(float));											//	拷贝数据到中间位置
	memcpy(Second + (Width + 1) * Channel, Data + (Width - 1) * Channel, Channel * sizeof(float));
	
	memcpy(First, Second, (Width + 2) * Channel * sizeof(float));												//	第一行和第二行一样
	
	memcpy(Third, Data + Stride, Channel * sizeof(float));												//	拷贝第二行数据
	memcpy(Third + Channel, Data + Stride, Width * Channel * sizeof(float));
	memcpy(Third + (Width + 1) * Channel, Data + Stride + (Width - 1) * Channel, Channel * sizeof(float));
	
	
	for (int Y = 0; Y < Height; Y++)
	{
		float *LinePD = Data + Y * Stride;
		if (Y != 0)
		{
			float *Temp = First;
			First = Second;
			Second = Third;
			Third = Temp;
		}
		if (Y == Height - 1)
		{
			memcpy(Third, Second, (Width + 2) * Channel* sizeof(float));
		}
		else
		{
			memcpy(Third, Data + (Y + 1) * Stride, Channel* sizeof(float));
			memcpy(Third + Channel, Data + (Y + 1) * Stride, Width * Channel* sizeof(float));									//	由于备份了前面一行的数据，这里即使Data和Dest相同也是没有问题的
			memcpy(Third + (Width + 1) * Channel, Data + (Y + 1) * Stride + (Width - 1) * Channel, Channel* sizeof(float));
		}
	
		for (int X = 0; X < Block * BlockSize; X += BlockSize)
		{
			__m128 FirstP0 = _mm_loadu_ps(First + X);
			__m128 FirstP1 = _mm_loadu_ps(First + X + Channel);
			__m128 FirstP2 = _mm_loadu_ps(First + X + Channel + Channel);

			__m128 SecondP0 = _mm_loadu_ps(Second + X);
			__m128 SecondP1 = _mm_loadu_ps(Second + X + Channel);
			__m128 SecondP2 = _mm_loadu_ps(Second + X + Channel + Channel);

			__m128 ThirdP0 = _mm_loadu_ps(Third + X);
			__m128 ThirdP1 = _mm_loadu_ps(Third + X + Channel);
			__m128 ThirdP2 = _mm_loadu_ps(Third + X + Channel + Channel);

			__m128 C5 = _mm_mul_ps(SecondP1, _mm_set1_ps(5));
			__m128 Dist0 = _mm_sub_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(FirstP0, FirstP1), _mm_add_ps(FirstP2, SecondP2)), ThirdP2), C5);
			__m128 Dist1 = _mm_sub_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(FirstP1, FirstP2), _mm_add_ps(SecondP2, ThirdP2)), ThirdP1), C5);
			__m128 Dist2 = _mm_sub_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(FirstP2, SecondP2), _mm_add_ps(ThirdP2, ThirdP1)), ThirdP0), C5);
			__m128 Dist3 = _mm_sub_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(SecondP2, ThirdP2), _mm_add_ps(ThirdP1, ThirdP0)), SecondP0), C5);
			__m128 Dist4 = _mm_sub_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(ThirdP2, ThirdP1), _mm_add_ps(ThirdP0, SecondP0)), FirstP0), C5);
			__m128 Dist5 = _mm_sub_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(ThirdP1, ThirdP0), _mm_add_ps(SecondP0, FirstP0)), FirstP1), C5);
			__m128 Dist6 = _mm_sub_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(ThirdP0, SecondP0), _mm_add_ps(FirstP0, FirstP1)), FirstP2), C5);
			__m128 Dist7 = _mm_sub_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(SecondP0, FirstP0), _mm_add_ps(FirstP1, FirstP2)), FirstP1), C5);

			//__m128 Temp1 = _mm_add_ps(_mm_add_ps(FirstP1, FirstP2), _mm_add_ps(SecondP2, ThirdP2));
			//__m128 Dist0 = _mm_sub_ps(_mm_add_ps(FirstP0, Temp1), C5);
			//__m128 Dist1 = _mm_sub_ps(_mm_add_ps(Temp1, ThirdP1), C5);
			//__m128 Temp2 = _mm_add_ps(_mm_add_ps(ThirdP0, ThirdP1), _mm_add_ps(ThirdP2, SecondP2));
			//__m128 Dist2 = _mm_sub_ps(_mm_add_ps(Temp2, FirstP2), C5);
			//__m128 Dist3 = _mm_sub_ps(_mm_add_ps(Temp2, SecondP0), C5);
			//__m128 Temp3 = _mm_add_ps(_mm_add_ps(FirstP0, SecondP0), _mm_add_ps(ThirdP0, ThirdP1));
			//__m128 Dist4 = _mm_sub_ps(_mm_add_ps(Temp3, ThirdP2), C5);
			//__m128 Dist5 = _mm_sub_ps(_mm_add_ps(Temp3, FirstP1), C5);
			//__m128 Temp4 = _mm_add_ps(_mm_add_ps(SecondP0, FirstP0), _mm_add_ps(FirstP1, FirstP2));
			//__m128 Dist6 = _mm_sub_ps(_mm_add_ps(Temp4, ThirdP0), C5);
			//__m128 Dist7 = _mm_sub_ps(_mm_add_ps(Temp4, SecondP2), C5);

			__m128 AbsDist0 = _mm_abs_ps(Dist0);
			__m128 AbsDist1 = _mm_abs_ps(Dist1);
			__m128 AbsDist2 = _mm_abs_ps(Dist2);
			__m128 AbsDist3 = _mm_abs_ps(Dist3);
			__m128 AbsDist4 = _mm_abs_ps(Dist4);
			__m128 AbsDist5 = _mm_abs_ps(Dist5);
			__m128 AbsDist6 = _mm_abs_ps(Dist6);
			__m128 AbsDist7 = _mm_abs_ps(Dist7);

			__m128 Min = _mm_min_ps(AbsDist0, AbsDist1);
			__m128 Value = _mm_blendv_ps(Dist0, Dist1, _mm_cmpeq_ps(Min, AbsDist1));
			Min = _mm_min_ps(Min, AbsDist2);
			Value = _mm_blendv_ps(Value, Dist2, _mm_cmpeq_ps(Min, AbsDist2));
			Min = _mm_min_ps(Min, AbsDist3);
			Value = _mm_blendv_ps(Value, Dist3, _mm_cmpeq_ps(Min, AbsDist3));
			Min = _mm_min_ps(Min, AbsDist4);
			Value = _mm_blendv_ps(Value, Dist4, _mm_cmpeq_ps(Min, AbsDist4));
			Min = _mm_min_ps(Min, AbsDist5);
			Value = _mm_blendv_ps(Value, Dist5, _mm_cmpeq_ps(Min, AbsDist5));
			Min = _mm_min_ps(Min, AbsDist6);
			Value = _mm_blendv_ps(Value, Dist6, _mm_cmpeq_ps(Min, AbsDist6));
			Min = _mm_min_ps(Min, AbsDist7);
			Value = _mm_blendv_ps(Value, Dist7, _mm_cmpeq_ps(Min, AbsDist7));

			_mm_storeu_ps(LinePD + X, _mm_add_ps(_mm_mul_ps(Value, _mm_set1_ps(0.20f)), SecondP1));

		}
		for (int X = Block * BlockSize; X < Width; X++)
		{
			//	TODO may some code...
		}
	}
}

void TV_Curvature_Filter_Float_AVX(float *Data, int Width, int Height, int Stride)
{
	int Channel = Stride / Width;
	int BlockSize = 8;
	int Block = (Width * Channel) / BlockSize;

	float *RowCopy = (float *)malloc((Width + 2) * 3 * Channel * sizeof(float));

	float *First = RowCopy;
	float *Second = RowCopy + (Width + 2) * Channel;
	float *Third = RowCopy + (Width + 2) * 2 * Channel;

	// 拷贝数据到中间位置，两边各扩展1个像素
	memcpy(Second, Data, Channel * sizeof(float));
	memcpy(Second + Channel, Data, Width * Channel * sizeof(float));
	memcpy(Second + (Width + 1) * Channel, Data + (Width - 1) * Channel, Channel * sizeof(float));

	// 第一行和第二行一样
	memcpy(First, Second, (Width + 2) * Channel * sizeof(float));

	// 拷贝Third数据，与Second类似
	// memcpy(Third, Data + Stride, Channel * sizeof(float));
	// memcpy(Third + Channel, Data + Stride, Width * Channel * sizeof(float));
	// memcpy(Third + (Width + 1) * Channel, Data + Stride + (Width - 1) * Channel, Channel * sizeof(float));

	for (int Y = 0; Y < Height; Y++)
	{
		float *LinePD = Data + Y * Stride;
		if (Y != 0)
		{
			float *Temp = First;
			First = Second;
			Second = Third;
			Third = Temp;
		}
		if (Y == Height - 1) {
			memcpy(Third, Second, (Width + 2) * Channel* sizeof(float));
		}
		else {
			memcpy(Third, Data + (Y + 1) * Stride, Channel* sizeof(float));
			// 由于备份了前面一行的数据，这里即使Data和Dest相同也是没有问题的
			memcpy(Third + Channel, Data + (Y + 1) * Stride, Width * Channel* sizeof(float));
			memcpy(Third + (Width + 1) * Channel, Data + (Y + 1) * Stride + (Width - 1) * Channel, Channel* sizeof(float));
		}

		for (int X = 0; X < Block * BlockSize; X += BlockSize)
		{
			__m256 FirstP0 = _mm256_loadu_ps(First + X);
			__m256 FirstP1 = _mm256_loadu_ps(First + X + Channel);
			__m256 FirstP2 = _mm256_loadu_ps(First + X + Channel + Channel);

			__m256 SecondP0 = _mm256_loadu_ps(Second + X);
			__m256 SecondP1 = _mm256_loadu_ps(Second + X + Channel);
			__m256 SecondP2 = _mm256_loadu_ps(Second + X + Channel + Channel);

			__m256 ThirdP0 = _mm256_loadu_ps(Third + X);
			__m256 ThirdP1 = _mm256_loadu_ps(Third + X + Channel);
			__m256 ThirdP2 = _mm256_loadu_ps(Third + X + Channel + Channel);

			__m256 C5 = _mm256_mul_ps(SecondP1, _mm256_set1_ps(5));
			__m256 Dist0 = _mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(FirstP0, FirstP1), _mm256_add_ps(FirstP2, SecondP2)), ThirdP2), C5);
			__m256 Dist1 = _mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(FirstP1, FirstP2), _mm256_add_ps(SecondP2, ThirdP2)), ThirdP1), C5);
			__m256 Dist2 = _mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(FirstP2, SecondP2), _mm256_add_ps(ThirdP2, ThirdP1)), ThirdP0), C5);
			__m256 Dist3 = _mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(SecondP2, ThirdP2), _mm256_add_ps(ThirdP1, ThirdP0)), SecondP0), C5);
			__m256 Dist4 = _mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(ThirdP2, ThirdP1), _mm256_add_ps(ThirdP0, SecondP0)), FirstP0), C5);
			__m256 Dist5 = _mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(ThirdP1, ThirdP0), _mm256_add_ps(SecondP0, FirstP0)), FirstP1), C5);
			__m256 Dist6 = _mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(ThirdP0, SecondP0), _mm256_add_ps(FirstP0, FirstP1)), FirstP2), C5);
			__m256 Dist7 = _mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(SecondP0, FirstP0), _mm256_add_ps(FirstP1, FirstP2)), FirstP1), C5);

			__m256 AbsDist0 = _mm256_abs_ps(Dist0);
			__m256 AbsDist1 = _mm256_abs_ps(Dist1);
			__m256 AbsDist2 = _mm256_abs_ps(Dist2);
			__m256 AbsDist3 = _mm256_abs_ps(Dist3);
			__m256 AbsDist4 = _mm256_abs_ps(Dist4);
			__m256 AbsDist5 = _mm256_abs_ps(Dist5);
			__m256 AbsDist6 = _mm256_abs_ps(Dist6);
			__m256 AbsDist7 = _mm256_abs_ps(Dist7);

			__m256 AbsMin;
			__m256 Value;
			AbsMin = _mm256_min_ps(AbsDist0, AbsDist1);
			Value = _mm256_blendv_ps(Dist0, Dist1, _mm256_cmp_ps(AbsMin, AbsDist1, _CMP_EQ_US));
			AbsMin = _mm256_min_ps(AbsMin, AbsDist2);
			Value = _mm256_blendv_ps(Value, Dist2, _mm256_cmp_ps(AbsMin, AbsDist2, _CMP_EQ_US));
			AbsMin = _mm256_min_ps(AbsMin, AbsDist3);
			Value = _mm256_blendv_ps(Value, Dist3, _mm256_cmp_ps(AbsMin, AbsDist3, _CMP_EQ_US));
			AbsMin = _mm256_min_ps(AbsMin, AbsDist4);
			Value = _mm256_blendv_ps(Value, Dist4, _mm256_cmp_ps(AbsMin, AbsDist4, _CMP_EQ_US));
			AbsMin = _mm256_min_ps(AbsMin, AbsDist5);
			Value = _mm256_blendv_ps(Value, Dist5, _mm256_cmp_ps(AbsMin, AbsDist5, _CMP_EQ_US));
			AbsMin = _mm256_min_ps(AbsMin, AbsDist6);
			Value = _mm256_blendv_ps(Value, Dist6, _mm256_cmp_ps(AbsMin, AbsDist6, _CMP_EQ_US));
			AbsMin = _mm256_min_ps(AbsMin, AbsDist7);
			Value = _mm256_blendv_ps(Value, Dist7, _mm256_cmp_ps(AbsMin, AbsDist7, _CMP_EQ_US));

			_mm256_storeu_ps(LinePD + X, _mm256_add_ps(_mm256_mul_ps(Value, _mm256_set1_ps(0.20f)), SecondP1));

		}

		// 整除block之后的余数部分，另外处理
		//	TODO may some code...
		// cout << "remain:" << Width * Channel - (Block * BlockSize)<<  "  need be deal with..." << endl;
		for (int X = Block * BlockSize; X < Width; X++)
		{

		}
	}
	_mm256_zeroupper();
}


void IM_TV_CurvatureFilter_Float(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride, int Iteration, bool UseAvx)
{
	float *Temp = (float *)malloc(Height * Stride * sizeof(float));
	for (int Y = 0; Y < Height * Stride; Y++)
		Temp[Y] = Src[Y];

	for (int Y = 0; Y < Iteration; Y++)
	{
		if (UseAvx == true)
			TV_Curvature_Filter_Float_AVX(Temp, Width, Height, Stride);
		else
			TV_Curvature_Filter_Float_SSE(Temp, Width, Height, Stride);
	}
	for (int Y = 0; Y < Height * Stride; Y++)
		Dest[Y] = IM_ClampToByte((int)(Temp[Y] + 0.4999999f));
	free(Temp);
}


void TV_Curvature_Filter_Short_SSE(short *Data, int Width, int Height, int Stride)
{
	int Channel = Stride / Width;

	short Multiplier5 = 0;
	int Shift5 = 0, Sign5 = 0;
	GetMSS_S(5, Multiplier5, Shift5, Sign5);

	int BlockSize = 8, Block = (Width * Channel) / BlockSize;

	short *RowCopy = (short *)malloc((Width + 2) * 3 * Channel * sizeof(short));

	short *First = RowCopy;
	short *Second = RowCopy + (Width + 2) * Channel;
	short *Third = RowCopy + (Width + 2) * 2 * Channel;

	memcpy(Second, Data, Channel * sizeof(short));
	memcpy(Second + Channel, Data, Width * Channel * sizeof(short));											//	拷贝数据到中间位置
	memcpy(Second + (Width + 1) * Channel, Data + (Width - 1) * Channel, Channel * sizeof(short));

	memcpy(First, Second, (Width + 2) * Channel * sizeof(short));												//	第一行和第二行一样

	memcpy(Third, Data + Stride, Channel * sizeof(short));												//	拷贝第二行数据
	memcpy(Third + Channel, Data + Stride, Width * Channel * sizeof(short));
	memcpy(Third + (Width + 1) * Channel, Data + Stride + (Width - 1) * Channel, Channel * sizeof(short));


	for (int Y = 0; Y < Height; Y++)
	{
		short *LinePD = Data + Y * Stride;
		if (Y != 0)
		{
			short *Temp = First; First = Second; Second = Third; Third = Temp;
		}
		if (Y == Height - 1)
		{
			memcpy(Third, Second, (Width + 2) * Channel * sizeof(short));
		}
		else
		{
			memcpy(Third, Data + (Y + 1) * Stride, Channel* sizeof(short));
			memcpy(Third + Channel, Data + (Y + 1) * Stride, Width * Channel* sizeof(short));									//	由于备份了前面一行的数据，这里即使Data和Dest相同也是没有问题的
			memcpy(Third + (Width + 1) * Channel, Data + (Y + 1) * Stride + (Width - 1) * Channel, Channel* sizeof(short));
		}

		for (int X = 0; X < Block * BlockSize; X += BlockSize)
		{
			__m128i FirstP0 = _mm_loadu_si128((__m128i *)(First + X));
			__m128i FirstP1 = _mm_loadu_si128((__m128i *)(First + X + Channel));
			__m128i FirstP2 = _mm_loadu_si128((__m128i *)(First + X + Channel + Channel));		//	

			__m128i SecondP0 = _mm_loadu_si128((__m128i *)(Second + X));
			__m128i SecondP1 = _mm_loadu_si128((__m128i *)(Second + X + Channel));
			__m128i SecondP2 = _mm_loadu_si128((__m128i *)(Second + X + Channel + Channel));

			__m128i ThirdP0 = _mm_loadu_si128((__m128i *)(Third + X));
			__m128i ThirdP1 = _mm_loadu_si128((__m128i *)(Third + X + Channel));
			__m128i ThirdP2 = _mm_loadu_si128((__m128i *)(Third + X + Channel + Channel));

			__m128i C5 = _mm_mullo_epi16(SecondP1, _mm_set1_epi16(5));

			__m128i Temp1 = _mm_add_epi16(_mm_add_epi16(FirstP1, FirstP2), _mm_add_epi16(SecondP2, ThirdP2));
			__m128i Dist0 = _mm_sub_epi16(_mm_add_epi16(FirstP0, Temp1), C5);
			__m128i Dist1 = _mm_sub_epi16(_mm_add_epi16(Temp1, ThirdP1), C5);
			__m128i Temp2 = _mm_add_epi16(_mm_add_epi16(ThirdP0, ThirdP1), _mm_add_epi16(ThirdP2, SecondP2));
			__m128i Dist2 = _mm_sub_epi16(_mm_add_epi16(Temp2, FirstP2), C5);
			__m128i Dist3 = _mm_sub_epi16(_mm_add_epi16(Temp2, SecondP0), C5);
			__m128i Temp3 = _mm_add_epi16(_mm_add_epi16(FirstP0, SecondP0), _mm_add_epi16(ThirdP0, ThirdP1));
			__m128i Dist4 = _mm_sub_epi16(_mm_add_epi16(Temp3, ThirdP2), C5);
			__m128i Dist5 = _mm_sub_epi16(_mm_add_epi16(Temp3, FirstP1), C5);
			__m128i Temp4 = _mm_add_epi16(_mm_add_epi16(SecondP0, FirstP0), _mm_add_epi16(FirstP1, FirstP2));
			__m128i Dist6 = _mm_sub_epi16(_mm_add_epi16(Temp4, ThirdP0), C5);
			__m128i Dist7 = _mm_sub_epi16(_mm_add_epi16(Temp4, SecondP2), C5);

			__m128i AbsDist0 = _mm_abs_epi16(Dist0);
			__m128i AbsDist1 = _mm_abs_epi16(Dist1);
			__m128i AbsDist2 = _mm_abs_epi16(Dist2);
			__m128i AbsDist3 = _mm_abs_epi16(Dist3);
			__m128i AbsDist4 = _mm_abs_epi16(Dist4);
			__m128i AbsDist5 = _mm_abs_epi16(Dist5);
			__m128i AbsDist6 = _mm_abs_epi16(Dist6);
			__m128i AbsDist7 = _mm_abs_epi16(Dist7);

			__m128i Cmp = _mm_cmpgt_epi16(AbsDist0, AbsDist1);
			__m128i Min = _mm_blendv_epi8(AbsDist0, AbsDist1, Cmp);
			__m128i Value = _mm_blendv_epi8(Dist0, Dist1, Cmp);

			Cmp = _mm_cmpgt_epi16(Min, AbsDist2);
			Min = _mm_blendv_epi8(Min, AbsDist2, Cmp);
			Value = _mm_blendv_epi8(Value, Dist2, Cmp);

			Cmp = _mm_cmpgt_epi16(Min, AbsDist3);
			Min = _mm_blendv_epi8(Min, AbsDist3, Cmp);
			Value = _mm_blendv_epi8(Value, Dist3, Cmp);

			Cmp = _mm_cmpgt_epi16(Min, AbsDist4);
			Min = _mm_blendv_epi8(Min, AbsDist4, Cmp);
			Value = _mm_blendv_epi8(Value, Dist4, Cmp);

			Cmp = _mm_cmpgt_epi16(Min, AbsDist5);
			Min = _mm_blendv_epi8(Min, AbsDist5, Cmp);
			Value = _mm_blendv_epi8(Value, Dist5, Cmp);

			Cmp = _mm_cmpgt_epi16(Min, AbsDist6);
			Min = _mm_blendv_epi8(Min, AbsDist6, Cmp);
			Value = _mm_blendv_epi8(Value, Dist6, Cmp);

			Cmp = _mm_cmpgt_epi16(Min, AbsDist7);
			Min = _mm_blendv_epi8(Min, AbsDist7, Cmp);
			Value = _mm_blendv_epi8(Value, Dist7, Cmp);

			__m128i DivY = _mm_divp_epi16(_mm_add_epi16(_mm_set1_epi16(2), Value), Multiplier5, Shift5);
			_mm_storeu_si128((__m128i *)(LinePD + X), _mm_add_epi16(DivY, SecondP1));

		}
		for (int X = Block * BlockSize; X < Width; X++)
		{
			//	请自行添加
		}
	}
	free(RowCopy);
}

void TV_Curvature_Filter_Short_AVX(short *Data, int Width, int Height, int Stride)
{
	int Channel = Stride / Width;

	short Multiplier5 = 0;
	int Shift5 = 0, Sign5 = 0;
	GetMSS_S(5, Multiplier5, Shift5, Sign5);

	int BlockSize = 16, Block = (Width * Channel) / BlockSize;

	short *RowCopy = (short *)malloc((Width + 2) * 3 * Channel * sizeof(short));

	short *First = RowCopy;
	short *Second = RowCopy + (Width + 2) * Channel;
	short *Third = RowCopy + (Width + 2) * 2 * Channel;

	memcpy(Second, Data, Channel * sizeof(short));
	memcpy(Second + Channel, Data, Width * Channel * sizeof(short));											//	拷贝数据到中间位置
	memcpy(Second + (Width + 1) * Channel, Data + (Width - 1) * Channel, Channel * sizeof(short));

	memcpy(First, Second, (Width + 2) * Channel * sizeof(short));												//	第一行和第二行一样

	memcpy(Third, Data + Stride, Channel * sizeof(short));												//	拷贝第二行数据
	memcpy(Third + Channel, Data + Stride, Width * Channel * sizeof(short));
	memcpy(Third + (Width + 1) * Channel, Data + Stride + (Width - 1) * Channel, Channel * sizeof(short));


	for (int Y = 0; Y < Height; Y++)
	{
		short *LinePD = Data + Y * Stride;
		if (Y != 0)
		{
			short *Temp = First; First = Second; Second = Third; Third = Temp;
		}
		if (Y == Height - 1)
		{
			memcpy(Third, Second, (Width + 2) * Channel * sizeof(short));
		}
		else
		{
			memcpy(Third, Data + (Y + 1) * Stride, Channel* sizeof(short));
			memcpy(Third + Channel, Data + (Y + 1) * Stride, Width * Channel* sizeof(short));									//	由于备份了前面一行的数据，这里即使Data和Dest相同也是没有问题的
			memcpy(Third + (Width + 1) * Channel, Data + (Y + 1) * Stride + (Width - 1) * Channel, Channel* sizeof(short));
		}

		for (int X = 0; X < Block * BlockSize; X += BlockSize)
		{
			__m256i FirstP0 = _mm256_loadu_si256((__m256i *)(First + X));
			__m256i FirstP1 = _mm256_loadu_si256((__m256i *)(First + X + Channel));
			__m256i FirstP2 = _mm256_loadu_si256((__m256i *)(First + X + Channel + Channel));

			__m256i SecondP0 = _mm256_loadu_si256((__m256i *)(Second + X));
			__m256i SecondP1 = _mm256_loadu_si256((__m256i *)(Second + X + Channel));
			__m256i SecondP2 = _mm256_loadu_si256((__m256i *)(Second + X + Channel + Channel));

			__m256i ThirdP0 = _mm256_loadu_si256((__m256i *)(Third + X));
			__m256i ThirdP1 = _mm256_loadu_si256((__m256i *)(Third + X + Channel));
			__m256i ThirdP2 = _mm256_loadu_si256((__m256i *)(Third + X + Channel + Channel));

			__m256i C5 = _mm256_mullo_epi16(SecondP1, _mm256_set1_epi16(5));

			__m256i Temp1 = _mm256_add_epi16(_mm256_add_epi16(FirstP1, FirstP2), _mm256_add_epi16(SecondP2, ThirdP2));
			__m256i Dist0 = _mm256_sub_epi16(_mm256_add_epi16(FirstP0, Temp1), C5);
			__m256i Dist1 = _mm256_sub_epi16(_mm256_add_epi16(Temp1, ThirdP1), C5);
			__m256i Temp2 = _mm256_add_epi16(_mm256_add_epi16(ThirdP0, ThirdP1), _mm256_add_epi16(ThirdP2, SecondP2));
			__m256i Dist2 = _mm256_sub_epi16(_mm256_add_epi16(Temp2, FirstP2), C5);
			__m256i Dist3 = _mm256_sub_epi16(_mm256_add_epi16(Temp2, SecondP0), C5);
			__m256i Temp3 = _mm256_add_epi16(_mm256_add_epi16(FirstP0, SecondP0), _mm256_add_epi16(ThirdP0, ThirdP1));
			__m256i Dist4 = _mm256_sub_epi16(_mm256_add_epi16(Temp3, ThirdP2), C5);
			__m256i Dist5 = _mm256_sub_epi16(_mm256_add_epi16(Temp3, FirstP1), C5);
			__m256i Temp4 = _mm256_add_epi16(_mm256_add_epi16(SecondP0, FirstP0), _mm256_add_epi16(FirstP1, FirstP2));
			__m256i Dist6 = _mm256_sub_epi16(_mm256_add_epi16(Temp4, ThirdP0), C5);
			__m256i Dist7 = _mm256_sub_epi16(_mm256_add_epi16(Temp4, SecondP2), C5);

			__m256i AbsDist0 = _mm256_abs_epi16(Dist0);
			__m256i AbsDist1 = _mm256_abs_epi16(Dist1);
			__m256i AbsDist2 = _mm256_abs_epi16(Dist2);
			__m256i AbsDist3 = _mm256_abs_epi16(Dist3);
			__m256i AbsDist4 = _mm256_abs_epi16(Dist4);
			__m256i AbsDist5 = _mm256_abs_epi16(Dist5);
			__m256i AbsDist6 = _mm256_abs_epi16(Dist6);
			__m256i AbsDist7 = _mm256_abs_epi16(Dist7);

			__m256i Cmp = _mm256_cmpgt_epi16(AbsDist0, AbsDist1);
			__m256i Min = _mm256_blendv_epi8(AbsDist0, AbsDist1, Cmp);
			__m256i Value = _mm256_blendv_epi8(Dist0, Dist1, Cmp);

			Cmp = _mm256_cmpgt_epi16(Min, AbsDist2);
			Min = _mm256_blendv_epi8(Min, AbsDist2, Cmp);
			Value = _mm256_blendv_epi8(Value, Dist2, Cmp);

			Cmp = _mm256_cmpgt_epi16(Min, AbsDist3);
			Min = _mm256_blendv_epi8(Min, AbsDist3, Cmp);
			Value = _mm256_blendv_epi8(Value, Dist3, Cmp);

			Cmp = _mm256_cmpgt_epi16(Min, AbsDist4);
			Min = _mm256_blendv_epi8(Min, AbsDist4, Cmp);
			Value = _mm256_blendv_epi8(Value, Dist4, Cmp);

			Cmp = _mm256_cmpgt_epi16(Min, AbsDist5);
			Min = _mm256_blendv_epi8(Min, AbsDist5, Cmp);
			Value = _mm256_blendv_epi8(Value, Dist5, Cmp);

			Cmp = _mm256_cmpgt_epi16(Min, AbsDist6);
			Min = _mm256_blendv_epi8(Min, AbsDist6, Cmp);
			Value = _mm256_blendv_epi8(Value, Dist6, Cmp);

			Cmp = _mm256_cmpgt_epi16(Min, AbsDist7);
			Min = _mm256_blendv_epi8(Min, AbsDist7, Cmp);
			Value = _mm256_blendv_epi8(Value, Dist7, Cmp);

			__m256i DivY = _mm256_divp_epi16(_mm256_add_epi16(_mm256_set1_epi16(2), Value), Multiplier5, Shift5);
			_mm256_storeu_si256((__m256i *)(LinePD + X), _mm256_add_epi16(DivY, SecondP1));

		}
		for (int X = Block * BlockSize; X < Width; X++)
		{
			//	请自行添加
		}
	}
	free(RowCopy);
	_mm256_zeroupper();				//	必须加这一句，不然SSE版本代码会严重减速
}

#ifdef AVX512
void TV_Curvature_Filter_Short_AVX2(short *Data, int Width, int Height, int Stride)
{
	int Channel = Stride / Width;

	short Multiplier5 = 0;
	int Shift5 = 0, Sign5 = 0;
	GetMSS_S(5, Multiplier5, Shift5, Sign5);

	int BlockSize = 32;
	int Block = (Width * Channel) / BlockSize;

	short *RowCopy = (short *)malloc((Width + 2) * 3 * Channel * sizeof(short));

	short *First = RowCopy;
	short *Second = RowCopy + (Width + 2) * Channel;
	short *Third = RowCopy + (Width + 2) * 2 * Channel;

	memcpy(Second, Data, Channel * sizeof(short));
	memcpy(Second + Channel, Data, Width * Channel * sizeof(short));											//	拷贝数据到中间位置
	memcpy(Second + (Width + 1) * Channel, Data + (Width - 1) * Channel, Channel * sizeof(short));

	memcpy(First, Second, (Width + 2) * Channel * sizeof(short));												//	第一行和第二行一样

	memcpy(Third, Data + Stride, Channel * sizeof(short));												//	拷贝第二行数据
	memcpy(Third + Channel, Data + Stride, Width * Channel * sizeof(short));
	memcpy(Third + (Width + 1) * Channel, Data + Stride + (Width - 1) * Channel, Channel * sizeof(short));

	for (int Y = 0; Y < Height; Y++)
	{
		short *LinePD = Data + Y * Stride;
		if (Y != 0)
		{
			short *Temp = First; First = Second; Second = Third; Third = Temp;
		}
		if (Y == Height - 1)
		{
			memcpy(Third, Second, (Width + 2) * Channel * sizeof(short));
		}
		else
		{
			memcpy(Third, Data + (Y + 1) * Stride, Channel* sizeof(short));
			memcpy(Third + Channel, Data + (Y + 1) * Stride, Width * Channel* sizeof(short));									//	由于备份了前面一行的数据，这里即使Data和Dest相同也是没有问题的
			memcpy(Third + (Width + 1) * Channel, Data + (Y + 1) * Stride + (Width - 1) * Channel, Channel* sizeof(short));
		}

		for (int X = 0; X < Block * BlockSize; X += BlockSize)
		{
			__m512i FirstP0 = _mm512_loadu_si512((__m512i *)(First + X));
			__m512i FirstP1 = _mm512_loadu_si512((__m512i *)(First + X + Channel));
			__m512i FirstP2 = _mm512_loadu_si512((__m512i *)(First + X + Channel + Channel));

			__m512i SecondP0 = _mm512_loadu_si512((__m512i *)(Second + X));
			__m512i SecondP1 = _mm512_loadu_si512((__m512i *)(Second + X + Channel));
			__m512i SecondP2 = _mm512_loadu_si512((__m512i *)(Second + X + Channel + Channel));

			__m512i ThirdP0 = _mm512_loadu_si512((__m512i *)(Third + X));
			__m512i ThirdP1 = _mm512_loadu_si512((__m512i *)(Third + X + Channel));
			__m512i ThirdP2 = _mm512_loadu_si512((__m512i *)(Third + X + Channel + Channel));

			__m512i C5 = _mm512_mullo_epi16(SecondP1, _mm512_set1_epi16(5));

			__m512i Temp1 = _mm512_add_epi16(_mm512_add_epi16(FirstP1, FirstP2), _mm512_add_epi16(SecondP2, ThirdP2));
			__m512i Dist0 = _mm512_sub_epi16(_mm512_add_epi16(FirstP0, Temp1), C5);
			__m512i Dist1 = _mm512_sub_epi16(_mm512_add_epi16(Temp1, ThirdP1), C5);
			__m512i Temp2 = _mm512_add_epi16(_mm512_add_epi16(ThirdP0, ThirdP1), _mm512_add_epi16(ThirdP2, SecondP2));
			__m512i Dist2 = _mm512_sub_epi16(_mm512_add_epi16(Temp2, FirstP2), C5);
			__m512i Dist3 = _mm512_sub_epi16(_mm512_add_epi16(Temp2, SecondP0), C5);
			__m512i Temp3 = _mm512_add_epi16(_mm512_add_epi16(FirstP0, SecondP0), _mm512_add_epi16(ThirdP0, ThirdP1));
			__m512i Dist4 = _mm512_sub_epi16(_mm512_add_epi16(Temp3, ThirdP2), C5);
			__m512i Dist5 = _mm512_sub_epi16(_mm512_add_epi16(Temp3, FirstP1), C5);
			__m512i Temp4 = _mm512_add_epi16(_mm512_add_epi16(SecondP0, FirstP0), _mm512_add_epi16(FirstP1, FirstP2));
			__m512i Dist6 = _mm512_sub_epi16(_mm512_add_epi16(Temp4, ThirdP0), C5);
			__m512i Dist7 = _mm512_sub_epi16(_mm512_add_epi16(Temp4, SecondP2), C5);

			__m512i AbsDist0 = _mm512_abs_epi16(Dist0);
			__m512i AbsDist1 = _mm512_abs_epi16(Dist1);
			__m512i AbsDist2 = _mm512_abs_epi16(Dist2);
			__m512i AbsDist3 = _mm512_abs_epi16(Dist3);
			__m512i AbsDist4 = _mm512_abs_epi16(Dist4);
			__m512i AbsDist5 = _mm512_abs_epi16(Dist5);
			__m512i AbsDist6 = _mm512_abs_epi16(Dist6);
			__m512i AbsDist7 = _mm512_abs_epi16(Dist7);

			__mmask32 Cmp = _mm512_cmpgt_epi16_mask(AbsDist0, AbsDist1);
			__m512i Min = _mm512_mask_blend_epi16(Cmp, AbsDist0, AbsDist1);
			__m512i Value = _mm512_mask_blend_epi16(Cmp, Dist0, Dist1);

			Cmp = _mm512_cmpgt_epi16_mask(Min, AbsDist2);
			Min = _mm512_mask_blend_epi16(Cmp, Min, AbsDist2);
			Value = _mm512_mask_blend_epi16(Cmp, Value, Dist2);

			Cmp = _mm512_cmpgt_epi16_mask(Min, AbsDist3);
			Min = _mm512_mask_blend_epi16(Cmp, Min, AbsDist3);
			Value = _mm512_mask_blend_epi16(Cmp, Value, Dist3);

			Cmp = _mm512_cmpgt_epi16_mask(Min, AbsDist4);
			Min = _mm512_mask_blend_epi16(Cmp, Min, AbsDist4);
			Value = _mm512_mask_blend_epi16(Cmp, Value, Dist4);

			Cmp = _mm512_cmpgt_epi16_mask(Min, AbsDist5);
			Min = _mm512_mask_blend_epi16(Cmp, Min, AbsDist5);
			Value = _mm512_mask_blend_epi16(Cmp, Value, Dist5);

			Cmp = _mm512_cmpgt_epi16_mask(Min, AbsDist6);
			Min = _mm512_mask_blend_epi16(Cmp, Min, AbsDist6);
			Value = _mm512_mask_blend_epi16(Cmp, Value, Dist6);

			Cmp = _mm512_cmpgt_epi16_mask(Min, AbsDist7);
			Min = _mm512_mask_blend_epi16(Cmp, Min, AbsDist7);
			Value = _mm512_mask_blend_epi16(Cmp, Value, Dist7);

			
			__m512i DivY = _mm512_divp_epi16(_mm512_add_epi16(_mm512_set1_epi16(2), Value), Multiplier5, Shift5);


			_mm512_storeu_si512((__m512i *)(LinePD + X), _mm512_add_epi16(DivY, SecondP1));

		}
	}
	free(RowCopy);
	//	必须加这一句，不然SSE版本代码会严重减速
	_mm256_zeroupper();
}
#endif // AVX512

void IM_TV_CurvatureFilter_Short(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride, int Iteration, bool UseAvx)
{
	short *Temp = (short *)malloc(Height * Stride * sizeof(short));
	for (int Y = 0; Y < Height * Stride; Y++)	Temp[Y] = Src[Y] * 16;

	for (int Y = 0; Y < Iteration; Y++)
	{
		if (UseAvx == true)
			TV_Curvature_Filter_Short_AVX(Temp, Width, Height, Stride);
		else
			TV_Curvature_Filter_Short_SSE(Temp, Width, Height, Stride);
	}
	for (int Y = 0; Y < Height * Stride; Y++)	Dest[Y] = IM_ClampToByte((Temp[Y] + 8) / 16);
	free(Temp);
}

#ifdef AVX512
void IM_TV_CurvatureFilter_Short_AVX512(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride, int Iteration)
{
	short *Temp = (short *)malloc(Height * Stride * sizeof(short));
	for (int Y = 0; Y < Height * Stride; Y++)	Temp[Y] = Src[Y] * 16;

	for (int Y = 0; Y < Iteration; Y++)
	{
		TV_Curvature_Filter_Short_AVX2(Temp, Width, Height, Stride);
	}
	for (int Y = 0; Y < Height * Stride; Y++)	Dest[Y] = IM_ClampToByte((Temp[Y] + 8) / 16);
	free(Temp);
}

#endif // AVX512

// extern "C"{ //在extern “C”中的函数才能被外部调用
	void denoising_test(const char* filepath)
	{
		int iteration = 4;
		clock_t start, finish;
		String out_ori("out_ori.jpg");
		String out_sse_float("out_sse_float.jpg");
		String out_sse_short("out_sse_short.jpg");
		String out_avx_float("out_avx_float.jpg");
		String out_avx_short("out_avx_short.jpg");
		String out_avx512_short("out_avx512_short.jpg");

		start = clock();
		IplImage *img = cvLoadImage(filepath, 1);
		// IplImage *img2 = cvLoadImage(filepath, 1);
		IplImage* img2=cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 3);
		finish = clock();
		cout << "load image:" << (double)(finish - start) / 1000 << "ms" << endl;
		cout << "size:" << img->width << "," << img->height<< endl;

		start = clock();
		IM_TV_CurvatureFilter_Original((unsigned char*)img->imageDataOrigin, (unsigned char*)img2->imageDataOrigin, img->width, img->height, img->widthStep, iteration);
		finish = clock();  
		cout << "original:" << (double)(finish - start) / 1000 << "ms"  << endl;
		cvSaveImage(out_ori.c_str(), img2);

		start = clock();
		IM_TV_CurvatureFilter_Float((unsigned char*)img->imageDataOrigin, (unsigned char*)img2->imageDataOrigin, img->width, img->height, img->widthStep, iteration, false);
		finish = clock();  
		cout << "SSE float:" << (double)(finish - start) / 1000 << "ms"  << endl;
		cvSaveImage(out_sse_float.c_str(), img2);

		start = clock();
		IM_TV_CurvatureFilter_Short((unsigned char*)img->imageDataOrigin, (unsigned char*)img2->imageDataOrigin, img->width, img->height, img->widthStep, iteration, false);
		finish = clock();  
		cout << "SSE short:" << (double)(finish - start) / 1000 << "ms"  << endl;
		cvSaveImage(out_sse_short.c_str(), img2);

		start = clock();
		IM_TV_CurvatureFilter_Float((unsigned char*)img->imageDataOrigin, (unsigned char*)img2->imageDataOrigin, img->width, img->height, img->widthStep, iteration, true);
		finish = clock();  
		cout << "AVX float:" << (double)(finish - start) / 1000 << "ms"  << endl;
		cvSaveImage(out_avx_float.c_str(), img2);

		start = clock();
		IM_TV_CurvatureFilter_Short((unsigned char*)img->imageDataOrigin, (unsigned char*)img2->imageDataOrigin, img->width, img->height, img->widthStep, iteration, true);
		finish = clock();  
		cout << "AVX short:" << (double)(finish - start) / 1000 << "ms"  << endl;
		cvSaveImage(out_avx_short.c_str(), img2);

	#ifdef AVX512
		start = clock();
		IM_TV_CurvatureFilter_Short_AVX512((unsigned char*)img->imageDataOrigin, (unsigned char*)img2->imageDataOrigin, img->width, img->height, img->widthStep, iteration);
		finish = clock();
		cout << "AVX-512 short:" << (double)(finish - start) / 1000 << "ms"  << endl;
		cvSaveImage(out_avx512_short.c_str(), img2);
	#endif // AVX512
	}

	Mat denoising(unsigned char *input_img, int width, int height, int channel, int iteration, bool use_avx)
	{
		clock_t start, finish;
		// unsigned char* out_img = (unsigned char*)malloc(sizeof(char)*width*height*channel);
		Mat out_mat;
		out_mat.create(height, width, CV_8UC3);
		IplImage out_img = IplImage(out_mat);

		// start = clock();
		IM_TV_CurvatureFilter_Float(input_img, (unsigned char*)out_img.imageDataOrigin, width, height, width*channel, iteration, use_avx);
		// finish = clock();
		// cv::Mat img(height, width, CV_8UC3, out_img);

		return out_mat;
	}
// }//匹配extern “C”中大括号 完成整个匹配