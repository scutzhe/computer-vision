
void CalcGaussCof(float Radius, float &B0, float &B1, float &B2, float &B3);
void ConvertBGR8U2BGRAF(unsigned char *Src, float *Dest, int Width, int Height, int Stride);
void ConvertBGR8U2BGRAF_SSE(unsigned char *Src, float *Dest, int Width, int Height, int Stride);
void GaussBlurFromLeftToRight(float *Data, int Width, int Height, float B0, float B1, float B2, float B3);
void GaussBlurFromLeftToRight_SSE(float *Data, int Width, int Height, float B0, float B1, float B2, float B3);
void GaussBlurFromRightToLeft(float *Data, int Width, int Height, float B0, float B1, float B2, float B3);
void GaussBlurFromRightToLeft_SSE(float *Data, int Width, int Height, float B0, float B1, float B2, float B3);
void GaussBlurFromTopToBottom(float *Data, int Width, int Height, float B0, float B1, float B2, float B3);
void GaussBlurFromTopToBottom_SSE(float *Data, int Width, int Height, float B0, float B1, float B2, float B3);
void GaussBlurFromBottomToTop(float *Data, int Width, int Height, float B0, float B1, float B2, float B3);
void GaussBlurFromBottomToTop_SSE(float *Data, int Width, int Height, float B0, float B1, float B2, float B3);
void ConvertBGRAF2BGR8U(float *Src, unsigned char *Dest, int Width, int Height, int Stride);
void ConvertBGRAF2BGR8U_SSE(float *Src, unsigned char *Dest, int Width, int Height, int Stride);
void GaussBlur(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride, float Radius);
void GaussBlur_SSE(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride, float Radius);

int test_GaussBlur_SSE();