#ifndef BMPLIB_H
#define BMPLIB_H

#include <stdint.h>

/**
 * BitmapFileHeader is organized as fixed-size
 * structure of 14 bytes length.
*/
typedef struct {
	uint16_t type;		// 4D42/424D (LE/BE)
	uint32_t size;		// file total size
	uint16_t rfu1;		// always 0
	uint16_t rfu2;		// always 0
	uint32_t offBits;	// offset to pixels from this field
} Header;

/**
 * BitmapInfoHeader.
*/
typedef struct {
	uint32_t size;			// this struct length
	int32_t  width;			// pic width in pixels
	int32_t  height;		// pic height in pixels (Note spec!)
	uint16_t planes;		// always set to '0001'
	uint16_t bitCount;		// number of bits per pixel
	uint32_t compression;	// 
	uint32_t sizeImage;		// total pixels num (excludes meta data)
	int32_t  xPelsPerMeter;	// pixel per meter in X axis
	int32_t  yPelsPerMeter;	// pixel per meter in Y axis
	uint32_t clrUsed;		// color table size in cells
	uint32_t clrImportant;	// color table's cell count
} BitmatInfo;

typedef struct {
	Header header;
	BitmatInfo info;
	uint8_t* pixels;
} Bitmap;
#endif // !BMPLIB_H