#ifndef COLOR_TO_GRAY_H
#define COLOR_TO_GRAY_H

#include <stdint.h>

#define OFFSET_BMP_HEADER_TYPE		0x00
#define OFFSET_BMP_FILE_SIZE		0X02
#define OFFSET_BMP_PIXELS_OFFSET	0X0A

#define OFFSET_BMP_VERSION			0x0E
#define OFFSET_BMP_WIDTH			0x12
#define OFFSET_BMP_HEIGHT			0x16

typedef enum {
	bmp_core	= 12,
	bmp_v3		= 40,
	bmp_v4		= 108,
	bmp_v5		= 124
} BMP_version;

typedef struct BMP_Header_File_t {
	int32_t type;					// BMP format designator.
	int32_t bmp_file_size;			// The whole file size in bytes.
	int32_t rfu;					// Reserved for future use.
	int32_t pixels_offset;			// pixels array.
} BMP_Header_File;

/** Main structure which describes the whole BMP file.
 * The version of the type can be found from its size:
 * CORE - 12 bytes
 * ver3 - 40 bytes
 * ver4 - 108 bytes
 * ver5 - 124 bytes
*/
typedef struct BMP_Info_t {
	int32_t this_size;				// BMP_info size
	int32_t pic_width;				// picture width
	int32_t pic_height;				// picture height
} BMP_Info;

typedef struct BMPicture_t {
	BMP_Header_File header;
	BMP_Info info;
	uint8_t* buff;
} BMPicture;

uint8_t
get_version(BMPicture* picture)
{
	switch (picture->info.this_size)
	{
		case bmp_core:	return bmp_core;
		case bmp_v3:	return bmp_v3;
		case bmp_v4:	return bmp_v4;
		case bmp_v5:	return bmp_v5;
		default:		return 0xFF;
	}
}

#endif /* COLOR_TO_GRAY_H */