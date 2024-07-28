#ifndef COLOR_TO_GRAY_H
#define COLOR_TO_GRAY_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/time.h>

#define OFFSET_BMP_HEADER_MAGIC			0x00
#define OFFSET_BMP_HEADER_FILE_SIZE		0X02
#define OFFSET_BMP_HEADER_PIXELS_OFFSET	0X0A

#define OFFSET_BMP_INFO_VERSION			0x0E
#define OFFSET_BMP_INFO_WIDTH			0x12
#define OFFSET_BMP_INFO_HEIGHT			0x16
#define OFFSET_BMP_INFO_BITCOUNT		0x1C
#define OFFSET_BMP_INFO_IMG_SIZE		0x22

#define ENUM_BMP_VERSION_CORE			12
#define ENUM_BMP_VERSION_3				40
#define ENUM_BMP_VERSION_4				108
#define ENUM_BMP_VERSION_5				124
#define ENUM_BMP_VERSION_UNKNOWN		255

typedef struct BMP_version {
	uint8_t code;
	char* name;
} BMP_version_t;

typedef struct BMP_Header_File_t {
	int32_t magic;			// BMP format designator.
	int32_t file_size;		// The whole file size in bytes.
	int32_t rfu;			// Reserved for future use.
	int32_t pixels_offset;	// beginning of the pixels array.
} BMP_Header_t;

/** Main structure which describes the whole BMP file.
 * The 'type' field actually holds the size of the BMPINFO struct.
 * The version of the BMPINFO type can be found from its size (in bytes):
 * CORE - 12; ver3 - 40; ver4 - 108; ver5 - 124. */
typedef struct BMP_Info_t {
	BMP_Header_t header;

	uint32_t version;
	uint32_t pic_width;		// picture width
	uint32_t pic_height;		// picture height
	uint32_t bit_count;
	uint32_t image_size;		// The size of the pixel array in bytes.
} BMP_Info;

typedef struct BMPicture_t {
	BMP_Info info;
	uint8_t* buff;
} BMP_image;


void bmp_load_file(BMP_image* image, const char* path);
void bmp_init_image(BMP_image* image);
void bmp_print_info(BMP_image* image);
void bmp_free(BMP_image* image);
void bmp_color_to_gray(BMP_image* image);
void bmp_save_file(BMP_image* image, const char* path);

#endif /* COLOR_TO_GRAY_H */