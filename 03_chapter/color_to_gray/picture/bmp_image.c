#include "bmp_image.h"

BMP_version_t const versions[] = {
	{ENUM_BMP_VERSION_CORE,		"Core"},
	{ENUM_BMP_VERSION_3,		"version 3"},
	{ENUM_BMP_VERSION_4,		"version 4"},
	{ENUM_BMP_VERSION_5,		"version 5"},
	{ENUM_BMP_VERSION_UNKNOWN,	"Unknown"},
};

static char*
get_version(BMP_Info* info)
{
	int8_t i;
	for (i = 0; versions[i].code != ENUM_BMP_VERSION_UNKNOWN; ++i) {
		if (versions[i].code == info->version)
			return versions[i].name;
	}

	return versions[i].name;
}

void
bmp_print_info(BMP_image* image)
{
	BMP_Header_t* header = &image->info.header;
	BMP_Info* info = &image->info;

	printf("Magic:             %02X\n", header->magic);
	printf("File size:         %d\n", header->file_size);
	printf("Pixels offset:     %d\n", header->pixels_offset);

	printf("BITMAPINFO version %s\n", get_version(info));
	printf("Picture width:     %d\n", info->pic_width);
	printf("Picture height:    %d\n", info->pic_height);
	printf("Bytes per pixel:   %d\n", info->bit_count);
	printf("Image size:        %d\n", info->image_size);
}

void
bmp_load_file(BMP_image* image, const char* path)
{
	FILE* file;
	size_t bytesRead;
	uint32_t file_size;

	file = fopen(path, "rb");
	if (NULL == file) {
		fprintf(stderr, "Couldn't open source file '%s'.\n", path);
		exit(74);
	}

	fseek(file, 0L, SEEK_END);	/* Move file prt to EOF. */
	file_size = ftell(file);		/* How far we are from start of the file? */
	rewind(file);				/* Rewind file ptr back to the beginning. */

	image->buff = malloc(file_size);
	if (NULL == image->buff) {
		fprintf(stderr, "Failed to allocate memory for source file '%s'.\n", path);
		exit(74);
	}

	bytesRead = fread(image->buff, sizeof(char), file_size, file);
	if (bytesRead < file_size) {
		fprintf(stderr, "Couldn't read source file '%s'.\n", path);
		exit(74);
	}

	fclose(file);
}

void
bmp_save_file(BMP_image* image, const char* path)
{
	FILE* file;
	size_t bytesWritten;

	file = fopen(path, "wb");
	if (NULL == file) {
		fprintf(stderr, "Couldn't open source file '%s'.\n", path);
		exit(74);
	}

	bytesWritten = fwrite(image->buff, sizeof(uint8_t), image->info.header.file_size, file);
	if (bytesWritten < image->info.header.file_size) {
		fprintf(stderr, "ERROR: failed to write to %s.\n"
		"File size: %d\nBytes written: %ld\n",
		path, image->info.header.file_size, bytesWritten);
	}

	fclose(file);
}

static uint32_t
get_int(uint8_t* buff)
{
	uint32_t result;
	
	result = (((uint32_t)buff[3] << 24) | ((uint32_t)buff[2] << 16) |
			  ((uint32_t)buff[1] << 8)  | (uint32_t)buff[0]);

	return result;
}

static uint32_t
get_short(uint8_t* buff)
{
	uint32_t result;

	result = (((uint32_t)buff[1] << 8) | (uint32_t)buff[0]);

	return result;
}


void
bmp_init_image(BMP_image* image)
{
	uint8_t* buff = image->buff;
	BMP_Header_t* header = &image->info.header;
	BMP_Info* info = &image->info;

	header->magic = get_short(&buff[OFFSET_BMP_HEADER_MAGIC]);
	header->file_size = get_int(&buff[OFFSET_BMP_HEADER_FILE_SIZE]);
	header->pixels_offset = get_int(&buff[OFFSET_BMP_HEADER_PIXELS_OFFSET]);

	info->version = get_int(&buff[OFFSET_BMP_INFO_VERSION]);
	info->pic_width = get_int(&buff[OFFSET_BMP_INFO_WIDTH]);
	info->pic_height = get_int(&buff[OFFSET_BMP_INFO_HEIGHT]);
	info->bit_count = (get_short(&buff[OFFSET_BMP_INFO_BITCOUNT]) / 8);
	info->image_size = get_int(&buff[OFFSET_BMP_INFO_IMG_SIZE]);
}

void
bmp_free(BMP_image* image)
{
	free(image->buff);
}

void
bmp_color_to_gray(BMP_image* image)
{
	BMP_Header_t* header = &image->info.header;
	BMP_Info* info		= &image->info;
	uint8_t* buff		= &image->buff[header->pixels_offset];
	uint32_t width		= info->pic_width;
	uint32_t height		= info->pic_height;
	uint32_t channel    = info->bit_count;

	uint8_t* result;

	result = malloc(info->image_size);
	if (NULL == result) {
		fprintf(stderr, "ERROR: failed to allocate temp buffer.\n");
		return;
	}

	uint32_t pix_off;
	uint8_t r;
	uint8_t g;
	uint8_t b;

	for (uint32_t col = 0; col < width; ++col) {

		for (uint32_t row = 0; row < height; ++row) {

			pix_off = (row * width + col) * channel;
			
			r = buff[pix_off    ];
			g = buff[pix_off + 1];
			b = buff[pix_off + 2];

			result[pix_off    ] = r * 0.21 + g * 0.71 + b * 0.07;
			result[pix_off + 1] = r * 0.21 + g * 0.71 + b * 0.07;
			result[pix_off + 2] = r * 0.21 + g * 0.71 + b * 0.07;
		}
	}

	memcpy(buff, result, info->image_size);
	free(result);
}