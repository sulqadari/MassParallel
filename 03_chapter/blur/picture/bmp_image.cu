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

CUDA_HOST void
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

CUDA_HOST uint8_t
bmp_load_file(BMP_image* image, const char* path)
{
	FILE* file;
	size_t bytesRead;
	uint32_t file_size;
	uint8_t res = 0;

	file = fopen(path, "rb");
	if (NULL == file) {
		fprintf(stderr, "Couldn't open source file '%s'.\n", path);
		return (1);
	}

	fseek(file, 0L, SEEK_END);	/* Move file prt to EOF. */
	file_size = ftell(file);		/* How far we are from start of the file? */
	rewind(file);				/* Rewind file ptr back to the beginning. */

	image->buff = (uint8_t*)malloc(file_size);
	if (NULL == image->buff) {
		fprintf(stderr, "Failed to allocate memory for source file '%s'.\n", path);
		res = 1;
		goto _close_file;
	}

	bytesRead = fread(image->buff, sizeof(char), file_size, file);
	if (bytesRead < file_size) {
		fprintf(stderr, "Couldn't read source file '%s'.\n", path);
		res = 1;
		goto _deallocate;
	} else {
		goto _close_file;
	}

_deallocate:
	free(image->buff);

_close_file:
	fclose(file);
	return (res);
}

CUDA_HOST void
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

CUDA_HOST static uint32_t
get_int(uint8_t* buff)
{
	return (((uint32_t)buff[3] << 24) | ((uint32_t)buff[2] << 16) |
			((uint32_t)buff[1] << 8)  | (uint32_t)buff[0]);
}

CUDA_HOST static uint32_t
get_short(uint8_t* buff)
{
	return (((uint32_t)buff[1] << 8) | (uint32_t)buff[0]);
}


CUDA_HOST void
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

CUDA_HOST void
bmp_free(BMP_image* image)
{
	free(image->buff);
}

CUDA_HD int32_t _max(int a, int b) {
    if (a >= b) { return a; } else { return b; }
}
CUDA_HD int32_t _min(int a, int b) {
    if (a <= b) { return a; } else { return b; }
}

CUDA_GLOBAL void
bmp_blurring(uint8_t* outputImg, uint8_t* inputImg,
			int32_t width, int32_t height, int32_t N)
{
	int32_t row = blockDim.y * blockIdx.y + threadIdx.y;
    int32_t col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col >= width || row >= height) { return; }

    int32_t blurAreaSum = 0;
    int32_t blurAreaSize = 0;
	
    for (int32_t row2 = _max(0, row-N); row2 <= _min(height, row+N); row2++) {
        for (int32_t col2 = _max(0, col-N); col2 <= _min(width, col+N); col2++) {
            blurAreaSum += inputImg[width * row2 + col2];
            blurAreaSize++;
        }
    }

    outputImg[width * row + col] = (uint8_t)(blurAreaSum / blurAreaSize);
}