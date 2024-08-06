```c
int max(int a, int b)
{
    if (a >= b) { return a; } else { return b; }
}

int min(int a, int b)
{
    if (a <= b) { return a; } else { return b; }
}

#define BLUR_SIZE 1

for (int cur_row = 0; cur_row < height; ++cur_row) {
	
	for (int cur_col = 0; cur_col < width; cur_col += 3) {
		
		char r = buff[(cur_row * width) + cur_col + 0];
		char g = buff[(cur_row * width) + cur_col + 1];
		char b = buff[(cur_row * width) + cur_col + 2];
		
		for (int blur_row = max(0, cur_row - BLUR_SIZE); blur_row <= min(height, cur_row + BLUR_SIZE); ++blur_row) {
			
			for (int blur_col = max(0, cur_col - BLUR_SIZE); blur_col <= min(width, cur_col + BLUR_SIZE); ++blur_col) {
				blurAreaSum += inputImg[width * blur_row + blur_col];
				blurAreaSize++;
			}
		}
		
		outputImg[width * cur_row + cur_col] = (blurAreaSum / blurAreaSize);
	}
}
```
										4.1 Архитектура GPU
	GPU состоит из массива потоковых мультипроцессоров (Streaming Multiprocessor, SM).
SM состоит из множества потоковых процессоров именуемых CUDA-ядрами. Также SM имеет блок управления и свою локальную память, которые являются общими для его CUDA-ядер. Например, видеокарта Ampere A100 имеет 108 SM с 64 CUDA-ядрами каждый.
	SM фрагментированы далее на блоки обработки по 16 ядер в каждом, т.е. в одном SM 4 блока
обработки по 16 ядер:

```c
typedef struct StreamingProcessor {
	
} CUDAcore;

typedef struct ProcessingBlock {
	CUDAcore cores[16];
} PB;

typedef struct StreamingMultiprocessor {
	// Warp 1
	PB block1;
	PB block2;

	// Warp 2
	PB block3;
	PB block4;
} SM;

typedef struct NVidia {
	SM sm_array[108];
} AmpereA100;
```
	Память видеокарты представлена двумя видами:
on-chip  - память SM;
off-chip - общие несколько гигабайт видеокарты.



										4.2 Диспетчеризация блоков
	Потоки присваиваются SM-ам поблочно. Количество одновременно запускаемых блоков зависит
от видеокарты. Видеокарта берет на себя заботу по планированию запуска блоков если их
количество превышает доступные ресурсы.
	Поблочное присвоение потоков к SM'ам гарантирует одновременный запуск потоков данного
блока, а также обеспечивает взаимодействие потоков одного блока между собой.



										4.4 Warp'ы и аппаратная поддержка SIMD
	Потоки одного блока могут выполняться с разной скоростью, поэтому, если код имеет фазы
выполнение которых требует синхронности, то такая синхронность должна быть предусмотрена
разрабом посредством __syncthreads().
	В современных GPU когда блок присваивается MS'у, он далее делится на warp'ы - единицы,
состоящие из 32 потоков. Разрабу надо хорошо понимать warp'ы, чтобы делать свой код лучше.
	Warp - это единица, которой орудует SM при диспетчеризации потоков. Например, есть три
блока присвоенные одному SM'у. Далее каждый из них делится на warp'ы для дальнейшей
диспетчеризации. Каждый warp состоит из 32 потоков с индексами threadIdx:
	Warp 1: threadIdx.x = 0 по threadIdx.x = 31
	Warp 2: threadIdx.x = 32 по threadIdx.x = 63
В общем случае warp 'N' начинается с потока 'N * 32' и заканчинается потоком
'((N + 1) * 32) - 1'. Если блок состоит из 256 потоков, тогда 256 / 32 = 8 warp'ов в одном
блоке. Если блоки не кратны 32, тогда последний warp будет добит неактивными потоками.
	Если блок состоит из двумерной структуры потоков, тогда они будут разложены в по
принципу row-major прежде чем разделены на warp'ы:

	Дано
	blockDim.x = 3
	blockDim.y = 3
	Тогда: y0:x0  y0:x1  y0:x2  y1:x0  y1:x1  y1:x2  y2:x0  y2:x1  y2:x2

	  .___1__2__3_> X
	  |	 ________
	0 |	|__|__|__|
	1 |	|__|__|__|
	2 |	|__|__|__|
	Y v

	SM'ы исполняют потоки в Warp'ах по принципу SIMD, т.е. в каждый момент времени все
потоки выполняют одну общую для всех инструкцию но при этом обрабатывают разные данные.



										4.5 Control Divergence
	Если потоки внутри Warp'а идут по разным потокам исполнения, аппаратная поддержка SIMD
выполнит несколько проходов - по одному для каждого пути. На каждом проходе потоки идущие
по пути "А" будут бездействовать до тех пор, пока потоки из пула "Б" не завершат свой. Этот
эффект называется 'Control divergence' - расхождение управления.
	Дивергенция влияет на производительность, однако есть нюансы:
	
	1. если блок состоит из 256 потоков, тогда для обработки вектора A[1000] понадобится 4
	блока. Один такой блок содержит 256 / 32 = 4 Warp'а. В таком случае лишь последний Warp
	подвергается дивергенции.
	
	2. если обрабатываем изображение 62х76, то будет очень много потоков у краев
	изображения, которые подвергнуться дивергенции.



										4.6 Warp Scheduling and Latency Tolerance
	В каждый момент времени SM может запустить только часть из всех потоков. Ранее SM мог
выполнить инструкцию лишь для одного Warp'а, однако современные GPU позволяют делать это
для множества Warp'ов одвновременно.
	Чем больше потоков присвоено данному SM, тем выше его эффективногость: вместо того,
чтобы аппаратура простаивала пока поток получит данные для обработки, она может выбрать
среди других потоков тот, что может быть запущен сразу же.



										4.7 Resource partitioning and occupancy
	Иногда невозможно присвоить SM'у максимальное количество warp'ов которое он может
поддержать. Соотношение количества Warp'ов присвоенных SM'у к максимально допустимому числу
которое он может принять называется 'occupancy' - занятость.
	Вычислительные ресурсы SM влючают в себя: регистры, общая память, слоты с блоками и
слоты с потоками, которые динамически разделяются между потоками. Например, Ampere A100
может поддерживать до 32 блоков на SM, до 64 Warp'а (2048 потоков) на SM и до 1024 потока
на блок. Если сетка запущена с блоком в 1024 потока, тогда 2048 потоковых слотов в каждом
SM'е будут разделены и присвоены двум блокам. В таком случае каждый SM может вместить до
двух блоков.

	threadNum = 2048
	BlockDim        1024    512    256    128    64
	PB required        2      4      8     16    32

	Динамическое разделение потоков делает SM'ы гибкими - они могут запускать много блоков с
малым количеством потоков, либо наоборот - мало блоков с большим количеством потоков. Однако
тут можно столкнуться с неожиданным ограничением. На примере Ampere A100 видим, что размер
блока варьирует от 1024 до 64, в итоге на SM приходится от 2 до 32 блоков и во всех этих
случаях общее количество потоков, присваиваемых SM равно 2048, что является максимально
эффективным использованием ресурсов.
	Однако, когда блок состоит из 32 потоков, 2048 потоковых слотов надо разделить и
присвоить 64 блокам, а, например, SM у Volta GPU поддерживает лишь 32 блока в моменте.
В итоге лишь 1024 потоковых слота будут использованы, т.е. 32 блока с 32 потоками в каждом.
Здесь коэффициет 'occupancy' будет равен 50% (1024 присвоенных потоков / 2048 максимум потоков). В данном конкретном примере блок должен содержать минимум 64 потока, чтобы
использовать ресурсы по максимуму.
	Такой же негативный эффект может возникнуть если максимальное количество потоков на
блок не делится на размер блока. На примере Ampere A100 видим, что SM может поддерживать до
2048 потоков, одна если блок состоит из 768 потоков, тогда SM будет состоять из двух
потоковых блоков (1536 потоков), а 512 останутся неиспользованными.
	Локальные переменные хранятся в регистрах, соответственно, чем их больше, тем меньше SM
может включить в себя потоков. Ampere A100 выделяет до 65536 регистров на SM и запуск
на максималках, тогда каждый из 2048 потоков не должен захватывать более 32 регистров.



										4.8 Querying Device Properties
```c
struct cudaDeviceProp {
	char name[256];
	cudaUUID_t uuid;
	size_t totalGlobalMem;
	size_t sharedMemPerBlock;
	int regsPerBlock;
	int warpSize;
	size_t memPitch;
	int maxThreadsPerBlock;
	int maxThreadsDim[3];
	int maxGridSize[3];
	int clockRate;
	size_t totalConstMem;
	int major;
	int minor;
	size_t textureAlignment;
	size_t texturePitchAlignment;
	int deviceOverlap;
	int multiProcessorCount;
	int kernelExecTimeoutEnabled;
	int integrated;
	int canMapHostMemory;
	int computeMode;
	int maxTexture1D;
	int maxTexture1DMipmap;
	int maxTexture1DLinear;
	int maxTexture2D[2];
	int maxTexture2DMipmap[2];
	int maxTexture2DLinear[3];
	int maxTexture2DGather[2];
	int maxTexture3D[3];
	int maxTexture3DAlt[3];
	int maxTextureCubemap;
	int maxTexture1DLayered[2];
	int maxTexture2DLayered[3];
	int maxTextureCubemapLayered[2];
	int maxSurface1D;
	int maxSurface2D[2];
	int maxSurface3D[3];
	int maxSurface1DLayered[2];
	int maxSurface2DLayered[3];
	int maxSurfaceCubemap;
	int maxSurfaceCubemapLayered[2];
	size_t surfaceAlignment;
	int concurrentKernels;
	int ECCEnabled;
	int pciBusID;
	int pciDeviceID;
	int pciDomainID;
	int tccDriver;
	int asyncEngineCount;
	int unifiedAddressing;
	int memoryClockRate;
	int memoryBusWidth;
	int l2CacheSize;
	int persistingL2CacheMaxSize;
	int maxThreadsPerMultiProcessor;
	int streamPrioritiesSupported;
	int globalL1CacheSupported;
	int localL1CacheSupported;
	size_t sharedMemPerMultiprocessor;
	int regsPerMultiprocessor;
	int managedMemory;
	int isMultiGpuBoard;
	int multiGpuBoardGroupID;
	int singleToDoublePrecisionPerfRatio;
	int pageableMemoryAccess;
	int concurrentManagedAccess;
	int computePreemptionSupported;
	int canUseHostPointerForRegisteredMem;
	int cooperativeLaunch;
	int cooperativeMultiDeviceLaunch;
	int pageableMemoryAccessUsesHostPageTables;
	int directManagedMemAccessFromHost;
	int accessPolicyMaxWindowSize;
}

__host__​__device__​cudaError_t
cudaGetDeviceCount(int* count);

__host__​cudaError_t
cudaGetDeviceProperties(cudaDeviceProp* prop, int device );
```
cudaGetDeviceCount() возвращает общее количество CUDA-совместимых устройств.
cudaGetDeviceProperties() возвращает параметры устройства.

Результат вызова:
```shell
                1. Common
version:                                       8.6 Ampere
Device name:                                   NVIDIA GeForce RTX 3060 Laptop GPU
CUDA cores:                                    3840

                2. Memory
32-bit registers available per block:          65536 
Shared memory available per block:             48.0 Kb
32-bit registers available per multiprocessor: 65536 
Shared memory available per multiprocessor:    100.0 Kb
Size of L2 cache                               3.0 Mb
Constant memory available on device:           64.0 Kb
Global memory available on device:             6.0 Gb
Global memory bus width                        192 bits

                3. Compute Capability
Warp size:                                    32 threads
Maximum number of threads per block:          1024
Maximum resident threads per multiprocessor:  1536
Number of multiprocessors on device:          30
Maximum size of each dimension of a block:
                                         X =  1024
                                         Y =  1024
                                         Z =  64
Maximum size of each dimension of a grid:
                                         X =  2147483647
                                         Y =  65535
                                         Z =  65535
```