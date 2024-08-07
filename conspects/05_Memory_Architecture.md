										5.1 Importance of Memory Access Efficiency
В примере по умножению матриц можно отметить самый активный вычислительный участок:

```c
for (uint32_t k = 0; k < width; ++k) {
	value += first[row * width + k] * second[k * width + col];
}
```
На каждой итерации осуществляется два обращения к глобальной памяти - один для извлечения
операндов для операции умножения, второй для сложения. Таким образом мы имеем 2 операции с
плавающей точкой (FLOP) на 8 байт информации или "2 FLOP / 8 B = 0.25 FLOP/B".
	Вышеуказанное уравнение описывает "соотношение вычислений к обращению к глобальной
памяти" в виде количества FLOP на каждый байт информации, извлеченного из глобальной
памяти. Часто это соотношение также называют "арифметическая интенсивность" или
"вычислительная интенсивность".
	Максимальная ширина пропускания глобальной памяти видеокарты Ampere A100 равна
1555 Gb/sec. 1555 Gb/sec * 0.25 FLOP/B = 389 GFLOPS, что является лишь 2% от пика
вычислительной мощности этой модели (19500 GFLOPS). Таким образом становится очевидно как
сильно производительность зависит от памяти.
	"Memory-bound program" - программа, скорость исполнения которой ограничена полосой
пропускания памяти.
	Чем выше коэффициент "вычисление/обращение к глобальной памяти", тем выше эффективность.
Для достижения этой цели необходимо сократить количество обращений к глобальной памяти.
Напримре, чтобы выжать из Ampere A100 все 19500 GFLOPS, соотношение должно быть:

	19500 GOP/sec : 1555 GB/sec = 12.5 OP/B,

т.е. на каждое обращение за 4-байтным значением типа float должно быть 50 операций с
плавающей точкой.



										5.2 CUDA Memory Types
Global Memory	- RW хостом и устройством.
Constant Memory	- RW хостом, RO устройством, однако низкая задержка.
Local Memory	- является составной частью Global Memory, т.е. высокая задержка при
				  обращении, однако является приватной зоной каждого потока где он хранит
				  данные доступные лишь ему, но по каким-то причинам непригодные для
				  хранения в регистрах, например статический массив, значения регистров и
				  прочие элементы из стека потока исполнения.
Registers		- является частью SM, доступны лишь внутри потока.
Shared Memory	- является частью SM, общая для thread block'а, т.е. доступна всем потокам
				  одного блока.
продолжить со страницы 99 (129)