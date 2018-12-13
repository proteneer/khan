Performance


Batch Size 256

Benchmark [384x256, 256x256, 256x1], both with CELU

torchani vs khan, time per epoch

gdb4: 46 seconds  vs 21 seconds (2.2x slower)
gdb5: 169 seconds vs 66 seconds (2.5x slower)

Batch Size 1024 (gdb5/6 are roughly the same size, gdb7 is much bigger)

gdb5: 95 seconds  vs 25 seconds (3.8x slower)
gdb6: 100 seconds vs 26 seconds (3.8x slower)
gdb7: 580 seconds vs 82 seconds (7.1x slower)

