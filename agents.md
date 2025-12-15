1. Optimized code has to generate bit-precise outputs
2. Loops need to be unrolled where possible for effectiveness of instruction pipeline
3. Use parallel.for as much as needed, considering that threads have no coupling on the frame level.
4. Do not createnew dimension variables in the row/pixel-level function to avoid GC stress and checking. Has to be stackalloc, pointer, reused buffer, whatever.
5. Do not make changes that can affect HFR calculations.
6. Comment ALL of the code, to explain what, why and how.
