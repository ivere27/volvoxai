#ifndef VOLVOXAI_WASM_FREESTANDING_STDIO_H
#define VOLVOXAI_WASM_FREESTANDING_STDIO_H

#include <stdarg.h>
#include <stddef.h>

#ifndef VOLVOXAI_WASM_FREESTANDING_SCOPE
#define VOLVOXAI_WASM_FREESTANDING_SCOPE extern
#endif

VOLVOXAI_WASM_FREESTANDING_SCOPE int snprintf(
    char* destination, size_t capacity, const char* format, ...);
VOLVOXAI_WASM_FREESTANDING_SCOPE int vsnprintf(
    char* destination,
    size_t capacity,
    const char* format,
    va_list arguments);

#if defined(VOLVOXAI_WASM_FREESTANDING_EXTENDED_LIBC)
/* cJSON calls these two, and nothing else here does. sprintf forwards to the
 * bounded form; sscanf understands one conversion. See wasm_libc.c. */
VOLVOXAI_WASM_FREESTANDING_SCOPE int sprintf(
    char* destination, const char* format, ...);
VOLVOXAI_WASM_FREESTANDING_SCOPE int sscanf(
    const char* text, const char* format, ...);
#endif

typedef struct FILE FILE;
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#define EOF (-1)

#define stderr ((FILE*)0)
VOLVOXAI_WASM_FREESTANDING_SCOPE int fprintf(FILE* stream, const char* format, ...);
VOLVOXAI_WASM_FREESTANDING_SCOPE int printf(const char* format, ...);
VOLVOXAI_WASM_FREESTANDING_SCOPE int rename(const char* oldpath, const char* newpath);
VOLVOXAI_WASM_FREESTANDING_SCOPE FILE* fopen(const char* path, const char* mode);
VOLVOXAI_WASM_FREESTANDING_SCOPE int fclose(FILE* stream);
VOLVOXAI_WASM_FREESTANDING_SCOPE size_t fread(void* ptr, size_t size, size_t count, FILE* stream);
VOLVOXAI_WASM_FREESTANDING_SCOPE size_t fwrite(const void* ptr, size_t size, size_t count, FILE* stream);
VOLVOXAI_WASM_FREESTANDING_SCOPE int fseek(FILE* stream, long offset, int origin);
VOLVOXAI_WASM_FREESTANDING_SCOPE long ftell(FILE* stream);
VOLVOXAI_WASM_FREESTANDING_SCOPE int ferror(FILE* stream);
VOLVOXAI_WASM_FREESTANDING_SCOPE int fflush(FILE* stream);
VOLVOXAI_WASM_FREESTANDING_SCOPE char* fgets(char* str, int num, FILE* stream);
VOLVOXAI_WASM_FREESTANDING_SCOPE int remove(const char* path);
VOLVOXAI_WASM_FREESTANDING_SCOPE int fputc(int character, FILE* stream);

#endif
