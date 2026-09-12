#ifndef VOLVOXAI_WASM_FREESTANDING_STRING_H
#define VOLVOXAI_WASM_FREESTANDING_STRING_H

#include <stddef.h>

#ifndef VOLVOXAI_WASM_FREESTANDING_SCOPE
#define VOLVOXAI_WASM_FREESTANDING_SCOPE extern
#endif

VOLVOXAI_WASM_FREESTANDING_SCOPE void* memcpy(
    void* destination, const void* source, size_t count);
VOLVOXAI_WASM_FREESTANDING_SCOPE void* memset(
    void* destination, int value, size_t count);
VOLVOXAI_WASM_FREESTANDING_SCOPE int strcmp(
    const char* left, const char* right);
VOLVOXAI_WASM_FREESTANDING_SCOPE int strncmp(
    const char* left, const char* right, size_t count);
VOLVOXAI_WASM_FREESTANDING_SCOPE size_t strlen(const char* value);

#if defined(VOLVOXAI_WASM_FREESTANDING_EXTENDED_LIBC)
VOLVOXAI_WASM_FREESTANDING_SCOPE int memcmp(
    const void* left, const void* right, size_t count);
VOLVOXAI_WASM_FREESTANDING_SCOPE void* memmove(
    void* destination, const void* source, size_t count);
VOLVOXAI_WASM_FREESTANDING_SCOPE char* strcpy(
    char* destination, const char* source);
VOLVOXAI_WASM_FREESTANDING_SCOPE char* strncpy(
    char* destination, const char* source, size_t count);
VOLVOXAI_WASM_FREESTANDING_SCOPE char* strchr(const char* text, int character);
VOLVOXAI_WASM_FREESTANDING_SCOPE char* strrchr(const char* text, int character);
VOLVOXAI_WASM_FREESTANDING_SCOPE char* strstr(const char* haystack, const char* needle);
VOLVOXAI_WASM_FREESTANDING_SCOPE void* memchr(const void* s, int c, size_t n);
#endif

#endif
