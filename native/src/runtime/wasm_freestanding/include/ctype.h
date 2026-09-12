#ifndef VOLVOXAI_WASM_FREESTANDING_CTYPE_H
#define VOLVOXAI_WASM_FREESTANDING_CTYPE_H

/* The character classes cJSON asks about, and nothing else.
 *
 * These are the C locale by definition here: a freestanding build has no
 * locale to consult, and JSON is defined in terms of ASCII regardless. Taking
 * an int and range-checking it is what keeps a negative char from indexing
 * off the front of a table, which is the classic way these go wrong. */

static inline int isdigit(int character) {
    return character >= '0' && character <= '9';
}

static inline int isxdigit(int character) {
    return isdigit(character) ||
           (character >= 'a' && character <= 'f') ||
           (character >= 'A' && character <= 'F');
}

static inline int isspace(int character) {
    return character == ' ' || character == '\t' || character == '\n' ||
           character == '\v' || character == '\f' || character == '\r';
}

static inline int isalpha(int character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= 'A' && character <= 'Z');
}

static inline int isalnum(int character) {
    return isalpha(character) || isdigit(character);
}

static inline int isupper(int character) {
    return character >= 'A' && character <= 'Z';
}

static inline int islower(int character) {
    return character >= 'a' && character <= 'z';
}

static inline int toupper(int character) {
    return islower(character) ? character - ('a' - 'A') : character;
}

static inline int tolower(int character) {
    return isupper(character) ? character + ('a' - 'A') : character;
}

#endif
