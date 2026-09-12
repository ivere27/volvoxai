/* The C recipe and browser entry choose the same generated shader closure. */
#ifndef VOLVOXAI_SHADER_CATALOG_PROFILE_H
#define VOLVOXAI_SHADER_CATALOG_PROFILE_H
#include "backend_config.h"
#if VOLVOXAI_ENABLE_TRAINING
#include "shader_catalog.h"
#else
#include "shader_catalog_inference.h"
#endif
#endif
