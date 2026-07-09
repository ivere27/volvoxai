#ifndef VOLVOXAI_KIE_RUNTIME_H
#define VOLVOXAI_KIE_RUNTIME_H

int kie_config_is_tiny_receipt(const char* config_path);
int kie_chat(const char* config_path, const char* weights_path, const char* image_path,
             const char* prompt, const char* family, int max_new, int debug);

#endif
