#ifndef VOLVOXAI_CPU_FEATURES_H
#define VOLVOXAI_CPU_FEATURES_H

#include <stdint.h>

/* Capability queries are safe on every target.  A false result means the
 * dispatcher must retain the portable scalar implementation. */
int vx_cpu_has_avx2(void);
int vx_cpu_has_avx_vnni(void);
int vx_cpu_has_avx512f(void);
int vx_cpu_has_avx512_vnni(void);
int vx_cpu_has_neon(void);
int vx_cpu_has_arm_dotprod(void);
int vx_cpu_has_arm_i8mm(void);
int vx_cpu_has_arm_sve(void);
int vx_cpu_has_arm_sve2(void);
int vx_cpu_has_arm_sve_i8mm(void);

/* SVE vector length is thread state, not a process-wide CPU constant.  This
 * query therefore intentionally is not cached.  It returns the calling
 * thread's vector length in bytes, or zero when SVE is unavailable or the OS
 * cannot report a valid length. */
uint32_t vx_cpu_arm_sve_vl_bytes(void);

#endif
