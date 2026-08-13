#ifndef VOLVOXAI_CPU_FEATURES_H
#define VOLVOXAI_CPU_FEATURES_H

/* Capability queries are safe on every target.  A false result means the
 * dispatcher must retain the portable scalar implementation. */
int vx_cpu_has_avx2(void);
int vx_cpu_has_avx_vnni(void);
int vx_cpu_has_avx512f(void);
int vx_cpu_has_avx512_vnni(void);
int vx_cpu_has_neon(void);
int vx_cpu_has_arm_dotprod(void);

#endif
