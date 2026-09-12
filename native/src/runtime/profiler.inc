// ---- per-op timing aggregation (--debug) ----------------------------------
void prof_reset(void) { g_nprof = 0; }

static void prof_add(const char* op, double ms) {
    for (int index = 0; index < g_nprof; index++) {
        if (!strcmp(g_prof[index].op, op)) {
            g_prof[index].ms += ms;
            g_prof[index].count++;
            return;
        }
    }
    if (g_nprof < (int)(sizeof(g_prof) / sizeof(g_prof[0]))) {
        strncpy(g_prof[g_nprof].op, op, sizeof(g_prof[g_nprof].op) - 1);
        g_prof[g_nprof].op[sizeof(g_prof[g_nprof].op) - 1] = 0;
        g_prof[g_nprof].ms = ms;
        g_prof[g_nprof].count = 1;
        g_nprof++;
    }
}

void prof_add_entry(const char* op, double ms) { prof_add(op, ms); }

void prof_report(void) {
    double total = 0.0;
    for (int index = 0; index < g_nprof; index++) {
        int maximum = index;
        for (int candidate = index + 1; candidate < g_nprof; candidate++) {
            if (g_prof[candidate].ms > g_prof[maximum].ms) maximum = candidate;
        }
        if (maximum != index) {
            ProfEntry entry = g_prof[index];
            g_prof[index] = g_prof[maximum];
            g_prof[maximum] = entry;
        }
        total += g_prof[index].ms;
    }
    vx_engine_log("[debug] --- op time summary (by total) ---\n");
    for (int index = 0; index < g_nprof; index++) {
        vx_engine_log("[debug]   %-16s %8.2f ms  (n=%d)\n",
                g_prof[index].op, g_prof[index].ms, g_prof[index].count);
    }
    vx_engine_log("[debug]   %-16s %8.2f ms\n", "TOTAL", total);
}
