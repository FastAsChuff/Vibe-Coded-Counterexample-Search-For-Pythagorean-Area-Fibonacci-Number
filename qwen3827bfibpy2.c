// Motivated by https://math.stackexchange.com/questions/5153990/is-there-a-fibonacci-number-that-is-the-area-of-a-right-triangle-with-integer-si?noredirect=1#comment11133122_5153990
// Test with F(180)= 2^2 * 3 * 7 * 13 * 101 * 1013 * 5923 * 9901 * 666667 * 769231 = 80000104000024×84000027999993/2
// m^2 + n^2 = c, m > n > 0
// (m^2 + n^2)^2 = c^2 = (m^2 - n^2)^2 + (2mn)^2
// Test F(12s) = mn(m^2 - n^2)
// For m = 9876543219876543210, n = 987654321123456789
// e.g. F(12s) = (2 * 3^2 * 5 * 7 * 11 * 13 * 17^2 * 19 * 52579 * 379721)(3^2 * 11 * 37 * 79 * 139 * 73589 * 333667)(3^5 11^2 131 149 787 91957 686209 2234483 1516362139)
// = 2 * 3^7 * 5 * 7 * 11^4 * 13 * 17^2 * 19 * 37 * 79 * 131 * 139 * 149 * 787 * 52579 * 73589 * 91957 * 333667 * 379721 * 686209 * 2234483 * 1516362139
// Pythagorean-area search for Fibonacci numbers F(12m), m >= 1
// Optimised: precomputed powers, in-place tree walk, incremental modular
// filter (no GMP division at the leaf), mpz_sqrtrem, OpenMP parallelism.
//
// Compile:  gcc -O3 -mssse3 -Wall -Wextra -o pyarea pyarea.c -lgmp -fopenmp
// Run:      ./pyarea            (runs self-test + file search)
//           ./pyarea --notest   (skips self-test)

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <gmp.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ======================== constants ======================== */
#define MAX_P  50
#define MAX_E  16
#define NF     10

/* ======================== globals ======================== */
static mpz_t g_primes[MAX_P];
static int   g_exps[MAX_P];
static int   g_np;
static mpz_t g_N;

static mpz_t ppow[MAX_P][MAX_E];

static volatile int g_found;
static mpz_t g_sol_d, g_sol_m, g_sol_n, g_sol_q;
static long  g_leaf_count;

static int   FPR[NF];
static int   n_active;
static int   inv_tab[NF][256];
static int   sqr_tab[NF][256];
static int   N_mod[NF];
static int   mpow_t[MAX_P][MAX_E][NF];

/* ======================== helpers ======================== */
static long mpow(long a, long e, long m)
{
    long r = 1 % m;
    a %= m;
    while (e > 0) {
        if (e & 1) r = (r * a) % m;
        a = (a * a) % m;
        e >>= 1;
    }
    return r;
}

/* Compute (x mod p) for a GMP integer x and small prime p */
static int gmp_mod_small(mpz_t x, unsigned long p)
{
    mpz_t tmp;
    mpz_init(tmp);
    mpz_fdiv_r_ui(tmp, x, p);
    int r = (int)mpz_get_ui(tmp);
    mpz_clear(tmp);
    return r;
}

/* ======================== filter tables ======================== */
static void build_filter(void)
{
    static const int cands[NF] =
        {97, 101, 103, 107, 109, 113, 127, 131, 137, 139};

    n_active = 0;
    for (int i = 0; i < NF; i++) {
        int p = cands[i];
        if (mpz_divisible_ui_p(g_N, (unsigned long)p)) continue;

        int fi = n_active;
        FPR[fi]  = p;
        N_mod[fi]= gmp_mod_small(g_N, (unsigned long)p);

        for (int x = 0; x < p; x++) {
            inv_tab[fi][x] = (x == 0) ? -1 : (int)mpow(x, p - 2, p);
            sqr_tab[fi][x] = -1;
        }
        for (int x = 1; x < p; x++) {
            int r = (x * x) % p;
            if (sqr_tab[fi][r] < 0) sqr_tab[fi][r] = x;
        }
        n_active++;
    }

    for (int pi = 0; pi < g_np; pi++) {
        for (int fi = 0; fi < n_active; fi++) {
            int p = FPR[fi];
            int base = gmp_mod_small(g_primes[pi], (unsigned long)p);
            mpow_t[pi][0][fi] = 1;
            for (int k = 1; k <= g_exps[pi]; k++)
                mpow_t[pi][k][fi] = (mpow_t[pi][k-1][fi] * base) % p;
        }
    }
}

/* ======================== fast leaf filter ======================== */
typedef struct { int d[NF]; int q[NF]; } MState;

static int modfilter(const MState *ms)
{
    for (int i = 0; i < n_active; i++) {
        int p = FPR[i];
        int dm = ms->d[i] % p;
        int qm = ms->q[i] % p;
        int d2 = (dm * dm) % p;
        if (d2 == 0 || qm == 0) continue;

        int inv_d2 = inv_tab[i][d2];
        int inv_qm = inv_tab[i][qm];
        int Mm  = (N_mod[i] * inv_d2) % p;
        int mnm = (Mm * inv_qm) % p;

        int S2  = (4 * mnm % p * mnm + qm * qm) % p;
        int s1  = sqr_tab[i][S2];
        if (s1 < 0) return 0;

        int ok = 0;
        int roots[2] = { s1, p - s1 };
        for (int k = 0; k < 2 && !ok; k++) {
            int diff = (roots[k] - 2 * mnm + p) % p;
            if (sqr_tab[i][diff] >= 0) ok = 1;
        }
        if (!ok) return 0;
    }
    return 1;
}

/* ======================== GMP leaf check ======================== */
static int try_candidate(mpz_t mn, mpz_t q)
{
    mpz_t S2, q2, S, Z, A, B, rem;
    mpz_init(S2); mpz_init(q2); mpz_init(S);
    mpz_init(Z);  mpz_init(A);  mpz_init(B);
    mpz_init(rem);

    /* S2 = 4 mn^2 + q^2 */
    mpz_mul(S2, mn, mn);
    mpz_mul_ui(S2, S2, 4);
    mpz_mul(q2, q, q);
    mpz_add(S2, S2, q2);

    int ok = 0;

    /* Check if S2 is a perfect square: sqrtrem(root, rem, n) sets rem = n 
- root^2 */
    mpz_sqrtrem(S, rem, S2);
    if (mpz_sgn(rem) == 0) {
        /* Z = S - 2 mn */
        mpz_sub(Z, S, mn);
        mpz_sub(Z, Z, mn);
        if (mpz_sgn(Z) > 0) {
            mpz_sqrtrem(A, rem, Z);
            if (mpz_sgn(rem) == 0) {
                /* A = m - n, must be odd positive */
                if (mpz_odd_p(A) && mpz_cmp_ui(A, 0) > 0) {
                    if (mpz_divisible_p(q, A)) {
                        mpz_div(B, q, A);   /* B = m + n */
                        if (mpz_odd_p(B) && mpz_cmp(B, A) > 0) {
                            mpz_t m, nn, ver;
                            mpz_init(m); mpz_init(nn); mpz_init(ver);
                            mpz_add(m, A, B);  mpz_div_ui(m, m, 2);
                            mpz_sub(nn, B, A); mpz_div_ui(nn, nn, 2);
                            if (mpz_cmp_ui(nn, 0) > 0) {
                                mpz_mul(ver, m, nn);
                                //mpz_mul(ver, ver, q); // BUG
                                if (mpz_cmp(ver, mn) == 0) {
                                    ok = 1;
                                    mpz_set(g_sol_m, m);
                                    mpz_set(g_sol_n, nn);
                                }
                                mpz_clear(ver);
                            }
                            mpz_clear(m); mpz_clear(nn);
                        }
                    }
                }
            }
        }
    }

    mpz_clear(S2); mpz_clear(q2); mpz_clear(S);
    mpz_clear(Z);  mpz_clear(A);  mpz_clear(B);
    mpz_clear(rem);
    return ok;
}

/* ======================== recursive search ======================== */
static void search(int idx, mpz_t d, mpz_t q, MState ms)
{
    if (g_found) return;

    if (idx == g_np) {
        g_leaf_count++;

        if (mpz_even_p(q)) return;

        if (!modfilter(&ms)) return;

        mpz_t d2, M, mn;
        mpz_init(d2); mpz_init(M); mpz_init(mn);
        mpz_mul(d2, d, d);
        mpz_div(M, g_N, d2);
        mpz_div(mn, M, q);

        if (try_candidate(mn, q)) {
            mpz_set(g_sol_d, d);
            mpz_set(g_sol_q, q);
            g_found = 1;
        }

        mpz_clear(d2); mpz_clear(M); mpz_clear(mn);
        return;
    }

    int e     = g_exps[idx];
    int max_t = e / 2;
    int max_qe = (mpz_cmp_ui(g_primes[idx], 2) == 0) ? 0 : e;

    for (int t = 0; t <= max_t && !g_found; t++) {
        MState ms2 = ms;

        if (t > 0) {
            mpz_mul(d, d, ppow[idx][t]);
            for (int fi = 0; fi < n_active; fi++)
                ms2.d[fi] = (ms2.d[fi] * mpow_t[idx][t][fi]) % FPR[fi];
        }

        int upper = (e - 2 * t) < max_qe ? (e - 2 * t) : max_qe;
        for (int qe = 0; qe <= upper && !g_found; qe++) {
            MState ms3 = ms2;

            if (qe > 0) {
                mpz_mul(q, q, ppow[idx][qe]);
                for (int fi = 0; fi < n_active; fi++)
                    ms3.q[fi] = (ms3.q[fi] * mpow_t[idx][qe][fi]) % 
FPR[fi];
            }

            search(idx + 1, d, q, ms3);

            if (qe > 0)
                mpz_div(q, q, ppow[idx][qe]);
        }

        if (t > 0)
            mpz_div(d, d, ppow[idx][t]);
    }
}

/* ======================== parallel driver ======================== */
static void run_search(void)
{
    g_found = 0;
    g_leaf_count = 0;

    if (g_np == 0) return;

    build_filter();

    for (int i = 0; i < g_np; i++) {
        mpz_set_ui(ppow[i][0], 1);
        for (int k = 1; k <= g_exps[i]; k++)
            mpz_mul(ppow[i][k], ppow[i][k-1], g_primes[i]);
    }

    MState ms0;
    for (int i = 0; i < NF; i++) { ms0.d[i] = 1; ms0.q[i] = 1; }

#ifdef _OPENMP
    if (g_np >= 2) {
        int e0 = g_exps[0], e1 = g_exps[1];
        int mt0 = e0 / 2, mt1 = e1 / 2;
        int mq0 = (mpz_cmp_ui(g_primes[0], 2) == 0) ? 0 : e0;
        int mq1 = (mpz_cmp_ui(g_primes[1], 2) == 0) ? 0 : e1;

        #pragma omp parallel for schedule(dynamic, 1)
        for (int t0 = 0; t0 <= mt0; t0++) {
            if (g_found) continue;
            int u0 = (e0 - 2*t0) < mq0 ? (e0 - 2*t0) : mq0;
            for (int qe0 = 0; qe0 <= u0 && !g_found; qe0++) {
                if (g_found) continue;
                mpz_t td, tq;
                mpz_init(td); mpz_init(tq);
                mpz_set_ui(td, 1); mpz_set_ui(tq, 1);
                if (t0)  mpz_mul(td, td, ppow[0][t0]);
                if (qe0) mpz_mul(tq, tq, ppow[0][qe0]);

                MState ms1 = ms0;
                for (int fi = 0; fi < n_active; fi++) {
                    if (t0)  ms1.d[fi] = (ms1.d[fi] * mpow_t[0][t0][fi])  
% FPR[fi];
                    if (qe0) ms1.q[fi] = (ms1.q[fi] * mpow_t[0][qe0][fi]) 
% FPR[fi];
                }

                for (int t1 = 0; t1 <= mt1 && !g_found; t1++) {
                    if (g_found) break;
                    int u1 = (e1 - 2*t1) < mq1 ? (e1 - 2*t1) : mq1;
                    MState ms2 = ms1;
                    if (t1) {
                        mpz_mul(td, td, ppow[1][t1]);
                        for (int fi = 0; fi < n_active; fi++)
                            ms2.d[fi] = (ms2.d[fi] * mpow_t[1][t1][fi]) % 
FPR[fi];
                    }
                    for (int qe1 = 0; qe1 <= u1 && !g_found; qe1++) {
                        if (g_found) break;
                        MState ms3 = ms2;
                        if (qe1) {
                            mpz_mul(tq, tq, ppow[1][qe1]);
                            for (int fi = 0; fi < n_active; fi++)
                                ms3.q[fi] = (ms3.q[fi] * 
mpow_t[1][qe1][fi]) % FPR[fi];
                        }
                        search(2, td, tq, ms3);
                        if (qe1) mpz_div(tq, tq, ppow[1][qe1]);
                    }
                    if (t1) mpz_div(td, td, ppow[1][t1]);
                }

                mpz_clear(td); mpz_clear(tq);
            }
        }
    } else
#endif
    {
        mpz_t d, q;
        mpz_init(d); mpz_init(q);
        mpz_set_ui(d, 1); mpz_set_ui(q, 1);
        search(0, d, q, ms0);
        mpz_clear(d); mpz_clear(q);
    }
}

/* ======================== self-test ======================== */
static void self_test(void)
{
    printf("=== SELF-TEST: area = 840 = 2^3 * 3 * 5 * 7 ===\n");

    g_np = 4;
    mpz_set_ui(g_primes[0], 2); g_exps[0] = 3;
    mpz_set_ui(g_primes[1], 3); g_exps[1] = 1;
    mpz_set_ui(g_primes[2], 5); g_exps[2] = 1;
    mpz_set_ui(g_primes[3], 7); g_exps[3] = 1;
    mpz_set_ui(g_N, 840);

    run_search();

    if (g_found) {
        printf("  PASS  d="); gmp_printf("%Zd", g_sol_d);
        printf("  m="); gmp_printf("%Zd", g_sol_m);
        printf("  n="); gmp_printf("%Zd", g_sol_n);
        printf("  q="); gmp_printf("%Zd", g_sol_q);
        printf("\n");

        mpz_t mm, nn, av, bv, cv, ar;
        mpz_init(mm); mpz_init(nn); mpz_init(av); mpz_init(bv); 
mpz_init(cv); mpz_init(ar);
        mpz_mul(mm, g_sol_m, g_sol_m);
        mpz_mul(nn, g_sol_n, g_sol_n);
        mpz_sub(av, mm, nn);  mpz_mul(av, av, g_sol_d);
        mpz_mul(bv, g_sol_m, g_sol_n); mpz_mul_ui(bv, bv, 2); mpz_mul(bv, 
bv, g_sol_d);
        mpz_add(cv, mm, nn);  mpz_mul(cv, cv, g_sol_d);
        mpz_mul(ar, av, bv);  mpz_div_ui(ar, ar, 2);
        printf("  legs "); gmp_printf("%Zd", av); printf(", "); 
gmp_printf("%Zd", bv);
        printf("  hyp "); gmp_printf("%Zd", cv);
        printf("  area "); gmp_printf("%Zd", ar);
        printf(" (expect 840)\n");
        mpz_clear(mm); mpz_clear(nn); mpz_clear(av); mpz_clear(bv); 
mpz_clear(cv); mpz_clear(ar);
    } else {
        printf("  *** FAIL: 840 not found! leaves=%ld ***\n", 
g_leaf_count);
    }
    printf("=== END SELF-TEST ===\n\n");
}

/* ======================== main ======================== */
int main(int argc, char *argv[])
{
    for (int i = 0; i < MAX_P; i++) {
        mpz_init(g_primes[i]);
        for (int k = 0; k < MAX_E; k++) mpz_init(ppow[i][k]);
    }
    mpz_init(g_N);
    mpz_init(g_sol_d); mpz_init(g_sol_m);
    mpz_init(g_sol_n); mpz_init(g_sol_q);

    if (!(argc > 1 && strcmp(argv[1], "--notest") == 0))
        self_test();

    FILE *f = fopen("fib1000.txt", "r"); // Fibonacci Factorisations from https://mersennus.net/fibonacci/f1000.txt
    if (!f) {
        fprintf(stderr, "Error: cannot open fib1000.txt\n");
        goto cleanup;
    }

    char line[65536];
    int line_no = 0, checked = 0;

    while (fgets(line, sizeof(line), f)) {
        line_no++;
        if (line[0] != 'F') continue;

        int n = 0;
        char *p = line + 1;
        if (*p != '(') continue;
        p++;
        while (*p >= '0' && *p <= '9') { n = n*10 + (*p-'0'); p++; }
        if (*p != ')') continue;
        if (n < 12 || n > 1000 || n % 12 != 0) continue;

        char *eq = strchr(line, '=');
        if (!eq) continue;
        eq++;
        while (*eq==' '||*eq=='\t') eq++;

        g_np = 0;
        while (*eq && g_np < MAX_P) {
            while (*eq==' '||*eq=='*'||*eq=='\n'||*eq=='\r'||*eq=='\t') 
eq++;
            if (!*eq || (*eq<'0'||*eq>'9')) break;
            char *ds = eq;
            while (*eq>='0' && *eq<='9') eq++;
            char tc = *eq; *eq = 0;
            mpz_set_str(g_primes[g_np], ds, 10);
            *eq = tc;
            if (*eq == '^') {
                eq++;
                g_exps[g_np] = atoi(eq);
                while (*eq>='0' && *eq<='9') eq++;
            } else {
                g_exps[g_np] = 1;
            }
            g_np++;
        }

        if (g_np == 0) {
            fprintf(stderr, "  WARN: F(%d) parsed 0 primes (line %d)\n", 
n, line_no);
            continue;
        }

        mpz_set_ui(g_N, 1);
        for (int i = 0; i < g_np; i++) {
            mpz_t pe; mpz_init(pe);
            mpz_pow_ui(pe, g_primes[i], (unsigned long)g_exps[i]);
            mpz_mul(g_N, g_N, pe);
            mpz_clear(pe);
        }

        printf("F(%d): ", n);
        for (int i = 0; i < g_np; i++) {
            gmp_printf("%Zd^%d", g_primes[i], g_exps[i]);
            if (i+1 < g_np) printf(" * ");
        }
        printf("\n");
        fflush(stdout);

        run_search();
        checked++;
        printf("  -> %ld leaves\n", g_leaf_count);
        fflush(stdout);

        if (g_found) {
            printf("\n*** FOUND: F(%d) is a Pythagorean area! ***\n", n);
            printf("  d = "); gmp_printf("%Zd\n", g_sol_d);
            printf("  m = "); gmp_printf("%Zd\n", g_sol_m);
            printf("  n = "); gmp_printf("%Zd\n", g_sol_n);
            printf("  q = m^2-n^2 = "); gmp_printf("%Zd\n", g_sol_q);

            mpz_t mm, nn, av, bv, cv, av2, bv2, cv2;
            mpz_init(mm); mpz_init(nn);
            mpz_init(av); mpz_init(bv); mpz_init(cv);
            mpz_init(av2); mpz_init(bv2); mpz_init(cv2);

            mpz_mul(mm, g_sol_m, g_sol_m);
            mpz_mul(nn, g_sol_n, g_sol_n);
            mpz_sub(av, mm, nn); mpz_mul(av, av, g_sol_d);
            mpz_mul(bv, g_sol_m, g_sol_n);
            mpz_mul_ui(bv, bv, 2); mpz_mul(bv, bv, g_sol_d);
            mpz_add(cv, mm, nn); mpz_mul(cv, cv, g_sol_d);

            printf("  a = "); gmp_printf("%Zd\n", av);
            printf("  b = "); gmp_printf("%Zd\n", bv);
            printf("  c = "); gmp_printf("%Zd\n", cv);

            mpz_mul(av2, av, av);
            mpz_mul(bv2, bv, bv);
            mpz_add(av2, av2, bv2);
            mpz_mul(cv2, cv, cv);
            printf("  a^2+b^2 = "); gmp_printf("%Zd\n", av2);
            printf("  c^2     = "); gmp_printf("%Zd\n", cv2);

            mpz_clear(mm); mpz_clear(nn);
            mpz_clear(av); mpz_clear(bv); mpz_clear(cv);
            mpz_clear(av2); mpz_clear(bv2); mpz_clear(cv2);
            goto cleanup;
        }
    }

    printf("\nTotal F(12m) checked: %d\n", checked);
    if (!g_found)
        printf("No F(12m) with 12m <= 1000 is a Pythagorean area.\n");

cleanup:
    if (f) fclose(f);
    mpz_clear(g_N);
    mpz_clear(g_sol_d); mpz_clear(g_sol_m);
    mpz_clear(g_sol_n); mpz_clear(g_sol_q);
    for (int i = 0; i < MAX_P; i++) {
        mpz_clear(g_primes[i]);
        for (int k = 0; k < MAX_E; k++) mpz_clear(ppow[i][k]);
    }
    return 0;
}

