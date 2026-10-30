/* ============================================================================
*  MPI MCMC for cosmological parameter estimation
 * ============================================================================
 *
 *  PIPELINE OVERVIEW
 *  -----------------
 *  1. Read param_reader.ini       → global_config (parameters, likelihoods, flags)
 *  2. MPI init                    → 1 master (rank 0) + CHAINS × SLAVEPARCHAIN slaves
 *  3. Startup scan (master0/slave0):
 *       - master0 draws 100 random starting points inside the prior box
 *       - slave0 evaluates χ² at each
 *       - master0 sorts them, keeps the CHAINS best as seeds
 *  4. Main MCMC (master/slave):
 *       - master proposes new θ via MultiGaussian (throwDice)
 *       - slave evaluates χ²(θ_new) — via CAMB/CLIK, or via emulator with
 *         fallback to CAMB when uncertainty exceeds threshold
 *       - master does Metropolis-Hastings accept/reject
 *       - every UPDATE_TIME accepted steps: recompute covariance across all
 *         chains, update proposal matrix, broadcast to slaves
 *       - every ~RMIN_POINTS steps: Gelman-Rubin R̂; if R̂ < RBREAK for all
 *         parameters and min chain length > MIN_SIZE_FOR_FREEZE_IN, freeze
 *         the covariance (stop adapting) and continue to fixed-length chain
 *  5. Stop when max chain length reached; send TAG_STOP to all slaves
 *
 *  THEORY — METROPOLIS-HASTINGS
 *  ----------------------------
 *  At each step, draw proposal θ' ~ q(θ' | θ). Accept with probability
 *       α = min(1, exp(logL(θ') − logL(θ)))                  [symmetric q]
 *  where logL includes χ²/2 + prior. Draw u ~ U(0,1); accept if u < α.
 *  For flat priors, exp(−Δχ²/2) is the acceptance ratio.
 *
 *  ADAPTIVE PROPOSAL
 *  -----------------
 *  After each cov-update window, compute the running covariance Σ of the
 *  chain across all CHAINS. Eigendecompose Σ = V Λ Vᵀ, draw y ~ N(0,Λ),
 *  propose θ' = θ + γ V y, where γ is a global scale factor tuned so that
 *  ~23% of proposals are accepted (optimal for high-dimensional MH).
 *
 *  TASK ARRAY LAYOUT
 *  -----------------
 *  Each MPI message is a fixed-length double[] (TASKARRAY_SIZE) carrying
 *  both the parameter vector and control/metadata. Positions are computed
 *  once in setVariables():
 *
 *      [0 .. PARAMETERS-1]                 θ (the MCMC state)
 *      [PARAMETERS]                        χ² of proposal
 *      [LOGLIKEPOS]                        logL of proposal = -χ²/2
 *      [MULTIPURPOSEPOS]                   # consecutive rejects (for adapt)
 *      [ADAPTIVEPOS]                       1 = covariance still adapting
 *      [DRAGPOS]                           1 = use drag/path sampling
 *      [FASTFACTORPOS]                     Fast-param step scale (drag mode)
 *      [TAKEPOS]                           MH decision (0/1)
 *      [PROBPOS]                           U(0,1) draw used in MH
 *      [RANDPOS..]                         Gaussian draws from proposal
 *      [EMU_BIAS_POS]                      |χ²_emu − χ²_CAMB| on accepts
 *      [EMU_USED_POS]                      1 = last step used emulator
 *      [CAMB_LOGLIKE_POS]                  cached logL from CAMB
 *      [EMU_LOGLIKE_POS]                   cached logL from emulator
 *      [STEP_POS]                          global MCMC step index
 *      [CURRENTSTATEPOS .. +PARAMETERS]    copy of θ at start of step
 *
 *  Anything not on this layout must be a hardcoded constant. That is
 *  why so many #defines map to global_config fields rather than to
 *  local variables — the master/slave code must see identical values.
 *
 *  EMULATOR WORKFLOW (when USE_EMULATOR=1)
 *  ---------------------------------------
 *  - Warm-up: slave collects (θ, Cls, logL) every accepted step.
 *  - After min_train_points samples, emulator fits PCA + per-mode poly/GP.
 *  - Subsequent proposals: slave calls emulator_predict to get Cls + σ.
 *  - Gate: if σ > FALLBACK_THRESHOLD, fall back to CAMB for that step.
 *  - On every *accepted* emulator step (unless TRAIN_ON_FALLBACK_ONLY):
 *      re-evaluate with CAMB to (a) verify, (b) get training data.
 *  - Track |χ²_emu − χ²_CAMB| in a rolling window; if mean > EMU_BIAS_MAX,
 *    quarantine the emulator for N_RECOVERY steps (master tells slave).
 *  - Free-reject training: CAMB-mode rejects with 0 ≤ Δχ² < 10 are added
 *    to the buffer at up to 2× the accept rate, as extra training data.
 *
 *  ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <string.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include "compile_config.h"        /* MAX_* caps + DEF_* runtime defaults */
#include "param_config_reader.h"   /* Config, ParameterConfig, load_config */
#include "camb_interface.h"        /* param_iface (CAMB), param_iface_bg */
double get_time(void);
#include "likelihood_interface.h"  /* calc_likelihood, calc_likelihood_drag */
#include "emulator/include/emulator.h"

/* --------------------------------------------------------------------------
 *  RUNTIME FLAG ALIASES
 *
 *  The ini parser stores these in global_config. The #defines below make
 *  the rest of the file read like constant usage — but they resolve at
 *  runtime, so `USE_EMULATOR` is `global_config.use_emulator` everywhere.
 *  This is deliberate: CLI overrides (-emu, -no-emu, etc.) only need to
 *  modify global_config once, and everything downstream sees the change.
 * -------------------------------------------------------------------------- */
#define EMULATOR_TYPE          (global_config.emulator_type)
#define POLY_DEGREE            (global_config.poly_degree)
#define FALLBACK_THRESHOLD     (global_config.fallback_threshold)

#define USE_EMULATOR           (global_config.use_emulator)
#define USE_DRAGGING           (global_config.use_dragging)
#define VERBOSE                (global_config.verbose)
#define TRAIN_ON_FALLBACK_ONLY (global_config.train_on_fallback_only)


#define DEBUG_PRINT(...) do { if (VERBOSE) { printf(__VA_ARGS__); fflush(stdout); } } while (0)

/* --------------------------------------------------------------------------
 *  FILE-SCOPE STATE (only master uses some, only slaves use others)
 * -------------------------------------------------------------------------- */
static Emulator *g_emulator = NULL;   /* slave-local: NULL on master */
static int local_step  = 0;           /* slave-local: count of training samples */
static int g_n_cosmo   = 0;           /* number of slow params (emulator input dim) */
static int mapping_done = 0;          /* one-shot flag for slave init */
static char outdir_buf[512];          /* broadcast from master's -o or ini */
static const char *LOAD_COV_PATH = NULL;  /* ini or -load-cov, master-only */


/* --------------------------------------------------------------------------
 *  NUMERICAL RECIPES STYLE HELPERS (legacy — kept for compatibility)
 * -------------------------------------------------------------------------- */
#define NR_END 1
#define FREE_ARG char *

static short int Astier = 0;          /* unused placeholder */
short int PARAMETERS = NOPARAM;       /* # of sampled parameters, set in setVariables() */

double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* --------------------------------------------------------------------------
 *  EMULATOR / ADAPTATION TUNABLES (ini → global_config → here)
 * -------------------------------------------------------------------------- */
#define N_DRAG                 (global_config.n_drag)

#define EMU_BIAS_MIN_N         (global_config.emu_bias_min_n)
#define N_RECOVERY             (global_config.n_recovery)
#define MAX_EMU_RESCUES        (global_config.max_emu_rescues)
#define EMU_BIAS_MAX           (global_config.emu_bias_max)
#define FREE_REJECT_DCHI2_MAX  (global_config.free_reject_dchi2_max)
#define FREE_REJECT_RATIO      (global_config.free_reject_ratio)

/* Per-chain ring buffer of emulator |χ²_emu − χ²_CAMB| bias on accepts.
 * Used by master to decide when to quarantine the emulator.
 * Once EMU_BIAS_MIN_N samples exist and their mean exceeds EMU_BIAS_MAX,
 * send TAG_EMU_QUARANTINE to that slave. Max MAX_EMU_RESCUES fires. */
static int    emu_ring_head[CHAINS]        = {0};
static int    emu_ring_n[CHAINS]           = {0};
static double emu_ring[CHAINS][EMU_BIAS_WINDOW];
static int    emu_quarantine_fires[CHAINS] = {0};
static int    g_emu_quarantine_left        = 0;   /* slave-local counter */

/* --------------------------------------------------------------------------
 *  PARAMETER CLASSIFICATION
 *
 *  slow: passes through CAMB (Omega_m_h2, ..., A_s). Emulator input dim = N_SLOW.
 *  fast: nuisances. Never enter CAMB. Only matter for the CLIK likelihood.
 *
 *  Drag mode exploits this: slow params get path-sampled, fast params get
 *  random-walk with the cached slow chi2. The fast-parameter covariance is
 *  eigendecomposed once per cov-update and broadcast to slaves.
 * -------------------------------------------------------------------------- */
static const char *SLOW_PARAM_NAMES[] = { "Omega_m_h2", "Omega_b_h2", "h", "tau", "n_s", "A_s" };
#define N_SLOW_NAMES ((int)(sizeof(SLOW_PARAM_NAMES)/sizeof(SLOW_PARAM_NAMES[0])))

int slow_idx[NOPARAM], fast_idx[NOPARAM];
int N_SLOW = 0, N_FAST = 0;

/* Fast-parameter eigendecomposition, broadcast from master to slaves.
 * MAX_FAST caps fast dim; packed buffer carries eigenvalues + eigenvectors. */
double fast_eval[MAX_FAST];
double fast_evec[MAX_FAST][MAX_FAST];
double fast_EntireFactor = 0.5;
int    fast_cov_ready = 0;
double fast_cov_buffer[MAX_FAST_PACKED];

Config global_config;   /* populated by load_config("param_reader.ini") */

/* --------------------------------------------------------------------------
 *  TASK ARRAY INDEX MAP (see header comment for full layout)
 * -------------------------------------------------------------------------- */
unsigned short int CURRENTSTATEPOS;   /* start of the copy of θ from prev step */
unsigned short int MULTIPURPOSEPOS;   /* # consecutive rejects */
unsigned short int TAKEPOS, PROBPOS;
unsigned short int ADAPTIVEPOS;
unsigned int LOGLIKEPOS;
int RANDPOS;
unsigned short int DRAGPOS;
unsigned short int FASTFACTORPOS;
int STEP_POS;
unsigned short int EMU_BIAS_POS;
unsigned short int EMU_USED_POS;
unsigned short int CAMB_LOGLIKE_POS;
unsigned short int EMU_LOGLIKE_POS;

static unsigned int MULTIPURPOSE_REQUIRED = 1;
static short int WMAP7 = 1;

/* --------------------------------------------------------------------------
 *  ADAPTATION / CONVERGENCE TUNABLES
 *
 *  BEGINCOVUPDATE   : first cov-update step count
 *  UPDATE_TIME      : how many accepts between cov updates
 *  DECREASE/INCREASE_STEP : global scale γ multipliers on acceptance drift
 *  HIGH/LOW_STEP_BOUND    : thresholds for γ adjustment
 *  BEGINDRAGGING    : chain length at which drag is enabled
 *  RMIN_POINTS      : min chain length before computing R̂
 *  RBREAK           : R̂ threshold for convergence
 *  MIN_SIZE_FOR_FREEZE_IN : min chain length to allow freeze-in
 *  MAXPREVIOUSPOINTS      : sliding window size for cov computation
 * -------------------------------------------------------------------------- */
#define BEGINCOVUPDATE         (global_config.begin_cov_update)
#define BEGINDRAGGING          (global_config.begin_dragging)
#define MAXCHAINLENGTH         (global_config.max_chain_length)
#define DECREASE_STEP          (global_config.decrease_step)
#define INCREASE_STEP          (global_config.increase_step)
#define HIGH_STEP_BOUND        (global_config.high_step_bound)
#define LOW_STEP_BOUND         (global_config.low_step_bound)
#define UPDATE_TIME            (global_config.update_time)
#define MAXPREVIOUSPOINTS      (global_config.max_previous_points)
#define RMIN_POINTS            (global_config.rmin_points)
#define RBREAK                 (global_config.rbreak)
#define MIN_SIZE_FOR_FREEZE_IN (global_config.min_size_for_freeze_in)

extern void test_likelihood_(int *inputflag, double *cl_in_tt, double *cl_in_te, double *cl_in_ee, double *cl_in_bb);
double drfactor;               /* global scale on the proposal step (adapts) */
int clik_process_initialized = 0;

/* --------------------------------------------------------------------------
 *  CHAIN NODE
 *
 *  Each entry stores the full task array plus bookkeeping. Multiplicity
 *  = how many rejected proposals happened before the accept that ended here.
 *  ReallyInvestigated = consecutive rejects since last accept (resets on accept).
 *
 *  Both fields feed the adaptive-step logic: a proposal that produces many
 *  consecutive rejects signals too-large steps → shrink γ.
 * -------------------------------------------------------------------------- */
typedef struct Task { double f[MAX_TASKARRAY_SIZE]; int Multiplicity; int ReallyInvestigated; } Task;

Task *free_Task() {
    Task *tsk = (Task *)malloc(sizeof(Task));
    for (int i = 0; i < MAX_TASKARRAY_SIZE; i++) tsk->f[i] = 0.0;
    tsk->Multiplicity = 0;
    tsk->ReallyInvestigated = 0;
    return tsk;
}

void Task_copy(Task *tska, Task *tskb) {
    for (int i = 0; i < MAX_TASKARRAY_SIZE; i++) tska->f[i] = tskb->f[i];
    tska->Multiplicity = tskb->Multiplicity;
    tska->ReallyInvestigated = tskb->ReallyInvestigated;
}

/* --------------------------------------------------------------------------
 *  ROLLING AVERAGE
 *
 *  Used to track the acceptance rate over the last `Size` steps. The
 *  average is used as the optimal step-size γ when freeze-in fires (≈0.23–0.3
 *  for high-dimensional MH).
 *
 *  Periodically recomputes from the ring buffer to prevent float drift.
 * -------------------------------------------------------------------------- */
typedef struct RollingAverage { unsigned int Size, Number; double Sum; double *y; unsigned int idx; int PerformFullCounter; } RollingAverage;

void Rolling_Average_push(RollingAverage *RA, double x) {
    RA->Sum -= RA->y[RA->idx];
    RA->Sum += x;
    RA->y[RA->idx++] = x;
    if (RA->idx == RA->Size) RA->idx = 0;
    RA->Number++;
    if (RA->Number >= RA->Size) RA->Number = RA->Size;
}

double RollingAverage_average(RollingAverage *RA) {
    if (RA->Number == 0) return 0.0;
    if (RA->PerformFullCounter++ < 10) return RA->Sum / RA->Number;
    RA->PerformFullCounter = 0;
    double Sum = 0.0;
    for (unsigned int i = 0; i < RA->Number; i++) Sum += RA->y[i];
    return Sum / RA->Number;
}

RollingAverage *new_RollingAverage(int Size) {
    RollingAverage *RA = (RollingAverage*)malloc(sizeof(RollingAverage));
    RA->y = (double*)malloc(Size * sizeof(double));
    RA->Size = Size; RA->idx = 0; RA->Number = 0; RA->Sum = 0.0; RA->PerformFullCounter = 0;
    for (int i = 0; i < Size; i++) RA->y[i] = 0.0;
    return RA;
}

/* --------------------------------------------------------------------------
 *  MULTIGAUSSIAN PROPOSAL
 *
 *  Represents N(0, Σ) where Σ = V Λ Vᵀ is the eigendecomposition of the
 *  running chain covariance. Sample y ~ N(0, Λ) in eigen-coordinates, then
 *  transform to parameter space via V y.
 *
 *  MasterMatrix[i][j] = i-th eigenvector's j-th component (row-major V).
 *  eigenvalues[i]     = Λ_ii, already scaled by EntireFactor².
 *
 *  Bounds enforcement: if the proposal falls outside [lbounds, hbounds],
 *  throwDice returns 0 and the master redraws (rejection sampling against
 *  the prior box). This guarantees the prior support is respected without
 *  needing a hard prior penalty — the proposal never leaves the box.
 * -------------------------------------------------------------------------- */
typedef struct MultiGaussian {
  double Scale; unsigned int SIZE; double **MasterMatrix;
  double *eigenvalues; double *generatedValues;
  double *lbounds; double *hbounds; double *randomq; double *center;
} MultiGaussian;

MultiGaussian *new_MultiGaussian(int Size) {
  MultiGaussian *MG = (MultiGaussian *)malloc(sizeof(MultiGaussian));
  MG->SIZE = Size;
  MG->MasterMatrix = (double **)malloc(Size * sizeof(double *));
  for (unsigned int i = 0; i < Size; i++) MG->MasterMatrix[i] = (double *)malloc(Size * sizeof(double));
  MG->eigenvalues = (double *)malloc(Size * sizeof(double));
  MG->generatedValues = (double *)malloc(Size * sizeof(double));
  MG->lbounds = (double *)malloc(Size * sizeof(double));
  MG->hbounds = (double *)malloc(Size * sizeof(double));
  MG->center = (double *)malloc(Size * sizeof(double));
  MG->randomq = (double *)malloc(Size * sizeof(double));
  return MG;
}

void MultiGaussian_setBounds(MultiGaussian *MG, double lowBound[], double highBound[]) {
  for (unsigned int i = 0; i < MG->SIZE; i++) { MG->lbounds[i] = lowBound[i]; MG->hbounds[i] = highBound[i]; }
}

void printInfo(MultiGaussian *MG) {
  DEBUG_PRINT("************************************\n");
  DEBUG_PRINT("MultiGaussian::printInfo:\n");
  DEBUG_PRINT("Eigenvalues (we used scale = %f):\n", MG->Scale);
  for (unsigned int i = 0; i < MG->SIZE; i++) {
    DEBUG_PRINT("%e  --> sigma:  %e  ", MG->eigenvalues[i], sqrt(MG->eigenvalues[i]));
    DEBUG_PRINT("eigenvector w/o scale: %e  ", MG->eigenvalues[i] / MG->Scale);
    DEBUG_PRINT("  --> sigma: %e\n", sqrt(MG->eigenvalues[i] / MG->Scale));
  }
  DEBUG_PRINT("\nEigenvectors:\n");
  for (unsigned int i = 0; i < MG->SIZE; i++) {
    for (unsigned int j = 0; j < MG->SIZE; j++) DEBUG_PRINT("%e  ", MG->MasterMatrix[i][j]);
    DEBUG_PRINT("\n");
  }
  DEBUG_PRINT("***********************************\n");
}

void nrerror(char s[]) { fprintf(stderr, "%s", s); exit(1); }

/* Legacy Numerical Recipes helpers — allocate 1-indexed arrays */
double **convert_matrix(double *a, long nrl, long nrh, long ncl, long nch) {
  long i, j, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
  double **m = (double **)malloc((unsigned int)((nrow + NR_END) * sizeof(double *)));
  if (!m) nrerror("allocation failure in convert_matrix()");
  m += NR_END; m -= nrl;
  m[nrl] = a - ncl;
  for (i = 1, j = nrl + 1; i < nrow; i++, j++) m[j] = m[j - 1] + ncol;
  return m;
}
void free_convert_matrix(double **b, long nrl, long nrh, long ncl, long nch) { free((FREE_ARG)(b + nrl - NR_END)); }

double *vector(long nl, long nh) {
  double *v = (double *)malloc((unsigned int)((nh - nl + 1 + NR_END) * sizeof(double)));
  if (!v) nrerror("allocation failure in vector()");
  return v - nl + NR_END;
}
void free_vector(double *v, long nl, long nh) { free((FREE_ARG)(v + nl - NR_END)); }

double *new_double(unsigned short int n) { return (double *)malloc(n * sizeof(double)); }

/* Note: local `fabs` shadows <math.h>'s — used only where sign handling matters */
double fabs(double take) { return (take < 0) ? -take : take; }
double posRnd(double max) { return rand() * (1.0 / RAND_MAX) * max; }
double ran1(float x) { return 2 * (rand() * (1.0 / RAND_MAX) - 0.5) * x; }

/* Box–Muller normal deviate */
double gasdev(double mean, double std) {
  double rsq, v1, v2;
  do { v1 = ran1(1.0); v2 = ran1(1.0); rsq = v1*v1 + v2*v2; } while (rsq >= 1.0 || rsq == 0.0);
  return v1 * sqrt(-2.0 * log(rsq) / rsq) * std + mean;
}

/* Draw y ~ N(0, Λ), set generatedValues = V y */
void generateRandom(MultiGaussian *MG) {
  double *y = (double *)malloc(MG->SIZE * sizeof(double));
  for (unsigned int i = 0; i < MG->SIZE; i++) y[i] = gasdev(0.0, sqrt(MG->eigenvalues[i]));
  for (unsigned int i = 0; i < MG->SIZE; i++) {
    MG->randomq[i] = y[i];
    MG->generatedValues[i] = 0;
    for (unsigned int j = 0; j < MG->SIZE; j++) MG->generatedValues[i] += MG->MasterMatrix[i][j] * y[j];
  }
  free(y);
}

/* --------------------------------------------------------------------------
 *  throwDice — draw a full proposal in all PARAMETERS dimensions.
 *
 *  Returns 1 if the proposal is inside the prior box, 0 otherwise.
 *  Also stores the raw Gaussian draws in next->f[RANDPOS+i] so the slave
 *  can undo the random walk when implementing delayed rejection (used by
 *  some variants of the sampler — currently unused but kept for compat).
 *
 *  `scale` is passed to pow(drfactor, scale); the (int)result[PROBPOS]-1 in
 *  the master stores the scale used so the acceptance-rate adaptation can
 *  reconstruct it.
 * -------------------------------------------------------------------------- */
int throwDice(Task chain, Task *next, MultiGaussian *MG, int scale) {
  for (unsigned int i = 0; i < MG->SIZE; i++) MG->center[i] = chain.f[i];
  generateRandom(MG);
  for (unsigned int i = 0; i < PARAMETERS; i++) {
    next->f[i] = MG->generatedValues[i] * pow(drfactor, scale) + MG->center[i];
    next->f[RANDPOS + i] = MG->randomq[i] * pow(drfactor, scale);
  }
  for (unsigned int i = 0; i < MG->SIZE; i++) {
    double v = MG->generatedValues[i] * pow(drfactor, scale) + MG->center[i];
    if (v < MG->lbounds[i] || v > MG->hbounds[i]) return 0;
  }
  return 1;
}

/* --------------------------------------------------------------------------
 *  generateEigenvectors — eigendecompose the covariance into V, Λ.
 *
 *  `scale` multiplies every eigenvalue; equivalent to using Σ' = scale·Σ.
 *  This is how EntireFactor tunes the step size without touching V.
 *
 *  Uses GSL's symmetric eigensolver. Results sorted by ascending |λ|,
 *  so the largest-variance direction is the last row of MasterMatrix.
 * -------------------------------------------------------------------------- */
void generateEigenvectors(MultiGaussian *MG, double **covarianceMatrix, double scale) {
  MG->Scale = scale;
  double *data = (double *)malloc((MG->SIZE) * (MG->SIZE) * sizeof(double));
  int k = 0;
  for (unsigned int j = 0; j < MG->SIZE; j++)
    for (unsigned int i = 0; i < MG->SIZE; i++) { data[k] = covarianceMatrix[i][j]; k++; }
  gsl_matrix_view m = gsl_matrix_view_array(data, MG->SIZE, MG->SIZE);
  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(MG->SIZE);
  gsl_vector *eval = gsl_vector_alloc(MG->SIZE);
  gsl_matrix *evec = gsl_matrix_alloc(MG->SIZE, MG->SIZE);
  gsl_eigen_symmv(&m.matrix, eval, evec, w);
  gsl_eigen_symmv_free(w);
  gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_ASC);
  for (unsigned short int i = 0; i < MG->SIZE; i++) {
    MG->eigenvalues[i] = gsl_vector_get(eval, i);
    for (unsigned short int j = 0; j < MG->SIZE; j++) MG->MasterMatrix[i][j] = gsl_matrix_get(evec, i, j);
  }
  for (unsigned short int i = 0; i < MG->SIZE; i++) MG->eigenvalues[i] *= scale;
  gsl_vector_free(eval); gsl_matrix_free(evec); free(data);
}

/* --------------------------------------------------------------------------
 *  ADAPTIVE / FREEZE-IN FLAGS
 *
 *  ADAPTIVE = 1 → keep updating covariance and step size
 *  FREEZE_IN = 1 → freeze covariance when R̂ < RBREAK
 *  Both are global (all chains share the same rate) — this is deliberate;
 *  independent adaptation can bias the stationary distribution.
 * -------------------------------------------------------------------------- */
unsigned short int ADAPTIVE = 1;
unsigned short int FREEZE_IN = 1;

void printExtInfo(MultiGaussian *MG) {
  DEBUG_PRINT("\n********************************************\n");
  DEBUG_PRINT("EXTENDED information :\nSize : %d\nBounds:\n", MG->SIZE);
  for (unsigned int i = 0; i < MG->SIZE; i++) DEBUG_PRINT("%d) %e  %e  %e\n", i, MG->lbounds[i], MG->hbounds[i], MG->generatedValues[i]);
  DEBUG_PRINT("\n********************************************\n\n\n");
  for (unsigned int i = 0; i < MG->SIZE; i++) {
    DEBUG_PRINT("%e  :  ", MG->eigenvalues[i]);
    for (unsigned int j = 0; j < MG->SIZE; j++) DEBUG_PRINT("  %.2e", MG->MasterMatrix[i][j]);
    DEBUG_PRINT("\n");
  }
  DEBUG_PRINT("\n********************************************\n");
}

double normd(double a[], int size) { double s = 0.0; for (int i = 0; i < size; i++) s += a[i]*a[i]; return s; }

/* Read [lower, upper, proposal_sigma] for each sampled parameter from ini.
 * Order matches the order of sampled declarations in param_reader.ini —
 * this is the index space of `task[]` as seen by the sampler. */
void get_parameter_bounds_from_config(double *lowbound, double *highbound, double *initial_sigma) {
    int param_index = 0;
    for (int i = 0; i < global_config.param_count && param_index < PARAMETERS; i++) {
        if (global_config.params[i].state == PARAM_SAMPLED) {
            lowbound[param_index]      = global_config.params[i].lower_bound;
            highbound[param_index]     = global_config.params[i].upper_bound;
            initial_sigma[param_index] = global_config.params[i].proposal_sigma;
            DEBUG_PRINT("Chain Param %d: %s [%.3f, %.3f]\n",
                        param_index, global_config.params[i].name,
                        lowbound[param_index], highbound[param_index]);
            param_index++;
        }
    }
    for (; param_index < PARAMETERS; param_index++) {
        lowbound[param_index] = 0.0;
        highbound[param_index] = 1.0;
        initial_sigma[param_index] = 0.1;
    }
}

int myslave_rank(int ii)   { return ii + 1; }
int middleman(int rank)    { (void)rank; return 1; }
int mymiddlemann(int rank) { return rank; }

/* --------------------------------------------------------------------------
 *  throwSlowDice — draw a proposal in the SLOW-parameter subspace only.
 *  Used in drag mode, where fast parameters are perturbed separately by
 *  the slave (using its cached fast covariance).
 * -------------------------------------------------------------------------- */
int throwSlowDice(Task chain_pt, Task *next, MultiGaussian *MGs, int scale) {
    for (unsigned int k = 0; k < MGs->SIZE; k++) MGs->center[k] = chain_pt.f[slow_idx[k]];
    generateRandom(MGs);
    for (unsigned int i = 0; i < PARAMETERS; i++) next->f[i] = chain_pt.f[i];
    for (unsigned int k = 0; k < MGs->SIZE; k++) {
        int idx = slow_idx[k];
        next->f[idx] = MGs->generatedValues[k] * pow(drfactor, scale) + MGs->center[k];
        if (next->f[idx] < MGs->lbounds[k] || next->f[idx] > MGs->hbounds[k]) return 0;
    }
    return 1;
}

/* --------------------------------------------------------------------------
 *  broadcast_fast_covariance — send fast-parameter eigendecomposition to slaves.
 *
 *  Only meaningful in drag mode. Slaves cache the packed arrays in
 *  fast_eval / fast_evec, which the drag-step sampler uses to perturb
 *  fast parameters at every intermediate path point.
 * -------------------------------------------------------------------------- */
void broadcast_fast_covariance(double **cov_full, int worldsize) {
    if (N_FAST == 0) return;
    static MultiGaussian *MGfast = NULL;
    if (!MGfast) MGfast = new_MultiGaussian(N_FAST);

    double **covFast = (double**)malloc(N_FAST * sizeof(double*));
    for (int k = 0; k < N_FAST; k++) {
        covFast[k] = (double*)malloc(N_FAST * sizeof(double));
        for (int j = 0; j < N_FAST; j++) covFast[k][j] = cov_full[fast_idx[k]][fast_idx[j]];
    }
    generateEigenvectors(MGfast, covFast, 1.0);

    /* Pack: [n_fast, evals..., eigenvectors row-major...] */
    static double packed[MAX_FAST_PACKED];
    packed[0] = (double)N_FAST;
    for (int k = 0; k < N_FAST; k++) packed[1+k] = MGfast->eigenvalues[k];
    int p = 1 + N_FAST;
    for (int i = 0; i < N_FAST; i++)
        for (int j = 0; j < N_FAST; j++) packed[p++] = MGfast->MasterMatrix[i][j];
    for (int r = 1; r < worldsize; r++)
        MPI_Send(packed, p, MPI_DOUBLE, r, TAG_UPDATE_COV, MPI_COMM_WORLD);
    for (int k = 0; k < N_FAST; k++) free(covFast[k]);
    free(covFast);
}

void update_local_fast_covariance(double *buf) {
    int n = (int)buf[0];
    if (n != N_FAST) return;
    for (int k = 0; k < N_FAST; k++) fast_eval[k] = buf[1+k];
    int p = 1 + N_FAST;
    for (int i = 0; i < N_FAST; i++)
        for (int j = 0; j < N_FAST; j++) fast_evec[i][j] = buf[p++];
    fast_cov_ready = 1;
}

/* --------------------------------------------------------------------------
 *  COVARIANCE FILE PARSER (for -load-cov / LOAD_COV_PATH)
 *
 *  Format expected: blocks written by master's covarianceMatrices file.
 *  Each block:
 *      Chain: i Step: k Points used: N
 *      Params: p1 p2 ... pn
 *      (n rows of n numbers)
 *      Entire factor: f
 *
 *  Reads the last block (overwriting), and maps the file's parameter
 *  names to the MCMC's parameter indices by name. Unmatched names get
 *  diagonal prior from proposal_sigma.
 * -------------------------------------------------------------------------- */
static int current_param_index(const char *name) {
    int idx = 0;
    for (int i = 0; i < global_config.param_count; i++) {
        if (global_config.params[i].state == PARAM_SAMPLED) {
            if (strcmp(global_config.params[i].name, name) == 0) return idx;
            idx++;
        }
    }
    return -1;
}

static int load_covariance_from_file(const char *path, int n_expected, double **cov_out, double *entire_factor_out) {
    if (!path || n_expected <= 0 || !cov_out) return 0;
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "load_covariance: cannot open '%s'\n", path); return 0; }
    static char file_names[MAX_FILE_PARAMS][MAX_PARAM_NAME_LENGTH];
    static double tmp[MAX_FILE_PARAMS][MAX_FILE_PARAMS];
    char line[16384];
    int file_n = 0, row = 0, in_block = 0, have_mat = 0; double ef = 1.0;

    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "Chain:", 6) == 0) { in_block = 1; row = 0; file_n = 0; ef = 1.0; continue; }
        if (strncmp(line, "Params:", 7) == 0) {
            file_n = 0; const char *p = line + 7;
            while (*p && file_n < MAX_FILE_PARAMS) {
                while (*p == ' ' || *p == '\t') p++;
                if (!*p || *p == '\n' || *p == '\r') break;
                const char *start = p;
                while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
                int len = (int)(p - start); if (len >= MAX_PARAM_NAME_LENGTH) len = MAX_PARAM_NAME_LENGTH - 1;
                memcpy(file_names[file_n], start, len); file_names[file_n][len] = '\0'; file_n++;
            }
            continue;
        }
        if (strncmp(line, "Entire factor:", 14) == 0) {
            if (in_block && file_n > 0 && row == file_n) {
                for (int i = 0; i < n_expected; i++) for (int j = 0; j < n_expected; j++) cov_out[i][j] = 0.0;
                int cur = 0;
                for (int i = 0; i < global_config.param_count && cur < n_expected; i++) {
                    if (global_config.params[i].state == PARAM_SAMPLED) {
                        double s = global_config.params[i].proposal_sigma; if (s <= 0) s = 1.0;
                        cov_out[cur][cur] = s * s; cur++;
                    }
                }
                int file_hits = 0, file_misses = 0;
                for (int fi = 0; fi < file_n; fi++) {
                    int ci = current_param_index(file_names[fi]); if (ci < 0 || ci >= n_expected) { file_misses++; continue; }
                    file_hits++;
                    for (int fj = 0; fj < file_n; fj++) {
                        int cj = current_param_index(file_names[fj]); if (cj < 0 || cj >= n_expected) continue;
                        cov_out[ci][cj] = tmp[fi][fj];
                    }
                }
                printf("Covariance: %d params in file, %d matched, %d unused\n", file_n, file_hits, file_misses); fflush(stdout);
                double v = 1.0; if (sscanf(line, "Entire factor: %lf", &v) == 1) ef = v;
                if (entire_factor_out) *entire_factor_out = ef; have_mat = 1;
            }
            row = 0; continue;
        }
        if (line[0] == '*' || line[0] == '\n' || line[0] == '\r') continue;
        if (in_block && row < MAX_FILE_PARAMS) {
            if (file_n == 0) {
                file_n = n_expected; int cur = 0;
                for (int i = 0; i < global_config.param_count && cur < n_expected; i++) {
                    if (global_config.params[i].state == PARAM_SAMPLED) {
                        strncpy(file_names[cur], global_config.params[i].name, MAX_PARAM_NAME_LENGTH - 1);
                        file_names[cur][MAX_PARAM_NAME_LENGTH - 1] = '\0'; cur++;
                    }
                }
            }
            if (row >= file_n) continue;
            double vals[MAX_FILE_PARAMS]; int cnt = 0; const char *p = line;
            while (*p && cnt < file_n) {
                char *end; double v = strtod(p, &end); if (end == p) break;
                vals[cnt++] = v; p = end;
            }
            if (cnt == file_n) { for (int j = 0; j < file_n; j++) tmp[row][j] = vals[j]; row++; }
            else { row = 0; in_block = 0; }
        }
    }
    fclose(fp); return have_mat;
}

/* ============================================================================
 *  MASTER — the MCMC orchestrator (runs only on rank 0)
 *
 *  RESPONSIBILITIES
 *    - Own the chain state (chain[chain][step])
 *    - Propose new points via throwDice / throwSlowDice
 *    - Receive χ² from slaves and apply MH acceptance
 *    - Periodically recompute covariance across all chains and rebroadcast
 *    - Run Gelman-Rubin R̂; freeze-in when converged
 *    - Write out chain files (montecarlo_chain, head, investigated, etc.)
 *
 *  PROTOCOL (with slaves)
 *    slave → master : GIVEMETASK (initial), TAKERESULT (with decision)
 *    master → slave : TAKETASK (proposal), TAG_UPDATE_COV, TAG_EMU_QUARANTINE,
 *                     TAG_STOP
 *
 *  MAIN LOOP
 *    Receive one message. If GIVEMETASK: build a new proposal, send it back.
 *    If TAKERESULT: record decision, maybe update covariance, maybe
 *    check convergence, maybe emit progress, maybe stop.
 * ============================================================================ */
int master(double **startnum, unsigned short int Restart) {
    DEBUG_PRINT("\nStarting master with %d chains and max chain length %d", CHAINS, MAXCHAINLENGTH);
    int total_proposals = 0;

    /* Per-chain bookkeeping.
     * chainSize[i]       = number of accepts on chain i
     * chainTotalPerformed[i] = number of proposals on chain i
     * chainBack[i]       = next write index in chain[i] (chainBack[i]-1 = last)
     * steps_since_update[i] = accepts since last covariance update
     */
    unsigned int chainSize[CHAINS] = {0};
    unsigned int chainTotalPerformed[CHAINS] = {0};
    unsigned int chainBack[CHAINS] = {0};
    unsigned int checkSize = 0;

    Task **chain = (Task**)malloc(CHAINS * sizeof(Task*));
    for (int c = 0; c < CHAINS; c++) chain[c] = (Task*)malloc(MAXCHAINLENGTH * sizeof(Task));

    int steps_since_update[CHAINS] = {0};
    double EntireFactor[CHAINS];           /* global step scale, adapts */
    RollingAverage *roll[CHAINS];          /* 500-step acceptance rate */
    double *sigma[CHAINS];                 /* per-param proposal sigma (init) */
    double **covMatrix[CHAINS];            /* per-chain covariance (init = diag) */
    for (int i = 0; i < CHAINS; i++) {
        sigma[i] = new_double(TASKARRAY_SIZE);
        covMatrix[i] = (double**)malloc(PARAMETERS * sizeof(double*));
        for (int j = 0; j < PARAMETERS; j++) covMatrix[i][j] = (double*)malloc(PARAMETERS * sizeof(double));
    }

    Task *next[CHAINS];
    MultiGaussian *mGauss[CHAINS];         /* full covariance eigendecomp */
    MultiGaussian *mGaussSlow[CHAINS];     /* slow-subspace eigendecomp (drag) */
    double lowboundSlow[NOPARAM], highboundSlow[NOPARAM];

    double *lowbound, *highbound, *initial_sigma;
    unsigned int break_at, N;
    double *y[CHAINS], *dist_y, *B, *W, *R, *average;
    double **cov;

    lowbound = new_double(PARAMETERS); highbound = new_double(PARAMETERS);
    initial_sigma = new_double(PARAMETERS); dist_y = new_double(PARAMETERS);
    B = new_double(PARAMETERS); W = new_double(PARAMETERS);
    R = new_double(PARAMETERS); average = new_double(PARAMETERS);
    for (int i = 0; i < CHAINS; i++) y[i] = new_double(PARAMETERS);
    cov = (double**)malloc(PARAMETERS * sizeof(double*));
    for (int i = 0; i < PARAMETERS; i++) cov[i] = new_double(PARAMETERS);

    get_parameter_bounds_from_config(lowbound, highbound, initial_sigma);
    srand((unsigned)time(NULL));

    /* Reset emulator bias rings */
    for (int i = 0; i < CHAINS; i++) {
        emu_ring_head[i] = 0; emu_ring_n[i] = 0; emu_quarantine_fires[i] = 0;
    }

    /* ------------------------------------------------------------------
     * Optional: load a previously-saved covariance matrix.
     * Enables restart from a prior run, or seeding from a crude first pass.
     * Validation checks symmetry and positive-definiteness of diagonals —
     * a bad matrix causes odd behavior later (asymmetric proposals), so
     * reject up front and fall back to diagonal.
     * ------------------------------------------------------------------ */
    int have_loaded_cov = 0;
    double **loaded_cov = NULL;
    double loaded_ef = 1.0;
    if (LOAD_COV_PATH) {
        loaded_cov = (double**)malloc(PARAMETERS * sizeof(double*));
        for (int i = 0; i < PARAMETERS; i++) loaded_cov[i] = (double*)malloc(PARAMETERS * sizeof(double));

        if (load_covariance_from_file(LOAD_COV_PATH, PARAMETERS, loaded_cov, &loaded_ef)) {
            int bad = 0;
            for (int k = 0; k < PARAMETERS && !bad; k++)
                for (int j = k+1; j < PARAMETERS && !bad; j++) {
                    double s = fabs(loaded_cov[k][k]) + 1e-30;
                    if (fabs(loaded_cov[k][j] - loaded_cov[j][k]) > 1e-6 * s) bad = 1;
                }
            for (int k = 0; k < PARAMETERS && !bad; k++) if (!isfinite(loaded_cov[k][k]) || loaded_cov[k][k] <= 0.0) bad = 1;

            if (bad) {
                fprintf(stderr, "WARNING: loaded covariance failed validation; using diagonal prior\n");
                for (int i = 0; i < PARAMETERS; i++) free(loaded_cov[i]);
                free(loaded_cov); loaded_cov = NULL; have_loaded_cov = 0;
            } else {
                have_loaded_cov = 1;
                printf("Loaded covariance from %s (EntireFactor = %g)\n", LOAD_COV_PATH, loaded_ef);
            }
        } else {
            fprintf(stderr, "WARNING: could not load covariance from %s\n", LOAD_COV_PATH);
            for (int i = 0; i < PARAMETERS; i++) free(loaded_cov[i]);
            free(loaded_cov); loaded_cov = NULL;
        }
    }

    /* ------------------------------------------------------------------
     * Initialize chains from the seed points returned by master0().
     * If `startnum` is NULL (or intelligentstartnum=0), draw uniform in box.
     * Otherwise each chain starts at startnum[i] (the top-CHAIN seed found
     * by master0 from its 100-sample scan).
     *
     * Emulator mode: logL, cached logL are packed into the seed task here.
     * ------------------------------------------------------------------ */
    int intelligentstartnum = 1;
    for (int i = 0; i < CHAINS; i++) {
        roll[i] = new_RollingAverage(500);
        Task t;
        for (int k = 0; k < TASKARRAY_SIZE; k++) t.f[k] = 0;
        mGauss[i] = new_MultiGaussian(PARAMETERS);
        MultiGaussian_setBounds(mGauss[i], lowbound, highbound);

        if (!intelligentstartnum) for (int j = 0; j < PARAMETERS; j++) t.f[j] = lowbound[j] + posRnd(highbound[j] - lowbound[j]);
        else for (int j = 0; j < PARAMETERS; j++) t.f[j] = startnum[i][j];
        for (int j = 0; j < PARAMETERS; j++) t.f[CURRENTSTATEPOS + j] = t.f[j];

        for (int j = 0; j < PARAMETERS; j++) sigma[i][j] = initial_sigma[j];

        if (have_loaded_cov) {
            for (int k = 0; k < PARAMETERS; k++) for (int j = 0; j < PARAMETERS; j++) covMatrix[i][k][j] = loaded_cov[k][j];
            EntireFactor[i] = loaded_ef;
        } else {
            for (int k = 0; k < PARAMETERS; k++) for (int j = 0; j < PARAMETERS; j++)
                covMatrix[i][k][j] = (j == k) ? sigma[i][k]*sigma[i][k] : 0.0;
            EntireFactor[i] = 1.0;
        }

        generateEigenvectors(mGauss[i], covMatrix[i], EntireFactor[i]*EntireFactor[i]);
        steps_since_update[i] = 0;

        t.f[LOGLIKEPOS]       = -0.5 * startnum[i][PARAMETERS];
        t.f[CAMB_LOGLIKE_POS] = t.f[LOGLIKEPOS]; t.f[EMU_LOGLIKE_POS] = 1e30;
        t.f[EMU_USED_POS]     = 0.0; t.f[EMU_BIAS_POS] = 0.0;
        t.Multiplicity = 0; t.ReallyInvestigated = 0; chainBack[i]++;

        Task_copy(&chain[i][chainBack[i]-1], &t);
        chainSize[i] = 0; chainTotalPerformed[i] = 0;

        /* Seed the slave: send initial task */
        MPI_Send(t.f, TASKARRAY_SIZE, MPI_DOUBLE, myslave_rank(i), TAKETASK, MPI_COMM_WORLD);

        /* Slow-subspace eigendecomp for drag mode */
        for (int k = 0; k < N_SLOW; k++) {
            lowboundSlow[k] = lowbound[slow_idx[k]]; highboundSlow[k] = highbound[slow_idx[k]];
        }
        mGaussSlow[i] = new_MultiGaussian(N_SLOW);
        MultiGaussian_setBounds(mGaussSlow[i], lowboundSlow, highboundSlow);

        double **covSlow0 = (double**)malloc(N_SLOW * sizeof(double*));
        for (int k = 0; k < N_SLOW; k++) {
            covSlow0[k] = (double*)calloc(N_SLOW, sizeof(double));
            if (have_loaded_cov) for (int j = 0; j < N_SLOW; j++) covSlow0[k][j] = loaded_cov[slow_idx[k]][slow_idx[j]];
            else covSlow0[k][k] = sigma[i][slow_idx[k]] * sigma[i][slow_idx[k]];
        }
        generateEigenvectors(mGaussSlow[i], covSlow0, EntireFactor[i]*EntireFactor[i]);
        for (int k = 0; k < N_SLOW; k++) free(covSlow0[k]);
        free(covSlow0);
    }

    /* Initial fast-param eigendecomp broadcast (drag mode only) */
    if (have_loaded_cov && USE_DRAGGING && N_FAST > 0) {
        int worldsz; MPI_Comm_size(MPI_COMM_WORLD, &worldsz);
        broadcast_fast_covariance(loaded_cov, worldsz);
    }

    /* Open output files (append so -restart works). */
    FILE *data[CHAINS], *head[CHAINS], *investigated[CHAINS];
    char montecarlo_name[] = "montecarlo_chain", head_name[] = "head", investigated_name[] = "investigated";
    char path[512];
    for (int k = 1; k <= CHAINS; k++) {
        snprintf(path, sizeof(path), "%s/%s_%d.d", outdir_buf, montecarlo_name, k);
        data[k-1] = fopen(path, "a+");
        snprintf(path, sizeof(path), "%s/%s_%d.d", outdir_buf, head_name, k);
        head[k-1] = fopen(path, "a+");
        snprintf(path, sizeof(path), "%s/%s_%d.d", outdir_buf, investigated_name, k);
        investigated[k-1] = fopen(path, "a+");

        if (Restart == 0) {
            fprintf(data[k-1], "%-5s %-16s", "Mult", "Chi2");
            int printed = 0;
            for (int p = 0; p < global_config.param_count && printed < PARAMETERS; p++) {
                if (global_config.params[p].state == PARAM_SAMPLED) {
                    fprintf(data[k-1], " %-18s", global_config.params[p].name); printed++;
                }
            }
            fprintf(data[k-1], "\n"); fflush(data[k-1]);
        }
    }

    snprintf(path, sizeof(path), "%s/progress.txt", outdir_buf);
    FILE *progress = fopen(path, "a+");
    snprintf(path, sizeof(path), "%s/gelmanRubin.txt", outdir_buf);
    FILE *gelmanRubin = fopen(path, "a+");
    snprintf(path, sizeof(path), "%s/covMatrix.txt", outdir_buf);
    FILE *covarianceMatrices = fopen(path, "a+");

    if (Restart == 1) fprintf(progress, "\n\n::::::::::::::::::: RESTARTED FROM HERE ::::::::::::::::::::::::\n\n");

    /* ------------------------------------------------------------------
     * MAIN LOOP
     * ------------------------------------------------------------------ */
    short int work = 1;
    MPI_Status status;
    double result[MAX_TASKARRAY_SIZE];
    int mult[CHAINS];
    int take = 0;
    unsigned int min_size = 100*100;
    int loopstop = 0;
    int global_step = 0;

    do {
        global_step++;
        MPI_Recv(result, TASKARRAY_SIZE, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        /* ---------- SLAVE ASKS FOR A NEW PROPOSAL ---------- */
        if (status.MPI_TAG == GIVEMETASK) {
            int i = status.MPI_SOURCE - 1;
            int use_dragging = (USE_DRAGGING && N_FAST > 0 && chainSize[i] >= BEGINDRAGGING) ? 1 : 0;
            total_proposals += 1;

            /* Draw a valid proposal (may redraw up to 100 times).
             * If we can't find one, shrink EntireFactor — the prior box
             * must be smaller than the covariance suggests. */
            mult[i] = 0; next[i] = free_Task();
            for (;;) {
                if (use_dragging) { if (throwSlowDice(chain[i][chainBack[i]-1], next[i], mGaussSlow[i], 0)) break; }
                else              { if (throwDice(chain[i][chainBack[i]-1], next[i], mGauss[i], 0)) break; }
                mult[i]++;
                if (mult[i] > 100) {
                    EntireFactor[i] *= DECREASE_STEP;
                    generateEigenvectors(mGauss[i], covMatrix[i], EntireFactor[i]*EntireFactor[i]);
                    if (use_dragging) {
                        double **covSlow = (double**)malloc(N_SLOW * sizeof(double*));
                        for (int k = 0; k < N_SLOW; k++) {
                            covSlow[k] = (double*)malloc(N_SLOW * sizeof(double));
                            for (int j = 0; j < N_SLOW; j++) covSlow[k][j] = covMatrix[i][slow_idx[k]][slow_idx[j]];
                        }
                        generateEigenvectors(mGaussSlow[i], covSlow, EntireFactor[i]*EntireFactor[i]);
                        for (int k = 0; k < N_SLOW; k++) free(covSlow[k]);
                        free(covSlow);
                    }
                    mult[i] = 0;
                }
                if (mult[i] > 100000) MPI_Abort(MPI_COMM_WORLD, 1);
            }

            /* Copy current state so slave can compare θ_A vs θ_B */
            for (int k = 0; k < PARAMETERS; k++) next[i]->f[CURRENTSTATEPOS + k] = chain[i][chainBack[i]-1].f[k];

            /* Pack control metadata. The slave only needs the proposal θ,
             * the current state for χ²_A, and the mode flags.
             * STEP_POS is used for logging; PROBPOS is a fresh U(0,1) that
             * the slave will use for the MH decision (master could also
             * draw it — keeps RNG on master for reproducibility). */
            next[i]->f[STEP_POS]         = (double)global_step;
            next[i]->f[PROBPOS]          = posRnd(1.0);
            next[i]->f[ADAPTIVEPOS]      = (double)ADAPTIVE;
            next[i]->f[DRAGPOS]          = (double)use_dragging;
            next[i]->f[FASTFACTORPOS]    = fast_EntireFactor;
            next[i]->f[TAKEPOS]          = (double)ADAPTIVE;
            next[i]->f[LOGLIKEPOS]       = chain[i][chainBack[i]-1].f[LOGLIKEPOS];
            next[i]->f[MULTIPURPOSEPOS]  = (double)chain[i][chainBack[i]-1].ReallyInvestigated;
            next[i]->f[EMU_BIAS_POS]     = 0.0;
            next[i]->f[CAMB_LOGLIKE_POS] = chain[i][chainBack[i]-1].f[CAMB_LOGLIKE_POS];
            next[i]->f[EMU_LOGLIKE_POS]  = chain[i][chainBack[i]-1].f[EMU_LOGLIKE_POS];
            next[i]->f[EMU_USED_POS]     = chain[i][chainBack[i]-1].f[EMU_USED_POS];

            for (int k = 0; k < PARAMETERS; k++) fprintf(head[i], "%e\t", next[i]->f[k]);
            fprintf(head[i], "\n");

            MPI_Send(next[i]->f, TASKARRAY_SIZE, MPI_DOUBLE, myslave_rank(i), TAKETASK, MPI_COMM_WORLD);
        }

        /* ---------- SLAVE RETURNS A RESULT ---------- */
        if (status.MPI_TAG == TAKERESULT) {
            int i = status.MPI_SOURCE - 1;
            for (int k = 0; k <= PARAMETERS; k++) fprintf(investigated[i], "%e  ", result[k]);
            fprintf(investigated[i], " %d\n", i);

            take = (int)result[TAKEPOS];
            chainTotalPerformed[i] += 1;
            chain[i][chainBack[i]-1].Multiplicity += mult[i];

            if (take == 1) {
                /* ---------- ACCEPT ---------- */
                for (int k = 0; k < PARAMETERS; k++) fprintf(data[i], "%e  ", chain[i][chainBack[i]-1].f[k]);
                fprintf(data[i], "%e  %e  ", chain[i][chainBack[i]-1].f[LOGLIKEPOS], chain[i][chainBack[i]-1].f[LOGLIKEPOS]);
                fprintf(data[i], "%d \n", chain[i][chainBack[i]-1].Multiplicity);

                /* If acceptance is too high, increase step size */
                if (ADAPTIVE == 1 && chain[i][chainBack[i]-1].ReallyInvestigated < LOW_STEP_BOUND && EntireFactor[i] < 3) {
                    EntireFactor[i] *= INCREASE_STEP;
                    generateEigenvectors(mGauss[i], covMatrix[i], EntireFactor[i]*EntireFactor[i]);
                }

                /* Track acceptance-weighted drfactor for freeze-in later */
                if (ADAPTIVE == 1) Rolling_Average_push(roll[i], pow(drfactor, (int)result[PROBPOS] - 1));

                /* Advance chain */
                for (int k = 0; k < TASKARRAY_SIZE; k++) chain[i][chainBack[i]].f[k] = result[k];
                chainBack[i]++; steps_since_update[i]++; chainSize[i]++;
                chain[i][chainBack[i]-1].Multiplicity = 1;
                chain[i][chainBack[i]-1].ReallyInvestigated = 1;
            } else {
                /* ---------- REJECT ---------- */
                chain[i][chainBack[i]-1].Multiplicity += 1;
                chain[i][chainBack[i]-1].ReallyInvestigated += 1;
            }

            /* ---------- EMULATOR BIAS MONITOR ---------- */
            /* If a step used the emulator, slave reports |χ²_emu − χ²_CAMB|
             * in EMU_BIAS_POS. Track mean over window; if too high, quarantine. */
            if ((int)result[EMU_USED_POS] == 1 && result[EMU_BIAS_POS] > 0.0) {
                emu_ring[i][emu_ring_head[i]] = result[EMU_BIAS_POS];
                emu_ring_head[i] = (emu_ring_head[i] + 1) % EMU_BIAS_WINDOW;
                if (emu_ring_n[i] < EMU_BIAS_WINDOW) emu_ring_n[i]++;

                if (emu_ring_n[i] >= EMU_BIAS_MIN_N && emu_quarantine_fires[i] < MAX_EMU_RESCUES) {
                    double s = 0.0;
                    for (int k = 0; k < emu_ring_n[i]; k++) s += emu_ring[i][k];
                    double mean_bias = s / emu_ring_n[i];
                    if (mean_bias > EMU_BIAS_MAX) {
                        int sig = 1; MPI_Send(&sig, 1, MPI_INT, myslave_rank(i), TAG_EMU_QUARANTINE, MPI_COMM_WORLD);
                        emu_quarantine_fires[i]++; emu_ring_head[i] = 0; emu_ring_n[i] = 0;
                        printf("[EMU_RESCUE] step=%d chain=%d bias=%.4f fire=%d/%d\n", global_step, i, mean_bias, emu_quarantine_fires[i], MAX_EMU_RESCUES);
                        fflush(stdout);
                    }
                }
            }

            /* ---------- COVARIANCE UPDATE ---------- */
            /* Every UPDATE_TIME accepts (after BEGINCOVUPDATE), recompute Σ
             * across all chains using a sliding window of MAXPREVIOUSPOINTS
             * recent steps. Re-eigendecompose for each chain. */
            if (ADAPTIVE == 1 && steps_since_update[i] >= UPDATE_TIME && chainSize[i] >= BEGINCOVUPDATE) {
                int TotalSize = 0; unsigned int covSize[CHAINS];
                for (int ii = 0; ii < CHAINS; ii++) {
                    if (chainSize[ii] < 2 * BEGINCOVUPDATE) covSize[ii] = chainSize[ii] / 2;
                    else covSize[ii] = chainSize[ii] - BEGINCOVUPDATE;
                    if (covSize[ii] > MAXPREVIOUSPOINTS) covSize[ii] = MAXPREVIOUSPOINTS;
                }
                /* Compute mean */
                for (int k = 0; k < PARAMETERS; k++) {
                    average[k] = 0.0;
                    for (int j = 0; j < PARAMETERS; j++) cov[k][j] = 0.0;
                }
                for (int ii = 0; ii < CHAINS; ii++)
                    for (unsigned int n = chainSize[ii] - covSize[ii]; n < chainSize[ii]; n++) {
                        for (int k = 0; k < PARAMETERS; k++) average[k] += chain[ii][n].f[k] * chain[ii][n].Multiplicity;
                        TotalSize += chain[ii][n].Multiplicity;
                    }
                for (int k = 0; k < PARAMETERS; k++) average[k] /= (double)TotalSize;

                /* Compute weighted covariance across all chains */
                for (int ii = 0; ii < CHAINS; ii++)
                    for (unsigned int n = chainSize[ii] - covSize[ii]; n < chainSize[ii]; n++)
                        for (int k = 0; k < PARAMETERS; k++)
                            for (int j = 0; j < PARAMETERS; j++)
                                cov[k][j] += (chain[ii][n].f[k] - average[k]) * (chain[ii][n].f[j] - average[j]) * chain[ii][n].Multiplicity;
                for (int k = 0; k < PARAMETERS; k++)
                    for (int j = 0; j < PARAMETERS; j++) cov[k][j] /= (TotalSize - 1.0);

                for (int ii = 0; ii < CHAINS; ii++)
                    for (int k = 0; k < PARAMETERS; k++)
                        for (int j = 0; j < PARAMETERS; j++) covMatrix[ii][k][j] = cov[k][j];

                /* Re-eigendecompose per chain */
                for (int ii = 0; ii < CHAINS; ii++) {
                    generateEigenvectors(mGauss[ii], cov, EntireFactor[ii]*EntireFactor[ii]);
                    double **covSlow = (double**)malloc(N_SLOW * sizeof(double*));
                    for (int k = 0; k < N_SLOW; k++) {
                        covSlow[k] = (double*)malloc(N_SLOW * sizeof(double));
                        for (int j = 0; j < N_SLOW; j++) covSlow[k][j] = cov[slow_idx[k]][slow_idx[j]];
                    }
                    generateEigenvectors(mGaussSlow[ii], covSlow, EntireFactor[ii]*EntireFactor[ii]);
                    for (int k = 0; k < N_SLOW; k++) free(covSlow[k]);
                    free(covSlow);
                }

                /* Broadcast fast-param eigendecomp to slaves */
                if (USE_DRAGGING && N_FAST > 0) {
                    int worldsz; MPI_Comm_size(MPI_COMM_WORLD, &worldsz);
                    broadcast_fast_covariance(cov, worldsz);
                }

                /* Log the new matrix for later restart */
                fprintf(covarianceMatrices, "Chain: %d Step: %d Points used: %d\n", i+1, chainSize[i], covSize[i]);
                fprintf(covarianceMatrices, "Params:");
                int printed = 0;
                for (int p = 0; p < global_config.param_count && printed < PARAMETERS; p++) {
                    if (global_config.params[p].state == PARAM_SAMPLED) { fprintf(covarianceMatrices, " %s", global_config.params[p].name); printed++; }
                }
                fprintf(covarianceMatrices, "\n");
                for (int j = 0; j < PARAMETERS; j++) {
                    for (int k = 0; k < PARAMETERS; k++) fprintf(covarianceMatrices, "%e  ", cov[j][k]);
                    fprintf(covarianceMatrices, "\n");
                }
                fprintf(covarianceMatrices, "\nEntire factor: %e", EntireFactor[i]);
                fprintf(covarianceMatrices, "\n***********************************************************\n");
                for (int ii = 0; ii < CHAINS; ii++) steps_since_update[ii] = 0;
            }

            /* ---------- STEP-SIZE ADAPTATION ---------- */
            /* If logL is infinite (bad model), reset EntireFactor.
             * If too many consecutive rejects, decrease step size. */
            if (ADAPTIVE == 1) {
                if (chain[i][chainBack[i]-1].f[LOGLIKEPOS] < -1e9) {
                    EntireFactor[i] = 1.0;
                    generateEigenvectors(mGauss[i], covMatrix[i], EntireFactor[i]*EntireFactor[i]);
                } else if (chain[i][chainBack[i]-1].ReallyInvestigated > HIGH_STEP_BOUND) {
                    if (EntireFactor[i] > 1e-5) {
                        EntireFactor[i] *= DECREASE_STEP;
                        generateEigenvectors(mGauss[i], covMatrix[i], EntireFactor[i]*EntireFactor[i]);
                        double **covSlow = (double**)malloc(N_SLOW * sizeof(double*));
                        for (int k = 0; k < N_SLOW; k++) {
                            covSlow[k] = (double*)malloc(N_SLOW * sizeof(double));
                            for (int j = 0; j < N_SLOW; j++) covSlow[k][j] = covMatrix[i][slow_idx[k]][slow_idx[j]];
                        }
                        generateEigenvectors(mGaussSlow[i], covSlow, EntireFactor[i]*EntireFactor[i]);
                        for (int k = 0; k < N_SLOW; k++) free(covSlow[k]);
                        free(covSlow);
                    }
                }
            }

            /* ---------- PROGRESS LOGGING ---------- */
            min_size = 1000*1000;
            for (unsigned int ii = 0; ii < CHAINS; ii++) if (chainSize[ii] < min_size) min_size = chainSize[ii];

            fprintf(progress, "performed/Chainsize: ");
            for (unsigned int ii = 0; ii < CHAINS; ii++) fprintf(progress, "%d/%d  ", chainTotalPerformed[ii], chainSize[ii]);
            fprintf(progress, "min_size: %d    Multiplicity [Really]: ", min_size);
            for (unsigned int ii = 0; ii < CHAINS; ii++) fprintf(progress, "%d[%d]  ", chain[ii][chainBack[ii]-1].Multiplicity, chain[ii][chainBack[ii]-1].ReallyInvestigated);
            fprintf(progress, "\n");

            /* ---------- GELMAN-RUBIN DIAGNOSTIC + FREEZE-IN ---------- */
            /* Run R̂ occasionally (5% of the time, once min_size > RMIN_POINTS).
             * If convergence reached, freeze covariance at the current value
             * so the chain remains a valid MCMC (no ongoing adaptation). */
            if ((min_size > RMIN_POINTS) && (posRnd(1.0) > 0.95)) {
                break_at = min_size / 2; N = min_size - break_at;
                for (int k = 0; k < PARAMETERS; k++) { dist_y[k] = 0; B[k] = 0; W[k] = 0; for (int ii = 0; ii < CHAINS; ii++) y[ii][k] = 0; }
                double M = (double)CHAINS;

                /* Per-chain means over second half */
                for (int ii = 0; ii < CHAINS; ii++) {
                    int TotalMultiplicity = 0;
                    for (unsigned int n = break_at; n < min_size; n++) {
                        for (int k = 0; k < PARAMETERS; k++) y[ii][k] += chain[ii][n].f[k] * chain[ii][n].Multiplicity;
                        TotalMultiplicity += chain[ii][n].Multiplicity;
                    }
                    N = TotalMultiplicity;
                    for (int k = 0; k < PARAMETERS; k++) { y[ii][k] /= (double)N; dist_y[k] += y[ii][k] / M; }
                }
                /* Between-chain variance B and within-chain variance W */
                for (int k = 0; k < PARAMETERS; k++)
                    for (int ii = 0; ii < CHAINS; ii++) { double t1 = y[ii][k] - dist_y[k]; B[k] += t1 * t1 / (M - 1.0); }
                for (int ii = 0; ii < CHAINS; ii++)
                    for (unsigned int n = break_at; n < min_size; n++)
                        for (int k = 0; k < PARAMETERS; k++) { double t2 = chain[ii][n].f[k] - y[ii][k]; W[k] += t2 * t2 * chain[ii][n].Multiplicity; }
                /* R̂ = sqrt(((N-1)/N·W + (1+1/N)·B) / W) — computed as R² here */
                for (int k = 0; k < PARAMETERS; k++) { W[k] /= (M * (N - 1.0)); R[k] = (N - 1.0) / N * W[k] + B[k] * (1.0 + 1.0 / N); R[k] /= W[k]; }

                /* Freeze-in if converged and chain long enough */
                if (FREEZE_IN == 1 && min_size > MIN_SIZE_FOR_FREEZE_IN) {
                    unsigned short int ConvergenceReached = 1;
                    for (int m = 0; m < PARAMETERS; m++) if (R[m] > RBREAK) ConvergenceReached = 0;
                    if (ConvergenceReached == 1) {
                        ADAPTIVE = 0; FREEZE_IN = 0; drfactor = 1.0;
                        FILE *final;
                        snprintf(path, sizeof(path), "%s/freezeInEigenvectors.txt", outdir_buf);
                        final = fopen(path, "a+");
                        for (int k = 1; k <= CHAINS; k++) {
                            fclose(data[k-1]);
                            snprintf(path, sizeof(path), "%s/%s_freeze_%d.d", outdir_buf, montecarlo_name, k);
                            data[k-1] = fopen(path, "a+");
                            fprintf(data[k-1], "%-5s %-16s", "Mult", "Chi2");
                            int printed = 0;
                            for (int p = 0; p < global_config.param_count && printed < PARAMETERS; p++) {
                                if (global_config.params[p].state == PARAM_SAMPLED) { fprintf(data[k-1], " %-18s", global_config.params[p].name); printed++; }
                            }
                            fprintf(data[k-1], "\n"); fflush(data[k-1]);
                        }
                        for (int j = 0; j < CHAINS; j++) {
                            Task *keep = chain[j] + chainBack[j] - 1;
                            Task_copy(chain[j], keep);
                            /* Optimal factor = acceptance rate from rolling avg */
                            double OptimalFactor = 2.4 / sqrt((double)PARAMETERS);
                            OptimalFactor = RollingAverage_average(roll[j]);
                            EntireFactor[j] = OptimalFactor;
                            fprintf(final, "CHAIN[%d] Eigenvector and value info: ", j);
                            fprintf(final, "************************************\n");
                            fprintf(final, "MultiGaussian::printInfo:\n");
                            fprintf(final, "Eigenvalues (scale = %f):\n", mGauss[j]->Scale);
                            for (unsigned int ii = 0; ii < mGauss[j]->SIZE; ii++) {
                                fprintf(final, "%e  --> sigma: %e  ", mGauss[j]->eigenvalues[ii], sqrt(mGauss[j]->eigenvalues[ii]));
                                fprintf(final, "eigenvector w/o scale: %e  ", mGauss[j]->eigenvalues[ii]/mGauss[j]->Scale);
                                fprintf(final, "  --> sigma: %e\n", sqrt(mGauss[j]->eigenvalues[ii]/mGauss[j]->Scale));
                            }
                            fprintf(final, "\nEigenvectors:\n");
                            for (unsigned int ii = 0; ii < mGauss[j]->SIZE; ii++) {
                                for (unsigned int jj = 0; jj < mGauss[j]->SIZE; jj++) fprintf(final, "%e  ", mGauss[j]->MasterMatrix[ii][jj]);
                                fprintf(final, "\n");
                            }
                            fprintf(final, "***********************************\n");
                            printInfo(mGauss[j]);
                            fprintf(final, "\n\n");
                        }
                        fclose(final);
                        fprintf(covarianceMatrices, "Stopped adaptive stepsize at %d for all chains (FREEZE_IN=true)", min_size);
                    }
                }
                fprintf(progress, "Statistics: \n");
                for (int k = 0; k < PARAMETERS; k++) fprintf(progress, "%d  dist_y: %e   B: %e   W: %e     R[k]:  %e \n", k, dist_y[k], B[k], W[k], R[k]);
                if (FREEZE_IN) fprintf(progress, "Multiplicity (and EntireFactor): ");
                else fprintf(progress, "Multiplicity (and frozen EntireFactor): ");
                for (int ii = 0; ii < CHAINS; ii++) fprintf(progress, "%d (%e)  ", chain[ii][chainBack[ii]-1].Multiplicity, EntireFactor[ii]);
                fprintf(progress, "\n");
                if (min_size != checkSize) {
                    fprintf(gelmanRubin, "%d   ", break_at);
                    for (int n = 0; n < PARAMETERS; n++) fprintf(gelmanRubin, "%e   ", R[n]);
                    fprintf(gelmanRubin, "%d \n", min_size);
                    checkSize = min_size;
                }
            }

            /* ---------- TERMINATION ---------- */
            if (FREEZE_IN == 0) loopstop++;
            unsigned int max_size = 0;
            for (int ii = 0; ii < CHAINS; ii++) if (chainBack[ii] > max_size) max_size = chainBack[ii];
            if (max_size >= MAXCHAINLENGTH || total_proposals > 2500000) work = 0;

            fflush(data[i]); fflush(head[i]); fflush(investigated[i]);
            fflush(progress); fflush(gelmanRubin); fflush(covarianceMatrices);
        }
    } while (work);

    fclose(progress); fclose(gelmanRubin); fclose(covarianceMatrices);
    for (int k = 1; k <= CHAINS; k++) { fclose(data[k-1]); fclose(head[k-1]); fclose(investigated[k-1]); }
    if (loaded_cov) { for (int i = 0; i < PARAMETERS; i++) free(loaded_cov[i]); free(loaded_cov); loaded_cov = NULL; }

    /* Signal all slaves to stop */
    DEBUG_PRINT("\nMCMC finished. Sending stop signal to all slaves...\n");
    int worldsize; MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
    for (int r = 1; r < worldsize; r++) MPI_Send(NULL, 0, MPI_INT, r, TAG_STOP, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}

int fact(int a) { return (a <= 1) ? 1 : a * fact(a - 1); }
double mind(double x, double y) { return (x < y) ? x : y; }

/* Convenience wrapper: CAMB + full χ² + prior in one call */
double eval_full_chi2(int rank, double *theta, double *Cl_TT, double *Cl_TE, double *Cl_EE, double *Cl_BB) {
    int ok = param_iface(rank, theta, Cl_TT, Cl_TE, Cl_EE, Cl_BB);
    if (!ok) return 1e30;
    double dummy = 0.0;
    double chi2 = calc_likelihood(rank, theta, Cl_TT, Cl_TE, Cl_EE, Cl_BB, 0, &dummy, NULL);
    return chi2 + compute_prior_penalty(theta);
}

/* ============================================================================
 *  evaluate_proposal — evaluate χ² at proposal θ_B and current θ_A
 *
 *  INPUTS
 *    task[]           : full proposal + control (see header layout)
 *    Cl_*_A/_B        : scratch buffers for CAMB Cls
 *    *_lowbound, ...  : bounds for fast-param OOB checks
 *
 *  OUTPUTS (via pointers)
 *    use_emu_out      : did this step use emulator?
 *    uncertainty_out  : emulator σ (used for logging and gate)
 *    emu_chi2_out     : emulator χ² at θ_B (for bias ring)
 *    cur_loglike_out  : logL at θ_A (for MH ratio)
 *
 *  BEHAVIOUR
 *    - Tries emulator at both θ_A and θ_B.
 *    - use_emu = both predictions valid AND both σ below threshold
 *      AND emulator not currently quarantined.
 *    - If use_emu: compute χ² from emulator Cls.
 *      Else: fall back to CAMB at θ_B (and possibly at θ_A if cache invalid).
 *
 *    - If use_dragging: run N_DRAG intermediate path points, each with:
 *        · interpolation of slow params between θ_A and θ_B
 *        · random-walk of fast params using cached fast covariance
 *        · MH step on (1−w)·χ²_A + w·χ²_B with L_cur / L_trial
 *      Accept/reject the whole path via final α = exp(−mean Δχ²/2).
 *
 *  CACHING
 *    When prev_mode matches (same source used last step), and the chain
 *    hasn't been stuck (ReallyInvestigated ≤ 10), read logL at θ_A from
 *    the cached CAMB_LOGLIKE_POS / EMU_LOGLIKE_POS fields.
 *    This saves a full CAMB/emulator evaluation of θ_A on most steps.
 * ============================================================================ */
static void evaluate_proposal(
    int rank,
    double task[MAX_TASKARRAY_SIZE],
    double *Cl_TT_A, double *Cl_TE_A, double *Cl_EE_A, double *Cl_BB_A,
    double *Cl_TT_B, double *Cl_TE_B, double *Cl_EE_B, double *Cl_BB_B,
    double g_lowbound[], double g_highbound[], double g_sigma[],
    int *use_emu_out, double *uncertainty_out,
    double *emu_chi2_out,
    double *cur_loglike_out)
{
    /* Unpack θ_A (previous state) and θ_B (proposal) */
    double theta_A[MAX_TASKARRAY_SIZE] = {0}, theta_B[MAX_TASKARRAY_SIZE] = {0};
    for (int k = 0; k < PARAMETERS; k++) {
        theta_A[k] = task[CURRENTSTATEPOS + k];
        theta_B[k] = task[k];
    }

    double final_chi2    = 1e30;
    double proposal_chi2 = 1e30;
    int use_dragging = (int)task[DRAGPOS];
    double fast_EntireFactor_local = task[FASTFACTORPOS];

    /* Per-mode uncertainty from emulator at θ_A and θ_B */
    double uncertainty_A = 1e10, uncertainty_B = 1e10;
    double emu_cl_tt_A[2601], emu_cl_te_A[2601], emu_cl_ee_A[2601], emu_cl_bb_A[2601];
    double emu_cl_tt_B[2601], emu_cl_te_B[2601], emu_cl_ee_B[2601], emu_cl_bb_B[2601];

    int emu_use_A = 0, emu_use_B = 0;

    *emu_chi2_out    = 1e30;
    *cur_loglike_out = -1e30;

    /* What mode did the LAST step use? 0=CAMB, 1=emulator */
    int prev_mode = (task[EMU_USED_POS] > 0.5) ? 1 : 0;

    /* Emulator only predicts slow params */
    double cosmo_A[N_SLOW], cosmo_B[N_SLOW];
    for (int k = 0; k < N_SLOW; k++) {
        cosmo_A[k] = theta_A[slow_idx[k]];
        cosmo_B[k] = theta_B[slow_idx[k]];
    }

    /* ------------------------------------------------------------------
     * Emulator prediction at both θ_A and θ_B.
     * Both must be OK and below threshold for use_emu=1.
     * ------------------------------------------------------------------ */
    if (USE_EMULATOR && g_emulator && emulator_is_ready(g_emulator)
        && g_emu_quarantine_left == 0) {
        int okA = emulator_predict(g_emulator, cosmo_A,
                                   emu_cl_tt_A, emu_cl_te_A, emu_cl_ee_A, emu_cl_bb_A,
                                   &uncertainty_A);

        EmulatorDiag diag_B;
        EmulatorDiag *diag_ptr = getenv("EMU_DIAG") ? &diag_B : NULL;

        int okB = emulator_predict_diag(g_emulator, cosmo_B,
                                        emu_cl_tt_B, emu_cl_te_B, emu_cl_ee_B, emu_cl_bb_B,
                                        &uncertainty_B, diag_ptr);

        /* Optional per-step diagnostic (set EMU_DIAG=1) */
        if (diag_ptr && okB) {
            int step = (int)task[STEP_POS];
            if (step % 5 == 0) {
                fprintf(stderr,
                    "[DIAG] rank=%d step=%d "
                    "unc_old=%.4f unc_abs=%.4f abs_log10=%.5f y_std=%.5f "
                    "sqrt_rv_w=%.4f sqrt_rv_max=%.4f "
                    "dnorm=%.3f/%.3f/%.3f/%.3f "
                    "whitened=%d modes=%d/%d/%d/%d\n",
                    rank, step,
                    diag_B.uncertainty, diag_B.uncertainty_abs,
                    diag_B.abs_log10_error, diag_B.signal_log10,
                    diag_B.sqrt_rv_weighted, diag_B.sqrt_rv_max,
                    diag_B.dnorm_tt, diag_B.dnorm_ee,
                    diag_B.dnorm_bb, diag_B.dnorm_rho,
                    diag_B.whitened,
                    diag_B.n_modes_tt, diag_B.n_modes_ee,
                    diag_B.n_modes_bb, diag_B.n_modes_rho);
            }
        }

        emu_use_A = okA && (uncertainty_A <= FALLBACK_THRESHOLD);
        emu_use_B = okB && (uncertainty_B <= FALLBACK_THRESHOLD);
    }

    int use_emu = (emu_use_A && emu_use_B) ? 1 : 0;

    /* ==================================================================
     * NON-DRAG PATH — plain Metropolis-Hastings
     * ================================================================== */
    if (!use_dragging) {
        int really_inv = (int)task[MULTIPURPOSEPOS];
        double cur_chi2 = 1e30;

        if (use_emu) {
            /* χ² at proposal from emulator Cls */
            proposal_chi2 = calc_likelihood(rank, theta_B, emu_cl_tt_B, emu_cl_te_B, emu_cl_ee_B, emu_cl_bb_B, 0, NULL, NULL)
                            + compute_prior_penalty(theta_B);
            *emu_chi2_out = proposal_chi2;

            /* χ² at current state (cached or recomputed) */
            if (prev_mode == 1 && really_inv <= 10) {
                double stored = task[EMU_LOGLIKE_POS];
                if (isfinite(stored) && stored < 1e29) {
                    cur_chi2 = -2.0 * stored;
                } else {
                    cur_chi2 = calc_likelihood(rank, theta_A, emu_cl_tt_A, emu_cl_te_A, emu_cl_ee_A, emu_cl_bb_A, 0, NULL, NULL)
                               + compute_prior_penalty(theta_A);
                }
            } else {
                cur_chi2 = calc_likelihood(rank, theta_A, emu_cl_tt_A, emu_cl_te_A, emu_cl_ee_A, emu_cl_bb_A, 0, NULL, NULL)
                           + compute_prior_penalty(theta_A);
            }
        } else {
            /* Fallback: CAMB at θ_B */
            int okB = param_iface(rank, theta_B, Cl_TT_B, Cl_TE_B, Cl_EE_B, Cl_BB_B);
            if (okB) {
                double dummy = 0.0;
                proposal_chi2 = calc_likelihood(rank, theta_B, Cl_TT_B, Cl_TE_B, Cl_EE_B, Cl_BB_B, 0, &dummy, NULL)
                                + compute_prior_penalty(theta_B);
            }

            /* CAMB at θ_A (cached or recomputed) */
            if (prev_mode == 0 && really_inv <= 10) {
                double stored = task[CAMB_LOGLIKE_POS];
                if (isfinite(stored) && stored < 1e29) {
                    cur_chi2 = -2.0 * stored;
                } else {
                    int okA = param_iface(rank, theta_A, Cl_TT_A, Cl_TE_A, Cl_EE_A, Cl_BB_A);
                    if (okA) {
                        double dummy = 0.0;
                        cur_chi2 = calc_likelihood(rank, theta_A, Cl_TT_A, Cl_TE_A, Cl_EE_A, Cl_BB_A, 0, &dummy, NULL)
                                   + compute_prior_penalty(theta_A);
                    }
                }
            } else {
                int okA = param_iface(rank, theta_A, Cl_TT_A, Cl_TE_A, Cl_EE_A, Cl_BB_A);
                if (okA) {
                    double dummy = 0.0;
                    cur_chi2 = calc_likelihood(rank, theta_A, Cl_TT_A, Cl_TE_A, Cl_EE_A, Cl_BB_A, 0, &dummy, NULL)
                               + compute_prior_penalty(theta_A);
                }
            }
        }

        task[PARAMETERS] = proposal_chi2;
        task[LOGLIKEPOS] = (proposal_chi2 < 1e29) ? -0.5 * proposal_chi2 : -1e30;
        *use_emu_out     = use_emu;
        *uncertainty_out = use_emu ? uncertainty_A : (uncertainty_A > uncertainty_B ? uncertainty_A : uncertainty_B);
        *cur_loglike_out = (cur_chi2 < 1e29) ? -0.5 * cur_chi2 : -1e30;
        return;
    }

    /* ==================================================================
     * DRAG PATH — path sampling with N_DRAG intermediate points
     *
     * At each point d/N_DRAG along the θ_A → θ_B segment:
     *   1. Interpolate slow params, jitter fast params
     *   2. Compute χ²_A(d) at slow_A + trial_fast and χ²_B(d) at slow_B + trial_fast
     *      (using drag-optimized calc_likelihood_drag, which reuses cached backgrounds)
     *   3. MH step on (1−w)·χ²_A + w·χ²_B
     *
     * After the path, the total acceptance ratio is exp(−mean Δχ²/2).
     * ================================================================== */
    double chi2_A_state, chi2_B_state;
    double ind_chi2_A = 0.0, ind_chi2_B = 0.0;

    if (use_emu) {
        chi2_A_state = calc_likelihood(rank, theta_A, emu_cl_tt_A, emu_cl_te_A, emu_cl_ee_A, emu_cl_bb_A, 0, NULL, &ind_chi2_A) + compute_prior_penalty(theta_A);
        chi2_B_state = calc_likelihood(rank, theta_B, emu_cl_tt_B, emu_cl_te_B, emu_cl_ee_B, emu_cl_bb_B, 0, NULL, &ind_chi2_B) + compute_prior_penalty(theta_B);
    } else {
        param_iface(rank, theta_A, Cl_TT_A, Cl_TE_A, Cl_EE_A, Cl_BB_A);
        param_iface(rank, theta_B, Cl_TT_B, Cl_TE_B, Cl_EE_B, Cl_BB_B);
        chi2_A_state = calc_likelihood(rank, theta_A, Cl_TT_A, Cl_TE_A, Cl_EE_A, Cl_BB_A, 0, NULL, &ind_chi2_A) + compute_prior_penalty(theta_A);
        chi2_B_state = calc_likelihood(rank, theta_B, Cl_TT_B, Cl_TE_B, Cl_EE_B, Cl_BB_B, 0, NULL, &ind_chi2_B) + compute_prior_penalty(theta_B);
    }
    proposal_chi2 = chi2_B_state;

    /* Choose Cl source for the drag loop */
    double *cl_A_tt = use_emu ? emu_cl_tt_A : Cl_TT_A;
    double *cl_A_te = use_emu ? emu_cl_te_A : Cl_TE_A;
    double *cl_A_ee = use_emu ? emu_cl_ee_A : Cl_EE_A;
    double *cl_A_bb = use_emu ? emu_cl_bb_A : Cl_BB_A;
    double *cl_B_tt = use_emu ? emu_cl_tt_B : Cl_TT_B;
    double *cl_B_te = use_emu ? emu_cl_te_B : Cl_TE_B;
    double *cl_B_ee = use_emu ? emu_cl_ee_B : Cl_EE_B;
    double *cl_B_bb = use_emu ? emu_cl_bb_B : Cl_BB_B;

    if (chi2_A_state < 1e29 && chi2_B_state < 1e29) {
        double chi2_A_original = chi2_A_state;
        double work_sum = chi2_B_state - chi2_A_state;
        double theta_f_path[MAX_TASKARRAY_SIZE] = {0};
        for (int k = 0; k < PARAMETERS; k++) theta_f_path[k] = theta_A[k];

        /* Precompute CAMB backgrounds at endpoints (reused at every path point) */
        int num_z = 500;
        double z_grid[500], DM_A[500], DM_B[500];
        int bg_A_ok = 1, bg_B_ok = 1;
        if (global_config.use_sn) {
            double H[500], DH[500], rd, zdrag;
            for (int i = 0; i < num_z; i++) z_grid[i] = i * 3.0 / (num_z - 1);
            if (!param_iface_bg(theta_A, num_z, z_grid, &rd, &zdrag, H, DM_A, DH)) bg_A_ok = 0;
            if (!param_iface_bg(theta_B, num_z, z_grid, &rd, &zdrag, H, DM_B, DH)) bg_B_ok = 0;
        }

        for (int d = 1; d <= N_DRAG; d++) {
            double w = (double)d / (double)N_DRAG;

            /* Build trial: interpolate slow, start fast from current path */
            double trial[MAX_TASKARRAY_SIZE] = {0};
            for (int k = 0; k < N_SLOW; k++) {
                int idx = slow_idx[k];
                trial[idx] = (1.0 - w) * theta_A[idx] + w * theta_B[idx];
            }
            for (int k = 0; k < N_FAST; k++) {
                int idx = fast_idx[k];
                trial[idx] = theta_f_path[idx];
            }

            /* Jitter fast params using cached fast covariance */
            int fast_oob = 0;
            if (fast_cov_ready && N_FAST > 0) {
                double jumps_f[MAX_FAST];
                for (int k = 0; k < N_FAST; k++) {
                    double stddev = (fast_eval[k] > 0) ? sqrt(fast_eval[k]) : 0.0;
                    jumps_f[k] = gasdev(0.0, stddev * fast_EntireFactor_local);
                }
                for (int k = 0; k < N_FAST; k++) {
                    int idx = fast_idx[k];
                    double delta = 0.0;
                    for (int j = 0; j < N_FAST; j++) delta += fast_evec[k][j] * jumps_f[j];
                    trial[idx] = theta_f_path[idx] + delta;
                    if (trial[idx] < g_lowbound[idx] || trial[idx] > g_highbound[idx]) fast_oob = 1;
                }
            }

            /* At this path point, χ²_A is evaluated with trial fast params and slow_A;
             * χ²_B with trial fast and slow_B. This is the drag trick: reuse
             * Cls at A and B, only the fast nuisance part of the likelihood
             * changes per path point (via calc_likelihood_drag). */
            double chi2_A_trial = 1e30, chi2_B_trial = 1e30;
            if (!fast_oob) {
                double thA[MAX_TASKARRAY_SIZE] = {0}, thB[MAX_TASKARRAY_SIZE] = {0};
                for (int k = 0; k < PARAMETERS; k++) { thA[k] = trial[k]; thB[k] = trial[k]; }
                for (int k = 0; k < N_SLOW; k++) {
                    thA[slow_idx[k]] = theta_A[slow_idx[k]];
                    thB[slow_idx[k]] = theta_B[slow_idx[k]];
                }

                if (bg_A_ok) {
                    chi2_A_trial = calc_likelihood_drag(rank, thA, cl_A_tt, cl_A_te, cl_A_ee, cl_A_bb, num_z, z_grid, DM_A, ind_chi2_A) + compute_prior_penalty(thA);
                }
                if (bg_B_ok) {
                    chi2_B_trial = calc_likelihood_drag(rank, thB, cl_B_tt, cl_B_te, cl_B_ee, cl_B_bb, num_z, z_grid, DM_B, ind_chi2_B) + compute_prior_penalty(thB);
                }
            }

            /* Interpolated likelihood between endpoints */
            double L_cur   = (1.0 - w) * chi2_A_state + w * chi2_B_state;
            double L_trial = (1.0 - w) * chi2_A_trial + w * chi2_B_trial;

            if (!fast_oob && chi2_A_trial < 1e29 && chi2_B_trial < 1e29 &&
                posRnd(1.0) < exp(-0.5 * (L_trial - L_cur))) {
                for (int k = 0; k < PARAMETERS; k++) theta_f_path[k] = trial[k];
                chi2_A_state = chi2_A_trial;
                chi2_B_state = chi2_B_trial;
            }
            if (d < N_DRAG) work_sum += (chi2_B_state - chi2_A_state);
        }

        /* Final acceptance ratio: average Δχ² over path */
        double ln_alpha = -0.5 * work_sum / (double)N_DRAG;
        double alpha = (ln_alpha >= 0.0) ? 1.0 : exp(ln_alpha);

        if (posRnd(1.0) < alpha) {
            for (int k = 0; k < PARAMETERS; k++) task[k] = theta_f_path[k];
            for (int k = 0; k < N_SLOW; k++) task[slow_idx[k]] = theta_B[slow_idx[k]];
            final_chi2    = chi2_B_state;
            proposal_chi2 = chi2_B_state;
            task[TAKEPOS] = 1.0;
        } else {
            for (int k = 0; k < PARAMETERS; k++) task[k] = theta_A[k];
            final_chi2    = chi2_A_original;
            proposal_chi2 = chi2_A_original;
            task[TAKEPOS] = 0.0;
        }
    } else {
        /* Endpoint χ² invalid — reject */
        for (int k = 0; k < PARAMETERS; k++) task[k] = theta_A[k];
        final_chi2 = 1e30; proposal_chi2 = 1e30;
        task[TAKEPOS] = 0.0;
    }

    task[PARAMETERS] = proposal_chi2;
    task[LOGLIKEPOS] = (final_chi2 < 1e29) ? -0.5 * final_chi2 : -1e30;

    /* If emulator was used AND drag accepted, save emu χ² for bias tracking */
    if (use_emu && task[TAKEPOS] == 1.0) {
        *emu_chi2_out = calc_likelihood(rank, task, emu_cl_tt_B, emu_cl_te_B, emu_cl_ee_B, emu_cl_bb_B, 0, NULL, NULL)
                        + compute_prior_penalty(task);
    }

    *use_emu_out     = use_emu;
    *uncertainty_out = uncertainty_A;
}

/* ============================================================================
 *  SLAVE — evaluator (runs on ranks 1..CHAIN*SLAVEPARCHAIN)
 *
 *  LOOP
 *    1. Check for TAG_STOP / TAG_EMU_QUARANTINE / TAG_UPDATE_COV
 *    2. Send GIVEMETASK (asking master for a proposal)
 *    3. Receive TAKETASK (the actual proposal)
 *    4. evaluate_proposal() — CAMB or emulator, with fallback
 *    5. Apply MH (single-step) or use drag decision
 *    6. Update emulator buffer (accepted emu steps: CAMB re-verify;
 *       accepted CAMB steps: direct; "free rejects" with small Δχ²)
 *    7. Send TAKERESULT back to master
 *
 *  Slave keeps no chain state — the master is the only owner.
 * ============================================================================ */
int slave(int rank) {
    DEBUG_PRINT("\nI am from slave %d", rank);
    MPI_Status status;
    double task[MAX_TASKARRAY_SIZE];

    int l_max_alloc = 3000;
    double *Cl_TT_A = (double*)malloc((l_max_alloc+1)*sizeof(double));
    double *Cl_TE_A = (double*)malloc((l_max_alloc+1)*sizeof(double));
    double *Cl_EE_A = (double*)malloc((l_max_alloc+1)*sizeof(double));
    double *Cl_BB_A = (double*)malloc((l_max_alloc+1)*sizeof(double));
    double *Cl_TT_B = (double*)malloc((l_max_alloc+1)*sizeof(double));
    double *Cl_TE_B = (double*)malloc((l_max_alloc+1)*sizeof(double));
    double *Cl_EE_B = (double*)malloc((l_max_alloc+1)*sizeof(double));
    double *Cl_BB_B = (double*)malloc((l_max_alloc+1)*sizeof(double));

    double g_lowbound[NOPARAM], g_highbound[NOPARAM], g_sigma[NOPARAM];
    get_parameter_bounds_from_config(g_lowbound, g_highbound, g_sigma);

    static int slave_n_acc = 0;
    static int slave_n_rej = 0;

    if (!mapping_done) {
        g_n_cosmo = N_SLOW; if (g_n_cosmo == 0) g_n_cosmo = 6; mapping_done = 1;
    }
    if (USE_EMULATOR && g_emulator == NULL) {
        EmulatorConfig emu_cfg = {
            .n_params = g_n_cosmo, .buffer_capacity = 500, .max_pca_modes = 40,
            .pca_interval = 200, .gp_interval = 50, .min_train_points = 200,
            .use_emulator_after = 250, .model_dir = "models",
            .emulator_type = EMULATOR_TYPE, .poly_degree = POLY_DEGREE
        };
        g_emulator = emulator_init(&emu_cfg);
        if (!g_emulator) fprintf(stderr, "Rank %d: Emulator init failed. Running without emulator.\n", rank);
        else DEBUG_PRINT("Rank %d: Emulator initialised (type %s) with %d cosmological parameters.\n", rank, (EMULATOR_TYPE == 0) ? "GP" : "Polynomial", g_n_cosmo);
    }

    if (VERBOSE) {
        printf("\n"
               "------------------------------------------------------------------------------------------------------------------------------\n"
               " Rank  MCMC_step  Emu  Fallback  Uncertainty    chi2      logL       Omega_m    Omega_b       h        tau      n_s       A_s\n"
               "------------------------------------------------------------------------------------------------------------------------------\n");
        fflush(stdout);
    }

    for (;;) {
        /* Poll for side-channel messages from master before asking for work */
        MPI_Status probe_status; int flag = 0;
        MPI_Iprobe(0, TAG_STOP, MPI_COMM_WORLD, &flag, &probe_status);
        if (flag) { MPI_Recv(NULL, 0, MPI_INT, 0, TAG_STOP, MPI_COMM_WORLD, &probe_status); goto slave_cleanup; }

        int qflag = 0; MPI_Status qst;
        MPI_Iprobe(0, TAG_EMU_QUARANTINE, MPI_COMM_WORLD, &qflag, &qst);
        if (qflag) {
            int sig = 0; MPI_Recv(&sig, 1, MPI_INT, 0, TAG_EMU_QUARANTINE, MPI_COMM_WORLD, &qst);
            if (sig == 1) { g_emu_quarantine_left = N_RECOVERY; DEBUG_PRINT("Rank %d: Emulator quarantined for %d steps\n", rank, N_RECOVERY); }
        }

        /* Ask master for a proposal */
        MPI_Send(task, TASKARRAY_SIZE, MPI_DOUBLE, 0, GIVEMETASK, MPI_COMM_WORLD);

        /* Wait for TAKETASK, handling any intervening broadcast */
        for (;;) {
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_STOP) {
                MPI_Recv(NULL, 0, MPI_INT, 0, TAG_STOP, MPI_COMM_WORLD, &status);
                goto slave_cleanup;
            }
            if (status.MPI_TAG == TAG_UPDATE_COV) {
                MPI_Recv(fast_cov_buffer, MAX_FAST_PACKED, MPI_DOUBLE, 0, TAG_UPDATE_COV, MPI_COMM_WORLD, &status);
                update_local_fast_covariance(fast_cov_buffer); continue;
            }
            if (status.MPI_TAG == TAG_EMU_QUARANTINE) {
                int sig = 0; MPI_Recv(&sig, 1, MPI_INT, 0, TAG_EMU_QUARANTINE, MPI_COMM_WORLD, &status);
                if (sig == 1) { g_emu_quarantine_left = N_RECOVERY; DEBUG_PRINT("Rank %d: Emulator quarantined for %d steps (inner)\n", rank, N_RECOVERY); }
                continue;
            }
            break;
        }
        MPI_Recv(task, TASKARRAY_SIZE, MPI_DOUBLE, 0, TAKETASK, MPI_COMM_WORLD, &status);

        int use_dragging = (int)task[DRAGPOS];
        int use_emu = 0; double uncertainty = 0.0; double emu_chi2_used = 1e30; double cur_loglike = -1e30;

        evaluate_proposal(rank, task,
                          Cl_TT_A, Cl_TE_A, Cl_EE_A, Cl_BB_A,
                          Cl_TT_B, Cl_TE_B, Cl_EE_B, Cl_BB_B,
                          g_lowbound, g_highbound, g_sigma,
                          &use_emu, &uncertainty,
                          &emu_chi2_used, &cur_loglike);

        double proposal_chi2_raw = task[PARAMETERS];
        double theta_B_cap[MAX_TASKARRAY_SIZE];
        for (int k = 0; k < PARAMETERS; k++) theta_B_cap[k] = task[k];

        /* MH decision: drag path already decided, else single-step MH here.
         * NOTE: master draws PROBPOS once and packs it; slave uses that
         * specific U(0,1) draw for the accept/reject decision. */
        int take;
        if (use_dragging) {
            take = (int)task[TAKEPOS];
        } else {
            double loglike_prop = task[LOGLIKEPOS];
            double alpha = mind(1.0, exp(loglike_prop - cur_loglike));
            take = (posRnd(1.0) < alpha) ? 1 : 0;
            if (take == 0) {
                for (int k = 0; k < PARAMETERS; k++) task[k] = task[CURRENTSTATEPOS + k];
                task[LOGLIKEPOS] = cur_loglike;
                task[PARAMETERS] = -2.0 * cur_loglike;
            }
        }

        task[TAKEPOS] = (double)take;
        task[PROBPOS] = (take == 1) ? 1.0 : 0.0;
        if (take && g_emu_quarantine_left > 0) g_emu_quarantine_left--;
        task[EMU_USED_POS] = (double)use_emu;
        task[EMU_BIAS_POS] = 0.0;

        /* Cache logL by mode — master reads this for next-step χ²_A */
        if (take) {
            slave_n_acc++;
            if (use_emu) { task[CAMB_LOGLIKE_POS] = 1e30; task[EMU_LOGLIKE_POS]  = task[LOGLIKEPOS]; }
            else         { task[CAMB_LOGLIKE_POS] = task[LOGLIKEPOS]; task[EMU_LOGLIKE_POS]  = 1e30; }
        }

        /* Emulator training buffer updates */
        if (take && g_emulator) {
            if (!use_emu) {
                /* CAMB accepted → direct training add */
                double *tt_c = (double*)malloc(2601 * sizeof(double));
                double *te_c = (double*)malloc(2601 * sizeof(double));
                double *ee_c = (double*)malloc(2601 * sizeof(double));
                double *bb_c = (double*)malloc(2601 * sizeof(double));
                if (tt_c && te_c && ee_c && bb_c) {
                    memcpy(tt_c, Cl_TT_B, 2601 * sizeof(double));
                    memcpy(te_c, Cl_TE_B, 2601 * sizeof(double));
                    memcpy(ee_c, Cl_EE_B, 2601 * sizeof(double));
                    memcpy(bb_c, Cl_BB_B, 2601 * sizeof(double));

                    double cosmo_final[N_SLOW];
                    for (int k = 0; k < N_SLOW; k++) cosmo_final[k] = task[slow_idx[k]];

                    if (!emulator_update(g_emulator, local_step++, cosmo_final, tt_c, te_c, ee_c, bb_c, task[LOGLIKEPOS])) {
                        free(tt_c); free(te_c); free(ee_c); free(bb_c);
                    }
                } else { free(tt_c); free(te_c); free(ee_c); free(bb_c); }
            } else if (!TRAIN_ON_FALLBACK_ONLY) {
                /* Emulator accepted → re-evaluate with CAMB for (a) verification
                 * (b) training data. This is the only way the emulator learns
                 * from its own trajectory. */
                double *Cl_TT_true = (double*)malloc(2601 * sizeof(double));
                double *Cl_TE_true = (double*)malloc(2601 * sizeof(double));
                double *Cl_EE_true = (double*)malloc(2601 * sizeof(double));
                double *Cl_BB_true = (double*)malloc(2601 * sizeof(double));

                if (Cl_TT_true && Cl_TE_true && Cl_EE_true && Cl_BB_true) {
                    if (param_iface(rank, task, Cl_TT_true, Cl_TE_true, Cl_EE_true, Cl_BB_true)) {
                        double dummy_true = 0.0;
                        double true_chi2 = calc_likelihood(rank, task, Cl_TT_true, Cl_TE_true, Cl_EE_true, Cl_BB_true, 0, &dummy_true, NULL)
                                           + compute_prior_penalty(task);
                        double true_logL = -0.5 * true_chi2;

                        task[CAMB_LOGLIKE_POS] = true_logL;
                        if (use_emu && emu_chi2_used < 1e29) task[EMU_BIAS_POS] = fabs(true_chi2 - emu_chi2_used);

                        double cosmo_final[N_SLOW];
                        for (int k = 0; k < N_SLOW; k++) cosmo_final[k] = task[slow_idx[k]];

                        if (!emulator_update(g_emulator, local_step++, cosmo_final, Cl_TT_true, Cl_TE_true, Cl_EE_true, Cl_BB_true, true_logL)) {
                            free(Cl_TT_true); free(Cl_TE_true); free(Cl_EE_true); free(Cl_BB_true);
                        }
                    } else { free(Cl_TT_true); free(Cl_TE_true); free(Cl_EE_true); free(Cl_BB_true); }
                } else { free(Cl_TT_true); free(Cl_TE_true); free(Cl_EE_true); free(Cl_BB_true); }
            }
        }
        else if (!take && !use_dragging && !use_emu && g_emulator) {
            /* Free-reject training: rejected CAMB proposals with small Δχ²
             * are still useful data. Rate-limited to 2× accept rate. */
            double cur_chi2_cap = -2.0 * cur_loglike;
            double delta_chi2   = proposal_chi2_raw - cur_chi2_cap;

            if (delta_chi2 >= 0.0 && delta_chi2 < FREE_REJECT_DCHI2_MAX && slave_n_rej < FREE_REJECT_RATIO * slave_n_acc) {
                double *tt_c = (double*)malloc(2601 * sizeof(double));
                double *te_c = (double*)malloc(2601 * sizeof(double));
                double *ee_c = (double*)malloc(2601 * sizeof(double));
                double *bb_c = (double*)malloc(2601 * sizeof(double));
                if (tt_c && te_c && ee_c && bb_c) {
                    memcpy(tt_c, Cl_TT_B, 2601 * sizeof(double));
                    memcpy(te_c, Cl_TE_B, 2601 * sizeof(double));
                    memcpy(ee_c, Cl_EE_B, 2601 * sizeof(double));
                    memcpy(bb_c, Cl_BB_B, 2601 * sizeof(double));

                    double cosmo_b[N_SLOW];
                    for (int k = 0; k < N_SLOW; k++) cosmo_b[k] = theta_B_cap[slow_idx[k]];

                    if (!emulator_update(g_emulator, local_step++, cosmo_b, tt_c, te_c, ee_c, bb_c, -0.5 * proposal_chi2_raw)) {
                        free(tt_c); free(te_c); free(ee_c); free(bb_c);
                    } else { slave_n_rej++; }
                } else { free(tt_c); free(te_c); free(ee_c); free(bb_c); }
            }
        }

        /* Force-train if buffer has accumulated enough */
        if (g_emulator && !emulator_is_ready(g_emulator) && emulator_buffer_size(g_emulator) >= 200) {
            DEBUG_PRINT("Rank %d: Forcing emulator training.\n", rank); emulator_train(g_emulator);
        }

        if (VERBOSE) {
            int mcmc_step = (int)task[STEP_POS];
            double chi2_to_print = -2.0 * task[LOGLIKEPOS];
            double logL_to_print = task[LOGLIKEPOS];
            int max_print = (g_n_cosmo < 6) ? g_n_cosmo : 6;
            printf(" %2d     %6d    %1d      %1d     %8.4e   %8.2f %8.2f   ", rank, mcmc_step, use_emu, (!use_emu), uncertainty, chi2_to_print, logL_to_print);
            for (int i = 0; i < max_print; i++) printf("%8.4f ", task[i]);
            for (int i = max_print; i < 6; i++) printf("%8s ", " ");
            printf("\n"); fflush(stdout);
        }

        MPI_Send(task, TASKARRAY_SIZE, MPI_DOUBLE, 0, TAKERESULT, MPI_COMM_WORLD);
    }

slave_cleanup: ;
    free(Cl_TT_A); free(Cl_TE_A); free(Cl_EE_A); free(Cl_BB_A);
    free(Cl_TT_B); free(Cl_TE_B); free(Cl_EE_B); free(Cl_BB_B);
    return 0;
}

/* ============================================================================
 *  slave0 — evaluates the initial 100-sample scan
 *
 *  master0 sends `runperchain` starting points. Each slave evaluates χ²
 *  (CAMB + CLIK + prior) and sends back with the χ² appended at
 *  task[PARAMETERS]. This is the only step where slave0 is used.
 * ============================================================================ */
int slave0(int rank,int runperchain) {
  MPI_Status  status; double *task; double Chi2X; int magicnum1;
  int l_max_alloc = 3000;
  double *Cl_TT = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  double *Cl_TE = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  double *Cl_EE = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  double *Cl_BB = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  if (!Cl_TT || !Cl_TE || !Cl_EE || !Cl_BB) { fprintf(stderr, "Rank %d: Failed to allocate memory.\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
  task = new_double(PARAMETERS+1);
  DEBUG_PRINT("My rank is here %d",rank);
  for (int i=0;i<runperchain;i++) {
      MPI_Recv(task,PARAMETERS+1,MPI_DOUBLE,0,200,MPI_COMM_WORLD,&status);
      magicnum1 = param_iface(rank, task, Cl_TT, Cl_TE, Cl_EE, Cl_BB);
      if (magicnum1 != 0) {
          double dummy = 0.0;
          Chi2X = calc_likelihood(rank, task, Cl_TT, Cl_TE, Cl_EE, Cl_BB, 0, &dummy, NULL) + compute_prior_penalty(task);
      } else { Chi2X = 10.0e10; }   /* CAMB failed — reject */
      if (WMAP7 == 1) task[PARAMETERS] = Chi2X;
      for (unsigned int k = 0;k < PARAMETERS+1; k++) {
          if (isnan(task[k])) { fprintf(stderr, "isnan element\n"); free(Cl_TT); free(Cl_TE); free(Cl_EE); free(Cl_BB); MPI_Abort(MPI_COMM_WORLD, 1); }
      }
      MPI_Send(task,PARAMETERS+1, MPI_DOUBLE,0,201,MPI_COMM_WORLD);
  }
  free(Cl_TT); free(Cl_TE); free(Cl_EE); free(Cl_BB);
  DEBUG_PRINT("\nI am now working fine.."); return 0;
}

/* ============================================================================
 *  master0 — starting-point scan
 *
 *  Draws 100 points, evaluates χ², returns the CHAINS best as starting
 *  points for the chains. Sub-samples from the prior (Gaussian prior uses
 *  gasdev centered at mean; flat uses midpoint + spread).
 * ============================================================================ */
double **master0(int worldsize, int runperchain) {
  MPI_Status status; double *startval[100];
  for (int i = 0; i < 100; i++) startval[i] = new_double(PARAMETERS + 1);
  double *lowbound = new_double(PARAMETERS); double *highbound = new_double(PARAMETERS);
  double *initial_sigma_dummy = new_double(PARAMETERS);
  get_parameter_bounds_from_config(lowbound, highbound, initial_sigma_dummy);

  /* Draw 100 points, respecting priors */
  for (int i = 0; i < 100; i++) {
    int param_idx = 0;
    for (int p = 0; p < global_config.param_count && param_idx < PARAMETERS; p++) {
      if (global_config.params[p].state == PARAM_SAMPLED) {
        double val;
        if (global_config.params[p].prior_type == PRIOR_GAUSSIAN) val = gasdev(global_config.params[p].gaussian_mean, global_config.params[p].gaussian_sigma);
        else { double center = (highbound[param_idx] + lowbound[param_idx]) / 2.0; val = center + gasdev(0.0, initial_sigma_dummy[param_idx] * 2.0); }
        if (val < lowbound[param_idx]) val = lowbound[param_idx];
        if (val > highbound[param_idx]) val = highbound[param_idx];
        startval[i][param_idx] = val; param_idx++;
      }
    }
    for (; param_idx < PARAMETERS; param_idx++) startval[i][param_idx] = lowbound[param_idx] + posRnd(highbound[param_idx] - lowbound[param_idx]);
    startval[i][PARAMETERS] = 0.0;
  }
  free(initial_sigma_dummy);

  /* Distribute to slaves, collect χ² */
  double *startvaltemp = new_double(PARAMETERS + 1);
  int n_slaves = worldsize - 1;
  for (int i = 0; i < runperchain; i++) {
    for (int j = 0; j < n_slaves; j++) MPI_Send(startval[i + j * runperchain], PARAMETERS + 1, MPI_DOUBLE, j + 1, 200, MPI_COMM_WORLD);
    for (int j = 0; j < n_slaves; j++) {
      MPI_Recv(startvaltemp, PARAMETERS + 1, MPI_DOUBLE, MPI_ANY_SOURCE, 201, MPI_COMM_WORLD, &status);
      for (int k = 0; k <= PARAMETERS; k++) startval[i + (status.MPI_SOURCE - 1) * runperchain][k] = startvaltemp[k];
    }
  }

  /* Sort by χ² ascending */
  int temparrange[100]; double temploglike[100];
  for (int i = 0; i < 100; i++) { temparrange[i] = i; temploglike[i] = startval[i][PARAMETERS]; }
  for (int i = 0; i < CHAINS; i++)
    for (int j = i; j < 100; j++) {
      if (temploglike[i] > temploglike[j]) {
        int tf = temparrange[i]; temparrange[i] = temparrange[j]; temparrange[j] = tf;
        double lf = temploglike[i]; temploglike[i] = temploglike[j]; temploglike[j] = lf;
      }
    }
  double **startvalfinal = (double **)malloc(CHAINS * sizeof(double *));
  for (int i = 0; i < CHAINS; i++) startvalfinal[i] = new_double(PARAMETERS + 1);
  for (int i = 0; i < CHAINS; i++) for (int j = 0; j <= PARAMETERS; j++) startvalfinal[i][j] = startval[temparrange[i]][j];
  return startvalfinal;
}

/* --------------------------------------------------------------------------
 *  classify_parameters — split sampled params into slow / fast.
 *  Must be called once after PARAMETERS is known and before any drag /
 *  emulator setup. Slow indices go to CAMB; fast indices never do.
 * -------------------------------------------------------------------------- */
void classify_parameters(void) {
    N_SLOW = 0; N_FAST = 0; int param_idx = 0;
    for (int i = 0; i < global_config.param_count && param_idx < PARAMETERS; i++) {
        if (global_config.params[i].state == PARAM_SAMPLED) {
            if (global_config.params[i].speed == PARAM_SPEED_SLOW) slow_idx[N_SLOW++] = param_idx;
            else fast_idx[N_FAST++] = param_idx;
            param_idx++;
        }
    }
}

/* --------------------------------------------------------------------------
 *  setVariables — populate global_config and compute ALL task-array positions
 *
 *  This is the single source of truth for the task array layout. Every
 *  index that appears elsewhere in the file (LOGLIKEPOS, TAKEPOS, ...)
 *  is computed here in a fixed order. See the header comment for the
 *  full layout.
 *
 *  LOGLIKEPOS is forced to at least 15 to leave room for extra diagnostic
 *  fields in future versions — the sampler doesn't care about exact
 *  positions, only that they don't collide.
 * -------------------------------------------------------------------------- */
void setVariables() {
    if (!load_config("param_reader.ini", &global_config)) { printf("CRITICAL ERROR: Could not load param_reader.ini\n"); exit(1); }
    DEBUG_PRINT("Loaded configuration from param_reader.ini.\n");
    int total_estimated = 0;
    for (int i = 0; i < global_config.param_count; i++) if (global_config.params[i].state == PARAM_SAMPLED) total_estimated++;
    PARAMETERS = (global_config.num_sampled_total > 0) ? global_config.num_sampled_total : NOPARAM;
    LOGLIKEPOS = PARAMETERS + 1;
    LOGLIKEPOS = (LOGLIKEPOS > 15) ? LOGLIKEPOS : 15;        /* reserve [0..14] for params+chisq */
    MULTIPURPOSEPOS = LOGLIKEPOS + 22;                        /* leave room for future fields */
    ADAPTIVEPOS = MULTIPURPOSEPOS + 1; DRAGPOS = ADAPTIVEPOS + 1;
    FASTFACTORPOS = DRAGPOS + 1; TAKEPOS = FASTFACTORPOS + 1;
    PROBPOS = TAKEPOS + 1; RANDPOS = PROBPOS + 1;
    EMU_BIAS_POS     = RANDPOS + PARAMETERS + MULTIPURPOSE_REQUIRED + 1;
    EMU_USED_POS     = EMU_BIAS_POS + 1; CAMB_LOGLIKE_POS = EMU_USED_POS + 1;
    EMU_LOGLIKE_POS  = CAMB_LOGLIKE_POS + 1; STEP_POS = EMU_LOGLIKE_POS + 1;
    CURRENTSTATEPOS  = STEP_POS + 1; TASKARRAY_SIZE = CURRENTSTATEPOS + PARAMETERS;
    if (TASKARRAY_SIZE > MAX_TASKARRAY_SIZE) { printf("ERROR: TASKARRAY_SIZE (%d) exceeds MAX_TASKARRAY_SIZE (%d).\n", TASKARRAY_SIZE, MAX_TASKARRAY_SIZE); exit(1); }
    classify_parameters();
    DEBUG_PRINT("Task Array Size: %d | Slow: %d | Fast: %d\n", TASKARRAY_SIZE, N_SLOW, N_FAST);
}

/* ============================================================================
 *  MAIN — initialization and dispatch
 *
 *  ORDER MATTERS
 *    1. MPI_Init, rank sizing
 *    2. setVariables → global_config loaded, task layout fixed
 *    3. CLI overrides (-emu, -drag, -o, -load-cov, ...)
 *    4. Broadcast outdir; disable drag if no fast params
 *    5. Symlink CAMB HighL template
 *    6. Init CLIK (once, all ranks) — fail → MPI_Abort
 *    7. Init SNe (master only? actually all — sn_chi2 needs it on slaves)
 *    8. master0 / slave0 — starting-point scan
 *    9. master / slave — main MCMC
 *   10. Cleanup and MPI_Finalize
 * ============================================================================ */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int myrank, worldsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
    srand((unsigned)(time(NULL) ^ (myrank * 2654435761u)));

    /* Step 1: load ini */
    setVariables();
    DEBUG_PRINT("Myrank : %d", myrank);

    const char *outdir = global_config.outdir[0] ? global_config.outdir : ".";
    const char *load_cov_path = global_config.load_cov_path[0] ? global_config.load_cov_path : NULL;
    unsigned short int Restart = global_config.restart;

    /* Step 2: CLI overrides */
    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "-restart") == 0)                Restart = 1;
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc)        outdir = argv[++i];
        else if (strcmp(argv[i], "-load-cov") == 0 && i+1 < argc) load_cov_path = argv[++i];
        else if (strcmp(argv[i], "-emu") == 0)                     global_config.use_emulator = 1;
        else if (strcmp(argv[i], "-no-emu") == 0)                  global_config.use_emulator = 0;
        else if (strcmp(argv[i], "-drag") == 0)                    global_config.use_dragging = 1;
        else if (strcmp(argv[i], "-no-drag") == 0)                 global_config.use_dragging = 0;
        else if (strcmp(argv[i], "-print") == 0)                   global_config.verbose = 1;
        else if (strcmp(argv[i], "-no-print") == 0)                global_config.verbose = 0;
        else if (strcmp(argv[i], "-train-fallback-only") == 0)     global_config.train_on_fallback_only = 1;
        else {
            if (myrank == 0)
                printf("Usage: %s [-restart] [-o outdir] [-load-cov file] "
                       "[-emu|-no-emu] [-drag|-no-drag] [-print|-no-print] "
                       "[-train-fallback-only]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    LOAD_COV_PATH = load_cov_path;

    /* Step 3: enforce rank topology (must match CHAINS × SLAVEPARCHAIN) */
    int expected_ranks = 1 + CHAINS * SLAVEPARCHAIN;
    if (worldsize != expected_ranks) {
        if (myrank == 0) fprintf(stderr, "ERROR: need exactly %d MPI ranks.\n", expected_ranks);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Step 4: create outdir (master only), broadcast */
    if (myrank == 0) {
        char mkdir_cmd[512]; snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", outdir); system(mkdir_cmd);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    strncpy(outdir_buf, outdir, sizeof(outdir_buf)-1); outdir_buf[sizeof(outdir_buf)-1] = '\0';
    MPI_Bcast(outdir_buf, sizeof(outdir_buf), MPI_CHAR, 0, MPI_COMM_WORLD);

    /* Step 5: auto-disable drag if no fast params */
    if (N_FAST == 0 && USE_DRAGGING) {
        if (myrank == 0) printf("\n[WARNING] N_FAST = 0 (No sampled fast parameters found).\n          Automatically disabling Drag mode.\n\n");
        USE_DRAGGING = 0;
    }

    /* Step 6: CAMB HighL template symlink (needed for lensed Cls) */
    if (strlen(global_config.camb_highl_template) > 0) {
        char cmd[512]; snprintf(cmd, sizeof(cmd), "ln -sf %s HighLExtrapTemplate_lenspotentialCls.dat", global_config.camb_highl_template);
        if (myrank == 0) system(cmd);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* Step 7: CLIK init — once, on every rank. Each rank has its own process
     * and its own CLIK cache. MPI_Abort if any rank fails. */
    error *clik_err = initError();
    if (!initialize_clik_objects(&clik_err)) {
        if (myrank == 0) fprintf(stderr, "[FATAL] Could not initialize CLIK objects.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    free(clik_err);

    /* Step 8: SNe init (per-rank, each has own copy of the data + Cholesky) */
    if (global_config.use_sn) {
        sn_resolve_param_indices();
    }
    print_and_validate_parameters(myrank);

    if (global_config.use_sn) {
        if (strlen(global_config.sn_data_path) > 0 && strlen(global_config.sn_cov_path) > 0) {
            sn_init(global_config.sn_data_path, global_config.sn_cov_path);
        } else { fprintf(stderr, "WARNING: USE_SN=1 but paths are empty in param_reader.ini.\n"); }
    } else {
        if (myrank == 0) printf("[INFO] SNe likelihood disabled via param_reader.ini (USE_SN=0).\n");
    }

    /* Step 9: Startup scan — draw 100 seeds, evaluate, keep CHAINS best */
    int totalrun, runperchain;
    drfactor = global_config.initial_drfactor;
    if (worldsize > 101) { totalrun = 100; runperchain = 1; }
    else                 { runperchain = 100 / (worldsize - 1); totalrun = runperchain * (worldsize - 1); }

    double **startnum; double start_time = 0.0;
    if (myrank == 0) start_time = get_time();

    if (myrank == 0) { DEBUG_PRINT("\nRestart = %d", Restart); startnum = master0(worldsize, runperchain); }
    else { slave0(myrank, runperchain); }

    /* Step 10: main MCMC */
    if (myrank == 0) { DEBUG_PRINT("\nRestart = %d", Restart); master(startnum, Restart); }
    else { DEBUG_PRINT("Slave is working"); slave(myrank); }

    /* Step 11: report and cleanup */
    if (myrank == 0) {
        double end_time = get_time(); double elapsed = end_time - start_time;
        int h = (int)(elapsed / 3600); int m = (int)((elapsed - h * 3600) / 60); int s = (int)(elapsed - h * 3600 - m * 60);
        printf("\n========================================\n");
        printf("Total elapsed time: %02d:%02d:%02d (hh:mm:ss)\n", h, m, s);
        printf("========================================\n");
    }

    cleanup_clik_objects();
    sn_free();
    if (g_emulator) emulator_free(g_emulator);
    MPI_Finalize();
    return 0;
}
