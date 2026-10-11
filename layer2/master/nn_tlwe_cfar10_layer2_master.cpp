
// ----------------------------------------------------------------------------|
// Title      : Fast Homomorphic Evaluation of Deep Discretized Neural Networks
// Project    : Showcase Fast Fully Homomorphic Evaluation of Encrypted Inputs
//              using Deep Discretized Neural Networks hence preserving Privacy
// ----------------------------------------------------------------------------|
// File       : nn_tlwe_cfar10_lyer2_masterM.cpp
// Authors    : Florian Bourse      <Florian.Bourse@ens.fr>
//              Michele Minelli     <Michele.Minelli@ens.fr>
//              Matthias Minihold   <Matthias.Minihold@RUB.de>
//              Pascal Paillier     <Pascal.Paillier@cryptoexperts.com>
//
// Reference  : TFHE: Fast Fully Homomorphic Encryption Library over the Torus
//              https://github.com/tfhe
// ----------------------------------------------------------------------------|
// Description:
//     LAYER 2 — Single-node master reference implementation.
//     Reads Layer 1 bootstrapped LWE ciphertexts (128 channels, 32x32) from
//     files written by the Layer 1 distributed nodes.
//     Performs Conv2D (3x3 kernel, 128 input ch -> 32 output ch).
//     Applies PBS / Sign activation via TFHE bootstrapping (OMP 4-thread).
//     Exports Layer 2 bootstrapped samples + index maps for downstream use.
// ----------------------------------------------------------------------------|
// Revisions  :
// Date        Version  Description
// 2017-11-16  0.3.0    Version for github, referenced by ePrint32_t paper
// (Layer 2 adaptation)
// ----------------------------------------------------------------------------|


// Includes
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <sys/time.h>
#include <stdexcept>
#include <omp.h>
#include <thread>
#include <mutex>
// tfhe-lib
#include "tfhe.h"
#include "tfhe_garbage_collector.h"
#include "tlwe.h"
#include "tlwe_functions.h"
#include "tgsw.h"
#include "lwesamples.h"
#include "lwekey.h"
#include "lweparams.h"
#include "polynomials.h"
// Multi-processing
#include <sys/wait.h>
#include <unistd.h>
// Handling
#include <sstream>
#include <ctime>
#include <signal.h>
#include <chrono>
volatile sig_atomic_t got_sigterm = 0;


// Defines
#define PROCESSES  1
#define VERBOSE    true
#define DEBUG      true
// -----------------------------------------------------------------------
// KEY GENERATION
//   M1569 (run once on pi3) is the single designated key generator for the
//   whole pipeline. This file already points at pi3's key paths, so as
//   long as M1569 has been run at least once, GENERATE=false here is
//   correct -- no copy needed since this also runs on pi3.
// -----------------------------------------------------------------------
#define GENERATE   false  // Layer 2 always reuses the keys generated in Layer 1
#define MODE_STATS true
#define MODE_LATEX true
#define MODE_BATCH true
#define MODE_NOISE true


// Security constants used for testing (same as Layer 1)
#define SEC_LEVEL            80
#define SEC_PARAMS_n         450
#define SEC_ALPHA            pow(2., -20)
#define SEC_PARAMS_STDDEV    pow(2., -30)
#define SEC_PARAMS_N         1024
#define SEC_PARAMS_k         1
#define SEC_PARAMS_BK_STDDEV pow(2., -36)
#define SEC_PARAMS_BK_BASEBITS  10
#define SEC_PARAMS_BK_LENGTH    3
#define SEC_PARAMS_KS_STDDEV pow(2., -25)
#define SEC_PARAMS_KS_BASEBITS  1
#define SEC_PARAMS_KS_LENGTH    18

#define NUM_NEURONS_INPUT  256
#define NUM_NEURONS_HIDDEN 30
#define NUM_NEURONS_OUTPUT 10
#define NUM_NEURON_LAYERS  3

#define CARD_TESTSET 10000
#define MSG_SLOTS    143

// -----------------------------------------------------------------------
// LAYER 2 TORUS SLOTS
//   Layer 1 outputs are bootstrapped into TORUS_SLOTS_L1 = 1569 slots.
//   Layer 2 MAC accumulates up to 128 inputs x 3x3 = 1152 values,
//   so we reuse the same slot count for L2 output bootstrapping.
// -----------------------------------------------------------------------
#define TORUS_SLOTS_L1  1569   // Space used by Layer 1 bootstrapped outputs (input to L2)
#define TORUS_SLOTS_L2  1569   // Space for Layer 2 bootstrapped outputs
#define TORUS_SLOTS2 270
#define TORUS_SLOTS3 170
#define TORUS_SLOTS4  30
#define TORUS_SLOTS22 11

// -----------------------------------------------------------------------
// LAYER 2 TOPOLOGY
//   num_channels_in  : 128  (Layer 1 output channels — all read as input)
//   num_channels_out : 32   (Layer 2 output channels to compute)
//   kernel_size      : 3    (3x3 spatial kernel)
//   pheight/pwidth   : 32   (spatial dimensions, same as Layer 1 output)
// -----------------------------------------------------------------------
#define L2_NUM_CHANNELS_IN   128
#define L2_NUM_CHANNELS_OUT  32
#define L2_KERNEL_SIZE       3

// File paths
#define FILE_TEST           "/home/soumik/fhe-dinn/tfhe/src/test/tests.txt"
#define FILE_TXT_IMG        "/home/soumik/fhe-dinn/tfhe/src/test/txt_img_test.txt"
#define FILE_TXT_BIASES     "/home/soumik/fhe-dinn/tfhe/src/test/txt_biases.txt"
#define FILE_TXT_WEIGHTS    "/home/soumik/fhe-dinn/tfhe/src/test/txt_weights.txt"
#define FILE_TXT_LABELS     "/home/soumik/fhe-dinn/tfhe/src/test/txt_labels.txt"
#define FILE_STATISTICS     "/home/soumik/fhe-dinn/tfhe/src/test/results_stats.txt"
#define FILE_LATEX          "/home/soumik/fhe-dinn/tfhe/src/test/results_LaTeX.tex"
// KEY: Layer 2 reuses the secret key written by Layer 1
#define FILE_USER_KEY       "/home/pi3/key_user_secret.bin"
#define FILE_CLOUD_KEY      "/home/pi3/tfhe/src/test/key_cloud.bin"
// Layer 1 bootstrapped LWE output files (written by L1 slave nodes)
// Pattern: /home/pi3/helloworld/image3_l1out_lwesample_{ch}_{ph}_{pw}_{output_i}
#define L1_OUTPUT_BASE_PATH "/home/pi3/helloworld/image3_l1out_lwesample_"
// Layer 2 bias and weight files
#define FILE_L2_BIAS        "/home/pi3/tfhe/build/layer2_bias.txt"
#define FILE_L2_WEIGHTS     "/home/pi3/tfhe/build/layer2_weights.txt"

#define THRESHOLD_WEIGHTS  9
#define THRESHOLD_SCORE -100


using namespace std;


// -----------------------------------------------------------------------
// Internal Function Declarations
// -----------------------------------------------------------------------
void deleteMatrix(int32_t**  matrix, int32_t dim_mat);
void deleteTensor(int32_t*** tensor, int32_t dim_mat, const int32_t* dim_vec);
void writeStatistics();
void my_signal_handler(int32_t sig);
TFheGateBootstrappingParameterSet *our_default_bootstrapping_parameters(int32_t minimum_lambda);
int32_t processNeuralNet();


// -----------------------------------------------------------------------
// Global Counters, Timings, Variables (carried from L1 structure)
// -----------------------------------------------------------------------
int32_t count_images;
int32_t count_errors;
int32_t count_errors_with_failed_bs;
int32_t count_disagreements;
int32_t count_disagreements_with_failed_bs;
int32_t count_disag_pro_clear;
int32_t count_disag_pro_hom;
int32_t count_wrong_bs;
double total_time_classifications, total_time_bootstrappings;
double avg_img;
double avg_img_bs;
double avg_time_per_classification;
double avg_time_per_bootstrapping;
double rel_error_percent;
double rel_time_bs_percent;


// -----------------------------------------------------------------------
// Global Constants
// -----------------------------------------------------------------------
const int32_t num_processes = PROCESSES;
const bool batched          = MODE_BATCH;

// Security
const int32_t minimum_lambda = SEC_LEVEL;
const double alpha = SEC_ALPHA;

static const int32_t n = SEC_PARAMS_n;
static const int32_t N = SEC_PARAMS_N;
static const int32_t k = SEC_PARAMS_k;
static const double max_stdev = SEC_PARAMS_STDDEV;

static const int32_t bk_Bgbit = SEC_PARAMS_BK_BASEBITS;
static const int32_t bk_l     = SEC_PARAMS_BK_LENGTH;
static const double bk_stdev  = SEC_PARAMS_BK_STDDEV;

static const int32_t ks_basebit = SEC_PARAMS_KS_BASEBITS;
static const int32_t ks_length  = SEC_PARAMS_KS_LENGTH;
static const double ks_stdev    = SEC_PARAMS_KS_STDDEV;

const bool noisyLWE = MODE_NOISE;
const int32_t num_images = CARD_TESTSET;
const int32_t total_num_hidden_neurons = CARD_TESTSET * NUM_NEURONS_HIDDEN;

// Torus slot spaces
const int32_t space_msg      = MSG_SLOTS;
const int32_t space_after_bs = TORUS_SLOTS_L1;  // L1 output bootstrap space (input to L2)
const int32_t space_after_l2 = TORUS_SLOTS_L2;  // L2 output bootstrap space


// -----------------------------------------------------------------------
// Helper: compute unique flat output index (same formula as L1 slave nodes)
//   Returns the global linear index for a spatial+channel position,
//   used to reconstruct file names of exported LWE samples.
//   For L1 outputs: OutDepth=128, lay_dim_hw_w=32
// -----------------------------------------------------------------------
uint64_t get_output_i_l1(uint32_t ph, uint32_t pw, uint32_t od)
{
    uint32_t OutDepth     = 128;  // L1 output channels
    uint32_t lay_dim_hw_w = 32;   // spatial width
    return (((ph) * lay_dim_hw_w + pw) * OutDepth + od);
}

// For L2 outputs: OutDepth=32
uint64_t get_output_i_l2(uint32_t ph, uint32_t pw, uint32_t od)
{
    uint32_t OutDepth     = 32;   // L2 output channels
    uint32_t lay_dim_hw_w = 32;
    return (((ph) * lay_dim_hw_w + pw) * OutDepth + od);
}


// -----------------------------------------------------------------------
// Helper: safely convert string token to uint32_t
// -----------------------------------------------------------------------
bool stringToUint32(const std::string &str, uint32_t &outValue) {
    try {
        size_t pos = 0;
        unsigned long value = std::stoul(str, &pos, 10);
        if (pos != str.size())              return false;
        if (value > std::numeric_limits<uint32_t>::max()) return false;
        outValue = static_cast<uint32_t>(value);
        return true;
    }
    catch (const std::invalid_argument &) { return false; }
    catch (const std::out_of_range &)     { return false; }
}


// -----------------------------------------------------------------------
// Bootstrapping parameters (identical to Layer 1)
// -----------------------------------------------------------------------
TFheGateBootstrappingParameterSet *our_default_bootstrapping_parameters(int32_t minimum_lambda)
{
    if (minimum_lambda != 80)
        cerr << "Sorry, for now, the parameters are only implemented for security-level of about 80 [bit].\n";

    LweParams  *params_in    = new_LweParams (n,    ks_stdev, max_stdev);
    TLweParams *params_accum = new_TLweParams(N, k, bk_stdev, max_stdev);
    TGswParams *params_bk    = new_TGswParams(bk_l, bk_Bgbit, params_accum);

    TfheGarbageCollector::register_param(params_in);
    TfheGarbageCollector::register_param(params_accum);
    TfheGarbageCollector::register_param(params_bk);

    TFheGateBootstrappingParameterSet *params;
    params = new TFheGateBootstrappingParameterSet(ks_length, ks_basebit, params_in, params_bk);
    return params;
}


// -----------------------------------------------------------------------
// processNeuralNet — LAYER 2 MAIN COMPUTATION
// -----------------------------------------------------------------------
int32_t processNeuralNet()
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // -----------------------------------------------------------------------
    // GROUP 1 — Layer 2 topology constants
    // -----------------------------------------------------------------------
    int32_t pheight          = 32;
    int32_t pwidth           = 32;
    int32_t n_strides        = pheight * pwidth;           // 1024 spatial positions
    int32_t num_channels_in  = L2_NUM_CHANNELS_IN;         // 128 — from Layer 1
    int32_t num_channels_out = L2_NUM_CHANNELS_OUT;        // 32  — Layer 2 output
    int32_t kernel_size      = L2_KERNEL_SIZE;             // 3x3
    int32_t half_k           = kernel_size / 2;            // 1 (for padding offset)
    int32_t lay_id           = 2;                          // Layer 2 identifier

    // -----------------------------------------------------------------------
    // Setup: generate / load TFHE parameters and secret key
    // -----------------------------------------------------------------------
    if (VERBOSE) cout << "[GEN] Generate parameters for security level " << minimum_lambda << " [bit]." << endl;
    TFheGateBootstrappingParameterSet *params = our_default_bootstrapping_parameters(minimum_lambda);

    TFheGateBootstrappingSecretKeySet *secret;

    // Layer 2 always reads the key written by Layer 1 (GENERATE = false)
    if (VERBOSE) cout << "[READ] User's secret key is read from file: " << FILE_USER_KEY << endl;
    FILE* sk = fopen(FILE_USER_KEY, "rb");
    if (!sk) { cerr << "Error: Cannot open key file " << FILE_USER_KEY << endl; return 1; }
    secret = new_tfheGateBootstrappingSecretKeySet_fromFile(sk);
    fclose(sk);

    // Key references
    const LweBootstrappingKeyFFT *bs_key    = secret->cloud.bkFFT;
    const TLweKey    *tLwe_key              = &((secret->tgsw_key)->tlwe_key);
    const TGswParams *tgsw_params           = ((secret->cloud.bk)->bk_params);
    const TLweParams *tlwe_params           = ((secret->cloud.bk)->accum_params);
    const LweParams  *final_params          = &(tlwe_params->extracted_lweparams);
    const LweParams  *in_out_params         = params->in_out_params;
    const LweKey     *user_sk               = secret->lwe_key;
    const LweParams  *user_in_out_params    = params->in_out_params;

    LweKey *final_key = new_LweKey(final_params);
    tLweExtractKey(final_key, tLwe_key);

    // Mu for Layer 2 bootstrapping output
    const Torus32 mu_boot_l2 = modSwitchToTorus32(1, space_after_l2);
    Torus32 mu, mub, phase;
    mub = modSwitchToTorus32(1, space_msg);

    // -----------------------------------------------------------------------
    // GROUP 4 — Open output files
    //   lconv_core2/3/4.txt  : per-channel bootstrap result log
    //   memopt_l2nodeimap.txt : global index map (p, ph, pw, output_i, sign)
    //   conf_bs_errors_l2.txt : wrong-bootstrap error log
    // -----------------------------------------------------------------------
    stringstream sss1, sss2, sss3, sss4;
    stringstream sss_1, sss_2, sss_3, sss_4;

    sss1  << "lconv_core2_l2_masterM.txt";
    sss2  << "lconv_core3_l2_masterM.txt";
    sss3  << "lconv_core4_l2_masterM.txt";
    sss4  << "lconv_core5_l2_masterM.txt";
    sss_1 << "memopt_l2nodeimap_core2.txt";
    sss_2 << "memopt_l2nodeimap_core3.txt";
    sss_3 << "memopt_l2nodeimap_core4.txt";
    sss_4 << "memopt_l2nodeimap_core5.txt";

    std::ofstream outputFile1(sss1.str());
    std::ofstream outputFile2(sss2.str());
    std::ofstream outputFile3(sss3.str());
    std::ofstream outputFile4(sss4.str());
    std::ofstream indexFile1(sss_1.str());
    std::ofstream indexFile2(sss_2.str());
    std::ofstream indexFile3(sss_3.str());
    std::ofstream indexFile4(sss_4.str());

    std::ofstream outFile("/home/pi3/conf_bs_errors_l2.txt");
    if (!outFile.is_open())
        throw std::runtime_error("Error: Could not open conf_bs_errors_l2.txt for writing.");
    else
        std::cout << "Layer 2 error-log file opened successfully.\n";
    outFile << "Layer 2 Bootstrap Error Log\n";

    int32_t wrong_boos = 0;
    std::vector<int> wrongbsvec;

    // -----------------------------------------------------------------------
    // GROUP 1 — Load Layer 2 bias from layer2_bias.txt
    // -----------------------------------------------------------------------
    std::cout << "Loading Layer 2 biases from: " << FILE_L2_BIAS << endl;
    std::ifstream file_bias(FILE_L2_BIAS);
    std::vector<int> l2_biases;
    if (file_bias.is_open()) {
        std::string line;
        while (std::getline(file_bias, line)) {
            std::istringstream stream(line);
            std::string word;
            while (stream >> word) {
                uint32_t number;
                if (stringToUint32(word, number))
                    l2_biases.push_back(int(number));
                else
                    std::cout << "Out of range/invalid value in layer2_bias.txt\n";
            }
        }
        file_bias.close();
    } else {
        cerr << "Error: Unable to open " << FILE_L2_BIAS << endl;
    }
    std::cout << "Layer 2 bias count == " << l2_biases.size() << endl;
    for (int b : l2_biases) std::cout << b << " ";
    std::cout << "\n\n";

    // -----------------------------------------------------------------------
    // GROUP 1 — Load Layer 2 kernel weights from layer2_weights.txt
    //   Format: flat integers, stored as [out_ch][in_ch][kh][kw]
    //   Total: 32 * 128 * 3 * 3 = 36864 values
    // -----------------------------------------------------------------------
    std::cout << "Loading Layer 2 weights from: " << FILE_L2_WEIGHTS << endl;
    std::ifstream file_wgt(FILE_L2_WEIGHTS);
    std::vector<int> l2_weights_flat;
    if (file_wgt.is_open()) {
        std::string line;
        while (std::getline(file_wgt, line)) {
            std::istringstream stream(line);
            std::string word;
            while (stream >> word) {
                uint32_t number;
                if (stringToUint32(word, number))
                    l2_weights_flat.push_back(int(number));
                else
                    std::cout << "Out of range/invalid value in layer2_weights.txt\n";
            }
        }
        file_wgt.close();
    } else {
        cerr << "Error: Unable to open " << FILE_L2_WEIGHTS << endl;
    }
    int32_t expected_weights = num_channels_out * num_channels_in * kernel_size * kernel_size;
    std::cout << "Layer 2 weights loaded: " << l2_weights_flat.size()
              << " (expected " << expected_weights << ")" << endl;

    // Weight accessor: w[out_ch][in_ch][kh][kw]
    // Index = out_ch*(128*9) + in_ch*9 + kh*3 + kw
    auto get_l2_weight = [&](int out_ch, int in_ch, int kh, int kw) -> int32_t {
        int idx = out_ch * (num_channels_in * kernel_size * kernel_size)
                + in_ch * (kernel_size * kernel_size)
                + kh   * kernel_size
                + kw;
        if (idx < 0 || idx >= (int)l2_weights_flat.size()) return 0;
        return l2_weights_flat[idx];
    };

    // -----------------------------------------------------------------------
    // GROUP 2 — Load Layer 1 bootstrapped LWE outputs from files
    //   Files written by L1 slave nodes, pattern:
    //     L1_OUTPUT_BASE_PATH + "{ch}_{ph}_{pw}_{output_i}"
    //   We load all 128 * 32 * 32 = 131072 LWE samples into memory.
    // -----------------------------------------------------------------------
    std::cout << "\nLoading Layer 1 bootstrapped LWE samples from files...\n";
    LweSample *l1_bootstrapped = new_LweSample_array(num_channels_in * n_strides, in_out_params);

    int32_t l1_load_ok = 0, l1_load_fail = 0;
    for (int nc = 0; nc < num_channels_in; nc++)
    {
        for (int ph = 0; ph < pheight; ph++)
        {
            for (int pw = 0; pw < pwidth; pw++)
            {
                uint64_t output_i = get_output_i_l1(ph, pw, nc);
                stringstream ssio;
                ssio << L1_OUTPUT_BASE_PATH << nc << "_" << ph << "_" << pw << "_" << output_i;
                string lwe_path = ssio.str();

                FILE *flwe = fopen(lwe_path.c_str(), "rb");
                int32_t in_offset = nc * n_strides + ph * pwidth + pw;
                if (flwe) {
                    import_lweSample_fromFile(flwe, l1_bootstrapped + in_offset, in_out_params);
                    fclose(flwe);
                    l1_load_ok++;
                } else {
                    // File not found: initialise to zero trivially
                    mu = modSwitchToTorus32(0, space_after_bs);
                    lweNoiselessTrivial(l1_bootstrapped + in_offset, mu, in_out_params);
                    l1_load_fail++;
                    if (DEBUG)
                        cerr << "WARN: L1 LWE file not found: " << lwe_path << endl;
                }
            }
        }
    }
    std::cout << "L1 samples loaded OK: " << l1_load_ok
              << "  | Failed (zero-filled): " << l1_load_fail << "\n\n";

    // -----------------------------------------------------------------------
    // GROUP 3 — Allocate Layer 2 MAC accumulators and output arrays
    // -----------------------------------------------------------------------
    LweSample *wgt_sum_l2         = new_LweSample_array(num_channels_out * n_strides, in_out_params);
    LweSample *bootstrapped_l2    = new_LweSample_array(num_channels_out * n_strides, in_out_params);
    LweSample *bias_sum           = new_LweSample_array(num_channels_out * n_strides, in_out_params);

    // -----------------------------------------------------------------------
    // GROUP 3 — Main Layer 2 Conv2D MAC loop
    //   Outer: output channel p (0..31)
    //   Middle: spatial position (ph, pw) (0..31 x 0..31)
    //   Inner: accumulate over all 128 input channels x 3x3 kernel window
    //
    //   Step 0: Initialise accumulator with Layer 2 bias (trivial ciphertext)
    //   Step 1: For each (in_ch, kh, kw) -> lweAddMulTo with PoT weight
    // -----------------------------------------------------------------------
    std::cout << "=== Layer 2 Conv2D MAC Accumulation Start ===\n";
    for (int p = 0; p < num_channels_out; p++)
    {
        for (int ph = 0; ph < pheight; ph++)
        {
            for (int pw = 0; pw < pwidth; pw++)
            {
                int32_t out_offset = p * n_strides + ph * pwidth + pw;

                // Step 0: Initialise MAC accumulator with L2 bias (in L1-bootstrapped space)
                Torus32 mu_l2_bias = modSwitchToTorus32(
                    (p < (int)l2_biases.size()) ? l2_biases[p] : 0,
                    space_after_bs);
                lweNoiselessTrivial(wgt_sum_l2 + out_offset, mu_l2_bias, in_out_params);

                if (DEBUG && pw == 0)
                    std::cout << "\nCh-out=" << p << " ph=" << ph << " pw=" << pw
                              << " bias=" << ((p < (int)l2_biases.size()) ? l2_biases[p] : 0) << endl;

                // Step 1: Accumulate over all input channels with 3x3 kernel
                for (int nc = 0; nc < num_channels_in; nc++)
                {
                    for (int kh = 0; kh < kernel_size; kh++)
                    {
                        int iph = ph + (kh - half_k);          // spatial row with padding
                        if (iph < 0 || iph >= pheight) continue; // zero-padding

                        for (int kw = 0; kw < kernel_size; kw++)
                        {
                            int ipw = pw + (kw - half_k);      // spatial col with padding
                            if (ipw < 0 || ipw >= pwidth)  continue; // zero-padding

                            int32_t in_offset = nc * n_strides + iph * pwidth + ipw;
                            int32_t w = get_l2_weight(p, nc, kh, kw);

                            if (w != 0)
                                lweAddMulTo(wgt_sum_l2 + out_offset, w,
                                            l1_bootstrapped + in_offset, in_out_params);
                        }
                    }
                }

                // Debug: decrypt and print accumulated MAC result (before bootstrap)
                Torus32 result_mac = lweSymDecrypt(wgt_sum_l2 + out_offset, user_sk, space_after_bs);
                if (DEBUG)
                    printf("L2 MAC result ch=%d ph=%d pw=%d => %d\n",
                           p, ph, pw, modSwitchFromTorus32(result_mac, space_after_bs));
            }
        }
    }
    std::cout << "=== Layer 2 Conv2D MAC Accumulation Done ===\n\n";

    // -----------------------------------------------------------------------
    // GROUP 3 + 4 — OMP-parallelised PBS / Sign Activation (4 threads)
    //   Applies 2x+1 sign encoding, then tfhe_bootstrap_FFT.
    //   Validation: compare decrypted MAC sign vs bootstrap output sign.
    //   Export: write bootstrapped LWE to binary files + index maps.
    // -----------------------------------------------------------------------
    int n_threads = 4;
    auto t1 = high_resolution_clock::now();

    std::cout << "=== Layer 2 PBS / Sign Activation Start (OMP " << n_threads << " threads) ===\n";

    for (int p = 0; p < num_channels_out; p++)
    {
        for (int ph = 0; ph < pheight; ph++)
        {
            // Process pwidth positions in chunks of n_threads
            for (int pw = 0; pw < pwidth; pw = pw + n_threads)
            {
                std::vector<Torus32> boos_out_tid;
                std::vector<Torus32> convoutres_tid;
                std::mutex mtx;

                #pragma omp parallel for \
                    shared(bias_sum, boos_out_tid, convoutres_tid, wrong_boos, bootstrapped_l2, wgt_sum_l2) \
                    firstprivate(space_msg, space_after_bs, space_after_l2, bs_key)
                for (int tid = 0; tid < n_threads; tid++)
                {
                    int t_id = omp_get_thread_num();
                    int32_t boos_offset = p * n_strides + ph * pwidth + pw + t_id;

                    if ((pw + t_id) >= pwidth) continue;  // guard against edge

                    // 2x+1 sign transformation to disambiguate zero
                    lweNoiselessTrivial(bias_sum + boos_offset, mub, in_out_params);
                    lweAddMulTo(bias_sum + boos_offset, 2, wgt_sum_l2 + boos_offset, in_out_params);

                    LweSample *LWE_in = bias_sum + boos_offset;
                    LweSample *LWE_bs = bootstrapped_l2 + boos_offset;

                    Torus32 convoutres;
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        convoutres = lweSymDecrypt(bias_sum + boos_offset, user_sk, space_msg);
                        convoutres_tid.push_back(convoutres);
                        std::cout << "L2 MAC (2x+1) ch=" << p << " ph=" << ph
                                  << " pw=" << (pw + t_id)
                                  << " => " << modSwitchFromTorus32(convoutres, space_msg) << "\n";
                    }

                    phase = lwePhase(wgt_sum_l2 + boos_offset, user_sk);
                    std::cout << "L2 pre-BS phase offset=" << boos_offset << " phase=" << phase << "\n";

                    // PBS: bootstrap + sign extraction (ReLU → Sign via test vector)
                    tfhe_bootstrap_FFT(LWE_bs, bs_key, mu_boot_l2, LWE_in);

                    phase = lwePhase(bootstrapped_l2 + boos_offset, user_sk);
                    std::cout << "L2 post-BS phase offset=" << boos_offset << " phase=" << phase << "\n";

                    Torus32 boos_out;
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        boos_out = lweSymDecrypt(bootstrapped_l2 + boos_offset, user_sk, space_after_l2);
                        boos_out_tid.push_back(boos_out);
                    }
                    std::cout << "L2 BS error (phase-msg)=" << (phase - boos_out) << "\n";

                    uint64_t output_i = get_output_i_l2(ph, pw + t_id, p);
                    std::cout << "Thread=" << t_id
                              << " L2 BS output index=" << output_i << "\n\n";

                } // end OMP parallel
            } // end pw stride loop
        } // end ph
    } // end p (output channels)

    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << "=== Layer 2 PBS done. Time: " << ms_int.count() << " ms ===\n\n";

    // -----------------------------------------------------------------------
    // GROUP 4 — Validation, error logging, and export
    //   Walk over all (p, ph, pw), validate bootstrap sign,
    //   write to output files and export LWE binary files.
    // -----------------------------------------------------------------------
    std::cout << "=== Layer 2 Export + Validation Start ===\n";

    for (int p = 0; p < num_channels_out; p++)
    {
        for (int ph = 0; ph < pheight; ph++)
        {
            for (int pw = 0; pw < pwidth; pw++)
            {
                int32_t samp_offset = p * n_strides + ph * pwidth + pw;

                Torus32 convoutres = lweSymDecrypt(bias_sum + samp_offset, user_sk, space_msg);
                Torus32 boos_out   = lweSymDecrypt(bootstrapped_l2 + samp_offset, user_sk, space_after_l2);
                uint64_t output_i  = get_output_i_l2(ph, pw, p);

                // --- Wrong bootstrapping: negative MAC classified as positive
                if (modSwitchFromTorus32(convoutres, space_msg) > (space_msg / 2)
                    && modSwitchFromTorus32(boos_out, space_after_l2) == 1)
                {
                    std::cout << "L2 WRONG BS (-ve misclassified) ch=" << p
                              << " ph=" << ph << " pw=" << pw << "\n";
                    wrong_boos++;
                    wrongbsvec.push_back(samp_offset);
                    outFile << p << " " << ph << " " << pw << "\n";
                    outputFile1 << p << "-" << ph << "-" << pw << "-"
                                << modSwitchFromTorus32(boos_out, space_after_l2) << "\n";
                    indexFile1  << p << "-" << ph << "-" << pw << "-" << output_i << "-"
                                << modSwitchFromTorus32(convoutres, space_msg) << "-"
                                << modSwitchFromTorus32(boos_out, space_after_l2) << "\n";
                    std::cout << "Error Config (ch-ph-pw)==" << p << "--" << ph << "--" << pw << "\n";
                }
                // --- Wrong bootstrapping: positive MAC classified as negative
                else if (modSwitchFromTorus32(convoutres, space_msg) > 0
                         && modSwitchFromTorus32(convoutres, space_msg) <= (space_msg / 2)
                         && modSwitchFromTorus32(boos_out, space_after_l2) == (space_after_l2 - 1))
                {
                    std::cout << "L2 WRONG BS (+ve misclassified) ch=" << p
                              << " ph=" << ph << " pw=" << pw << "\n";
                    wrong_boos++;
                    wrongbsvec.push_back(samp_offset);
                    outFile << p << " " << ph << " " << pw << "\n";
                    outputFile1 << p << "-" << ph << "-" << pw << "-"
                                << modSwitchFromTorus32(boos_out, space_after_l2) << "\n";
                    indexFile1  << p << "-" << ph << "-" << pw << "-" << output_i << "-"
                                << modSwitchFromTorus32(convoutres, space_msg) << "-"
                                << modSwitchFromTorus32(boos_out, space_after_l2) << "\n";
                    std::cout << "Error Config (ch-ph-pw)==" << p << "--" << ph << "--" << pw << "\n";
                }
                // --- Wrong bootstrapping: zero-output edge case
                else if (modSwitchFromTorus32(convoutres, space_msg) == 0
                         && modSwitchFromTorus32(boos_out, space_after_l2) == (space_after_l2 - 1))
                {
                    std::cout << "L2 WRONG BS (zero sign error) ch=" << p
                              << " ph=" << ph << " pw=" << pw << "\n";
                    wrong_boos++;
                    wrongbsvec.push_back(samp_offset);
                    outFile << p << " " << ph << " " << pw << "\n";
                    outputFile1 << p << "-" << ph << "-" << pw << "-"
                                << modSwitchFromTorus32(boos_out, space_after_l2) << "\n";
                    indexFile1  << p << "-" << ph << "-" << pw << "-" << output_i << "-"
                                << modSwitchFromTorus32(convoutres, space_msg) << "-"
                                << modSwitchFromTorus32(boos_out, space_after_l2) << "\n";
                    std::cout << "Error Config (0-sign)==" << p << "--" << ph << "--" << pw << "\n";
                }
                // --- Correct bootstrapping
                else
                {
                    std::cout << "L2 Correct BS ch=" << p << " ph=" << ph << " pw=" << pw << "\n";
                    outputFile1 << p << "-" << ph << "-" << pw << "-"
                                << modSwitchFromTorus32(convoutres, space_msg) << "-"
                                << modSwitchFromTorus32(boos_out, space_after_l2) << "\n";
                    indexFile1  << p << "-" << ph << "-" << pw << "-" << output_i << "-"
                                << modSwitchFromTorus32(convoutres, space_msg) << "-"
                                << modSwitchFromTorus32(boos_out, space_after_l2) << "\n";
                }

                std::cout << "Wrong BS so far (L2)==" << wrong_boos << "\n\n";

                // --- Export bootstrapped L2 LWE sample to binary file
                stringstream ssio;
                ssio << "/home/pi3/helloworld/image3_l2out_lwesample_"
                     << p << "_" << ph << "_" << pw << "_" << output_i;
                string accumsample_path = ssio.str();
                FILE *flwesample_write = fopen(accumsample_path.c_str(), "wb");
                if (flwesample_write) {
                    export_lweSample_toFile(flwesample_write, bootstrapped_l2 + samp_offset, in_out_params);
                    fclose(flwesample_write);
                } else {
                    cerr << "WARN: Cannot write L2 LWE sample to " << accumsample_path << "\n";
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // GROUP 4 — Close files, print summary
    // -----------------------------------------------------------------------
    outFile.close();
    outputFile1.close();
    outputFile2.close();
    outputFile3.close();
    outputFile4.close();
    indexFile1.close();
    indexFile2.close();
    indexFile3.close();
    indexFile4.close();

    std::cout << "\n=== Layer 2 Export + Validation Done ===\n\n";
    std::cout << "Total L2 wrong-bootstrap errors: " << wrong_boos << "\n";
    std::cout << "Total L2 correct samples:        "
              << (num_channels_out * n_strides - wrong_boos) << "\n";

    std::cout << "Indices of wrong-bootstrapped L2 samples:\n";
    for (int i = 0; i < (int)wrongbsvec.size(); ++i)
        std::cout << wrongbsvec[i] << " ";
    std::cout << "\n\n";

    // -----------------------------------------------------------------------
    // GROUP 5 — Cleanup: free all allocated LWE arrays
    //   (The old L2/L3 stub blocks from the M1569 Layer 1 file are removed.
    //    Layer 2 result (bootstrapped_l2) feeds Layer 3 AvgPool2D next.)
    // -----------------------------------------------------------------------
    delete_LweSample_array(num_channels_in * n_strides,  l1_bootstrapped);
    delete_LweSample_array(num_channels_out * n_strides, wgt_sum_l2);
    delete_LweSample_array(num_channels_out * n_strides, bootstrapped_l2);
    delete_LweSample_array(num_channels_out * n_strides, bias_sum);
    delete_LweKey(final_key);

    std::cout << "Layer 2 complete. 32-channel bootstrapped feature maps ready for AvgPool2D / Layer 3.\n";

    return 0;
}


// -----------------------------------------------------------------------
// Signal handler
// -----------------------------------------------------------------------
void my_signal_handler(int32_t sig)
{
    cerr << "*** Received SIGINT *** (Stopping gracefully...)\n";
    got_sigterm = 1;
    return;
}


// -----------------------------------------------------------------------
// writeStatistics (retained for compatibility)
// -----------------------------------------------------------------------
void writeStatistics()
{
    if (MODE_STATS) {
        ostringstream stats;
        stats << "\n[STAT] Layer 2 Statistics\n";
        stats << "Errors: "              << count_errors        << "\n";
        stats << "Wrong bootstraps: "    << count_wrong_bs      << "\n";
        stats << "Avg time (class): "    << avg_time_per_classification  << " s\n";
        stats << "Avg time (BS): "       << avg_time_per_bootstrapping   << " s\n";
        string str = stats.str();
        cout << str;
        ofstream stat(FILE_STATISTICS);
        stat << str;
        stat.close();
        cout << "\n[SAVE] Wrote statistics to: " << FILE_STATISTICS << endl;
    }
    if (MODE_LATEX) {
        ostringstream latex;
        latex << "%\\input{" << FILE_LATEX << "}\n";
        latex << "% Layer 2 Experiments\n";
        latex << "\\newcommand{\\EXPnumBS}{$"     << total_num_hidden_neurons         << "$}\n";
        latex << "\\newcommand{\\EXPbsEXACT}{$"   << avg_time_per_bootstrapping       << "$\\ [sec/bootstrapping]}\n";
        latex << "\\newcommand{\\EXPtimeEXACT}{$" << avg_time_per_classification      << "$\\ [sec/classification]}\n";
        latex << "\\newcommand{\\EXPnumERRabs}{$" << count_errors                     << "$}\n";
        latex << "\\newcommand{\\EXPwrongBSabs}{$"<< count_wrong_bs                   << "$}\n";
        ofstream lat(FILE_LATEX);
        lat << latex.str();
        lat.close();
        cout << "[SAVE] Wrote LaTeX result to: " << FILE_LATEX << endl;
    }
}


// -----------------------------------------------------------------------
// deleteMatrix / deleteTensor (retained for compatibility)
// -----------------------------------------------------------------------
void deleteMatrix(int32_t** matrix, int32_t dim_mat)
{
    for (int32_t i = 0; i < dim_mat; ++i) delete[] matrix[i];
    delete[] matrix;
}

void deleteTensor(int32_t*** tensor, int32_t dim_tensor, const int32_t* dim_vec)
{
    for (int32_t i = 0; i < dim_tensor; ++i)
        deleteMatrix(tensor[i], dim_vec[i]);
    delete[] tensor;
}


// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------
int32_t main(int32_t argc, char *argv[])
{
    time_t timestamp = time(nullptr);
    ostringstream test;
    test << asctime(localtime(&timestamp));
    test << "Layer 2 Master — Single Node. Call: " << argv[0] << "\n";

    if (signal((int) SIGINT, my_signal_handler) == SIG_ERR)
        cerr << "Registering my_signal_handler failed!\n";

    if (VERBOSE) {
        cout << test.str();
        cout << "Starting Layer 2 evaluation (TFHE Conv2D, 128->32 channels, 32x32 spatial).\n";
        cout << "Alpha = " << alpha << "\n";
        cout << "*** Graceful exit on CTRL+C ***\n\n";
    }

    processNeuralNet();

    test << avg_time_per_bootstrapping  << " [sec/bootstrapping]\n";
    test << avg_time_per_classification << " [sec/classification]\n\n";

    ofstream testf(FILE_TEST, ios_base::app);
    testf << test.str();
    testf.close();
    cout << "[SAVE] Appended Layer 2 results to: " << FILE_TEST << endl;

    return 0;
}
