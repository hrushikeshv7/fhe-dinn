
// ----------------------------------------------------------------------------|
// Title      : Fast Homomorphic Evaluation of Deep Discretized Neural Networks
// Project    : Showcase Fast Fully Homomorphic Evaluation of Encrypted Inputs
//              using Deep Discretized Neural Networks hence preserving Privacy
// ----------------------------------------------------------------------------|
// File       : nn.cpp
// Authors    : Florian Bourse      <Florian.Bourse@ens.fr>
//              Michele Minelli     <Michele.Minelli@ens.fr>
//              Matthias Minihold   <Matthias.Minihold@RUB.de>
//              Pascal Paillier     <Pascal.Paillier@cryptoexperts.com>
//
// Reference  : TFHE: Fast Fully Homomorphic Encryption Library over the Torus
//              https://github.com/tfhe
// ----------------------------------------------------------------------------|
// Description:
//     Showcases how to efficiently evaluate privacy-perserving neural networks.
// ----------------------------------------------------------------------------|
// Revisions  :
// Date        Version  Description
// 2017-11-16  0.3.0    Version for github, referenced by ePrint32_t paper
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
volatile sig_atomic_t got_sigterm = 0;  /// Declared volatile to let (optimizing) compilers know it changes asynchronously.


// Defines
#define PROCESSES  1   /// Use multiple processes.
#define VERBOSE    true /// Some output.
#define DEBUG      true /// Some more output.
// -----------------------------------------------------------------------
// KEY GENERATION
//   This node MUST reuse the single shared key generated once by M1569
//   on pi3 (/home/pi3/key_user_secret.bin, /home/pi3/tfhe/src/test/key_cloud.bin).
//   Copy those two files to this Pi's FILE_USER_KEY / FILE_CLOUD_KEY paths
//   BEFORE running this binary. GENERATE must stay false -- this Layer-2
//   node imports Layer-1 ciphertexts from other Pis and must use the SAME
//   key those were encrypted/bootstrapped under, or all results here are
//   cryptographically meaningless.
// -----------------------------------------------------------------------
#define GENERATE   false /// If true: Generates fresh keys, else reuses saved ones.
#define MODE_STATS true /// If true: Writes a file with some statistics, else not.
#define MODE_LATEX true /// If true: Writes a file with some LaTeX formatted listing, else not.
#define MODE_BATCH true /// If true: Batching multiple inputs into one polynomial, else individual.
#define MODE_NOISE true /// If true: Adds Gaussian noise to samples, else trivially broken or for testing.


/*
# ePrint32_t and github release:
+ TODO rename functions, files,
+ TODO add a separate make for them,
+ TODO maybe use and test FFTW,
+ TODO answer&clear all TODOs, comments throughout the files,
+ TODO integrate the tfhelib sources correctly as submodule from github,
+ TODO add disclaimer, copyright/-left/licence notes like this?,
    Disclaimer
    This is an implementation for academic purposes, hence NOT to be used AS AS, not to be considered SECURED nor is the code well documented. Following the code-sharing idea, we publish it under license as a way to support  and to allow the community to verify easily both the correctness and the efficiency of this homomorphic evaluation.
    https://creativecommons.org/publicdomain/zero/1.0/
+ [DONE] clean nn.cpp
+ [DONE] annotate nn.cpp for future experiments/optimization,
+ [DONE] prepare nn.cpp for parallelization (processes, threads),
+ [DONE] adapt nn.cpp to handle multiple hidden layers as in the hom-mnist.py
+ [DONE] add codeblocks project for building,
+ [DONE] add howto use for manual building, e.g.,
            cd build (files are here)
            cmake ../src/ -DENABLE_TESTS=on -DENABLE_FFTW=off -DENABLE_NAYUKI=off -DCMAKE_BUILD_TYPE=optim
            make
            cd test
            ./nn-spqlios-fma (or ./nn-spqlios-avx)

// Security constants used for submission
#define SECLEVEL 80
#define SECNOISE true
#define SECALPHA pow(2., -30)
#define SEC_PARAMS_STDDEV    pow(2., -30)
#define SEC_PARAMS_n  450                   ///  LweParams
#define SEC_PARAMS_N 1024                   /// TLweParams
#define SEC_PARAMS_k    1                   /// TLweParams
#define SEC_PARAMS_BK_STDDEV pow(2., -38)   /// TLweParams
#define SEC_PARAMS_BK_BASEBITS  8           /// TGswParams
#define SEC_PARAMS_BK_LENGTH    3           /// TGswParams
#define SEC_PARAMS_KS_STDDEV pow(2., -20)   /// Key Switching Params
#define SEC_PARAMS_KS_BASEBITS  1           /// Key Switching Params
#define SEC_PARAMS_KS_LENGTH   18           /// Key Switching Params
*/

// Security constants used for testing
#define SEC_LEVEL            80
#define SEC_PARAMS_n         450           ///  LweParams
//#define SEC_PARAMS_n         600
#define SEC_ALPHA            pow(2., -20)   ///  LweParams
//#define SEC_ALPHA            pow(2., -30)
#define SEC_PARAMS_STDDEV    pow(2., -30)   /// TLweParams and LweParams (max)
//#define SEC_PARAMS_N         1024          /// TLweParams
#define SEC_PARAMS_N         1024
#define SEC_PARAMS_k         1              /// TLweParams
#define SEC_PARAMS_BK_STDDEV pow(2., -36)   /// TLweParams
//#define SEC_PARAMS_BK_STDDEV pow(2., -18)
#define SEC_PARAMS_BK_BASEBITS  10          /// TGswParams
#define SEC_PARAMS_BK_LENGTH    3           /// TGswParams
// Best parameters evar!
#define SEC_PARAMS_KS_STDDEV pow(2., -25)   /// Key Switching Params
#define SEC_PARAMS_KS_BASEBITS  1           /// Key Switching Params
#define SEC_PARAMS_KS_LENGTH    18           /// Key Switching Params
//#define SEC_PARAMS_KS_BASEBITS  3           /// Key Switching Params
//#define SEC_PARAMS_KS_LENGTH    5
// The expected topology of the provided neural network is 256:30:10
#define NUM_NEURONS_INPUT  256
#define NUM_NEURONS_HIDDEN 30
#define NUM_NEURONS_OUTPUT 10
#define NUM_NEURON_LAYERS 3

#define CARD_TESTSET 10000
#define MSG_SLOTS    143
//#define MSG_SLOTS   71

//#define TORUS_SLOTS  260
//#define TORUS_SLOTS  11
//#define TORUS_SLOTS  25
//#define TORUS_SLOTS  231


//#define TORUS_SLOTS  143
#define TORUS_SLOTS  1569
//#define TORUS_SLOTS  71
#define TORUS_SLOTS2 270
#define TORUS_SLOTS3 170
#define TORUS_SLOTS4  30
#define TORUS_SLOTS22 11


// Files are expected in the executable's directory
#define FILE_TEST           "/home/soumik/fhe-dinn/tfhe/src/test/tests.txt"
#define FILE_TXT_IMG        "/home/soumik/fhe-dinn/tfhe/src/test/txt_img_test.txt"
#define FILE_TXT_BIASES     "/home/soumik/fhe-dinn/tfhe/src/test/txt_biases.txt"
#define FILE_TXT_WEIGHTS    "/home/soumik/fhe-dinn/tfhe/src/test/txt_weights.txt"
#define FILE_TXT_LABELS     "/home/soumik/fhe-dinn/tfhe/src/test/txt_labels.txt"
#define FILE_STATISTICS     "/home/soumik/fhe-dinn/tfhe/src/test/results_stats.txt"
#define FILE_LATEX          "/home/soumik/fhe-dinn/tfhe/src/test/results_LaTeX.tex"
//#define FILE_USER_KEY       "/home/pi/fhe-dinn/tfhe/src/test/key_user_secret.bin"
#define FILE_USER_KEY       "/home/pi10/key_user_secret.bin"
//#define FILE_CLOUD_KEY      "/home/soumik/fhe-dinn/tfhe/src/test/key_cloud.bin"
#define FILE_CLOUD_KEY      "/home/pi10/tfhe/src/test/key_cloud.bin"

// -----------------------------------------------------------------------
// LAYER 2 TOPOLOGY (matches lyer2_masterM.cpp reference)
//   num_channels_in  : 128  (ALL Layer-1 output channels, read by every node)
//   num_channels_out : 32   (Layer-2 output channels, split across 12 nodes)
//   kernel_size      : 3    (3x3 spatial kernel, zero-padded at edges)
// -----------------------------------------------------------------------
#define L2_NUM_CHANNELS_IN   128
#define L2_NUM_CHANNELS_OUT  32
#define L2_KERNEL_SIZE       3

// Layer-1 output files (synced copy of ALL 12 Layer-1 nodes' outputs,
// must be present locally as image3_l1out_lwesample_{ch}_{ph}_{pw}_{idx}).
#define L1_OUTPUT_BASE_PATH "/home/pi10/accsamples_12nodes/image3_l1out_lwesample_"
// Layer-2 bias and kernel-weight files (same format/content as on pi3,
// 32 biases and 32*128*3*3=36864 weight ints respectively).
#define FILE_L2_BIAS        "/home/pi10/tfhe/build/layer2_bias.txt"
#define FILE_L2_WEIGHTS     "/home/pi10/tfhe/build/layer2_weights.txt"
// Where this node writes its Layer-2 output for Layer-3 / sync
#define L2_OUTPUT_BASE_PATH "/home/pi10/accsamples_12nodes/image3_l2out_lwesample_"

// Tweak trained neural network
#define THRESHOLD_WEIGHTS  9
#define THRESHOLD_SCORE -100


using namespace std;


// Internal Functions
/// deleteMatrix
void deleteMatrix(int32_t**  matrix, int32_t dim_mat);

/// deleteTensor
void deleteTensor(int32_t*** tensor, int32_t dim_mat, const int32_t* dim_vec);

/// Display/Write some statistics and/or LaTeX output
void writeStatistics();

/// Handle user-initiated end of computation i.e. SIGINT, when interrupted from keyboard (CRTL+C) and save progress!
void my_signal_handler(int32_t sig);

/// Generate bootstrapping parameters for FHE_NN
TFheGateBootstrappingParameterSet *our_default_bootstrapping_parameters(int32_t minimum_lambda);

/// (un)batched (Multi-Threaded) Processing of given Neural Net that writes/outputs some global statistics.
int32_t processNeuralNet();


// Global Counters, Timings, Variables
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


// Global Constants
const int32_t num_processes = PROCESSES;
const bool batched          = MODE_BATCH;

// Security
const int32_t minimum_lambda = SEC_LEVEL;
const double alpha = SEC_ALPHA;

static const int32_t n = SEC_PARAMS_n;
static const int32_t N = SEC_PARAMS_N;
static const int32_t k = SEC_PARAMS_k;
static const double max_stdev = SEC_PARAMS_STDDEV;   //mulBySqrtTwoOverPi(pow(2., -30));    //max standard deviation for a 1/4 msg space

static const int32_t bk_Bgbit = SEC_PARAMS_BK_BASEBITS;  //<-- ld, thus: 2^10
static const int32_t bk_l     = SEC_PARAMS_BK_LENGTH;
static const double bk_stdev  = SEC_PARAMS_BK_STDDEV; //mulBySqrtTwoOverPi(pow(2., -30));   //standard deviation

static const int32_t ks_basebit = SEC_PARAMS_KS_BASEBITS; //<-- ld, thus: 2^1
static const int32_t ks_length  = SEC_PARAMS_KS_LENGTH;
static const double ks_stdev    = SEC_PARAMS_KS_STDDEV; //mulBySqrtTwoOverPi(pow(2., -30));   //standard deviation

const bool noisyLWE      = MODE_NOISE;

// Input data
const int32_t num_images = CARD_TESTSET;

// Network specific
//const int32_t num_neuron_layers = NUM_NEURON_LAYERS;
//const int32_t num_wire_layers = NUM_NEURON_LAYERS - 1;
//const int32_t num_neurons_in = NUM_NEURONS_INPUT;
//const int32_t num_neurons_hidden = NUM_NEURONS_HIDDEN;
//const int32_t num_neurons_out = NUM_NEURONS_OUTPUT;
const int32_t total_num_hidden_neurons = CARD_TESTSET * NUM_NEURONS_HIDDEN;  //TODO (sum all num_neurons_hidden)*n_images


// Derived quantitie (fsor statistics output)
//const double clocks2seconds = 1. / CLOCKS_PER_SEC;
//const double avg_bs = 1./NUM_NEURONS_HIDDEN;
//const double avg_total_bs = 1./total_num_hidden_neurons;
//const int32_t batch = (CARD_TESTSET+num_processes-1)/num_processes; // =ceil(1./PROCESSES*n_images) quick ceiling hack


// TODO µ Suggestion: make topology definition variable, passing on CMD line
// Vector of number of neurons in layer_in, layer_H1, layer_H2, ..., layer_Hd, layer_out;
//const int32_t topology[num_neuron_layers] = {num_neurons_in, num_neurons_hidden, num_neurons_out};

const int32_t space_msg = MSG_SLOTS;
const int32_t space_after_bs = TORUS_SLOTS;
//const int32_t space_after_l2=TORUS_SLOTS3;
//const int32_t space_after_l3 = TORUS_SLOTS22;
//const int32_t space_after_l22=TORUS_SLOTS4;
//const int32_t space_after_n2=TORUS_SLOTS22;


const int32_t space_after_l2=TORUS_SLOTS;
//const int32_t space_after_l3 = TORUS_SLOTS;
//const int32_t space_after_l22=TORUS_SLOTS;
const int32_t space_after_n2=TORUS_SLOTS;

const int node_id=10;

//const bool clamp_biases  = false;
//const bool clamp_weights = false;

//const int32_t threshold_biases  = THRESHOLD_WEIGHTS;
//const int32_t threshold_weights = THRESHOLD_WEIGHTS;
//const int32_t threshold_scores  = THRESHOLD_SCORE;


TFheGateBootstrappingParameterSet *our_default_bootstrapping_parameters(int32_t minimum_lambda)
{
    if ( minimum_lambda != 80 ) // (minimum_lambda < 80)  || (128 < minimum_lambda) ) //TODO change hardcoded security level.
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

void tfhe_ReLU_FFT(LweSample *result,
		   const LweBootstrappingKeyFFT *bk,
		   Torus32 mu,
		   int32_t slots,
		   const LweSample *u) {

    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t Nx2 = 2 * N;
    const int32_t n = in_params->n;

    LweSample *x = new_LweSample(in_params);
    lweKeySwitch(x, bk->ks, u);

    TorusPolynomial *testvect = new_TorusPolynomial(N);
    // int32_t *bara = new int[(3*n)/2];
    int32_t *bara = new int[N];

    // Modulus switching
    int32_t barb = modSwitchFromTorus32(x->b, Nx2);
    for (int32_t i = 0; i < n/2; i++) {
        bara[3*i] = modSwitchFromTorus32(x->a[2*i]+x->a[2*i+1], Nx2);
        bara[3*i+1] = modSwitchFromTorus32(x->a[2*i], Nx2);
        bara[3*i+2] = modSwitchFromTorus32(x->a[2*i+1], Nx2);
    }

    // the initial testvec = [mu,mu,mu,...,mu]
    for (int32_t i = 0; i < N/2; i++) testvect->coefsT[i] = ((i*slots)+slots/2)/Nx2 * mu;
    for (int32_t i = N/2; i < N; i++) testvect->coefsT[i] = 0;

    // Bootstrapping rotation and extraction
    tfhe_blindRotateAndExtract_FFT(result, testvect, bk->bkFFT, barb, bara, n, bk_params);

    delete[] bara;
    delete_TorusPolynomial(testvect);
    delete_LweSample(x);
}

//Inplace max: result = max(x,result)
void tfhe_max_FFT(LweSample *result, const LweSample *x,
		   const LweBootstrappingKeyFFT *bk,
		   Torus32 mu,
		   int32_t slots)
{
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *final_params = &(accum_params->extracted_lweparams);
    lweSubTo(result, x, final_params);
    LweSample *u = new_LweSample(final_params);
    lweCopy(u, result, final_params);
    tfhe_ReLU_FFT(result, bk, mu, slots, u);
    lweAddTo(result, x, final_params);
}


// Function to safely convert string to uint32_t
bool stringToUint32(const std::string &str, uint32_t &outValue) {
    try {
        size_t pos = 0;
        unsigned long value = std::stoul(str, &pos, 10); // base 10

        // Ensure the whole string was parsed
        if (pos != str.size()) {
            return false; // extra characters found
        }

        // Check range for uint32_t
        if (value > std::numeric_limits<uint32_t>::max()) {
            return false; // out of range
        }

        outValue = static_cast<uint32_t>(value);
        return true;
    }
    catch (const std::invalid_argument &) {
        // Not a valid number
        return false;
    }
    catch (const std::out_of_range &) {
        // Number too large for unsigned long
        return false;
    }
}



uint64_t get_output_i(uint32_t ph, uint32_t pw, uint32_t od)
{
    //uint32_t OutDepth=512;
    //uint32_t OutDepth=128;  // Layer 1: 128 output channels
    uint32_t OutDepth=32;     // Layer 2: 32 output channels
    uint32_t lay_dim_hw_w=32;

    //uint32_t lay_dim_hw_w=8;
    return (((ph)*lay_dim_hw_w + pw)*OutDepth + od) ;
}

// Same flat-index formula, but with OutDepth=128 -- this is the formula the
// Layer-1 nodes used when naming their exported image3_l1out_lwesample_*
// files (get_output_i in the Layer-1 12-node/slave files). Layer 2 needs
// this to reconstruct those file names when reading all 128 input channels.
uint64_t get_output_i_l1(uint32_t ph, uint32_t pw, uint32_t od)
{
    uint32_t OutDepth=128;
    uint32_t lay_dim_hw_w=32;
    return (((ph)*lay_dim_hw_w + pw)*OutDepth + od) ;
}





int32_t processNeuralNet()
{

    // Some derived values of future interest
    // const int32_t torus_slices_upper_half[num_wire_layers] = {space_msg,       space_after_bs};
    // const int32_t torus_slices_all       [num_wire_layers] = {2*space_msg,   2*space_after_bs};
    // const int32_t torus_slices_allowed   [num_wire_layers] = {2*space_msg-1, 2*space_after_bs-1};


    // Huge arrays
    /*LagrangeHalfCPolynomial ***fft_weights = new LagrangeHalfCPolynomial**[num_wire_layers];
    int32_t*** weights = new int32_t**[num_wire_layers];  // allocate and fill matrices holding the weights
    int32_t ** biases  = new int32_t* [num_wire_layers];  // allocate and fill vectors holding the biases
    int32_t ** images  = new int32_t* [num_images];
    int32_t  * labels  = new int32_t  [num_images];*/

    //const int32_t window_size = 2;
    // Temporary variables
    string line;
    //int32_t el, l;
    //int32_t num_neurons_current_layer_in, num_neurons_current_layer_out;

    if (VERBOSE) cout << "[GEN] Generate parameters for a security level (according to  CGGI16a) of at least " << minimum_lambda << " [bit]." << endl;
    TFheGateBootstrappingParameterSet *params = our_default_bootstrapping_parameters(minimum_lambda);
    //instead of: TFheGateBootstrappingParameterSet *params = new_default_gate_bootstrapping_parameters(minimum_lambda);

    TFheGateBootstrappingSecretKeySet *secret;

    if(GENERATE)
    {
        // Cloud sets up keys
        if (VERBOSE) cout << "[GEN] Generate the secret keyset." << endl;
        //secret = new_random_gate_bootstrapping_secret_keyset(params, window_size);   commented for compiling w/o window_size
        secret = new_random_gate_bootstrapping_secret_keyset(params);
        //const LweBootstrappingKey* const cloud_bk = secret->cloud.bk;

        // Save keys to file
        if (VERBOSE) cout << "[GEN] User's secret key is exported to file: " << FILE_USER_KEY << endl;
        FILE* sk = fopen(FILE_USER_KEY,"wb");
        export_tfheGateBootstrappingSecretKeySet_toFile(sk, secret);
        fclose(sk);

        // TODO Separate into user / cloud
        /*
        if (VERBOSE) cout << "[GEN] Cloud's key is exported to file: " << FILE_CLOUD_KEY << endl << endl;
        FILE* cloud_key = fopen(FILE_CLOUD_KEY,"wb");
        export_tfheGateBootstrappingCloudKeySet_toFile(cloud_key, &secret->cloud);
        fclose(cloud_key);
        */

    }
    else
    {
        // Read keys from file
        if (VERBOSE) cout << "[READ] User's secret key is read from file: " << FILE_USER_KEY << endl;
        FILE* sk = fopen(FILE_USER_KEY,"rb");
        secret = new_tfheGateBootstrappingSecretKeySet_fromFile(sk);
        fclose(sk);

        /*
        if (VERBOSE) cout << "[READ] Cloud's key is read from file: " << FILE_CLOUD_KEY << endl << endl ;
        FILE* cloud_key = fopen(FILE_CLOUD_KEY,"rb");
        TFheGateBootstrappingCloudKeySet *cloud_keyset = new_tfheGateBootstrappingCloudKeySet_fromFile(cloud_key);
        fclose(cloud_key);
        */
    }

    // Keys
    const LweBootstrappingKeyFFT *bs_key = secret->cloud.bkFFT;
    const TLweKey *tLwe_key = &((secret->tgsw_key)->tlwe_key);
    const TGswParams *tgsw_params = ((secret->cloud.bk)->bk_params);
    const TLweParams *tlwe_params = ((secret->cloud.bk)->accum_params);
    const LweParams *final_params = &(tlwe_params->extracted_lweparams);
    LweKey *final_key = new_LweKey(final_params);
    tLweExtractKey(final_key, tLwe_key);


    LweKey *final_dummy_key = new_LweKey(final_params);
    tLweExtractKey(final_dummy_key, tLwe_key);

    // asking for generation of lookup table for X^a-1
    TGswSampleFFT *tmp = new_TGswSampleFFT(tgsw_params);
    TGswSampleFFT *tmp2 = new_TGswSampleFFT(tgsw_params);
    //tGswFFTMulByXaiMinusOne(tmp,0,tmp2,tgsw_params);  Soumik:Temporary comment 2 check compilation errs
    delete_TGswSampleFFT(tmp);
    delete_TGswSampleFFT(tmp2);

    const LweKey* user_sk = secret->lwe_key;
    const LweParams* user_in_out_params = params->in_out_params;
    const LweParams* in_out_params = params->in_out_params;

    // Program the wheel to value(s) after Bootstrapping
    const Torus32 mu_boot = modSwitchToTorus32(1, space_after_bs);
    //const Torus32 mu_boot22 = modSwitchToTorus32(1, space_after_l22);






    
    // Temporary variables and Pointers to existing arrays for convenience (they are always written, before read)
    Torus32 mu, phase;
    /*int32_t** weight_layer;
    int32_t *bias, *image;
    int32_t pixel, label;
    int32_t x, w, w0;
    bool notSameSign;*/

    // TODO µ Suggestion: delete/new/overwrite one array of length max if more than one hidden layer
    LweSample      *bootstrapped_samples;
    LweSample *bootstrapped_samples_n2,*wgt_sum_l2,*wgt_sum_l3,*bootstrapped_samples_n3;
    //TLweSample    *enc_image_TLWE;
    //TLweSampleFFT *fft_enc_image;

    //clock_t bs_begin, bs_end, net_begin, net_end;
    //double time_for_classification, time_for_bootstrapping;
    //int32_t multi_sum_clear[num_neurons_hidden];
    //int32_t output_clear   [num_neurons_out];

    // Once per id_proc/batch re-initialize counters, timings, ...
//    int32_t max_score,max_score_clear,class_enc,class_clear,score,score_clear;
  //  int32_t batch_accum_img, batch_accum_count_errors, batch_accum_count_disagreements, batch_accum_count_disag_pro_clear, batch_accum_count_disag_pro_hom, batch_accum_count_wrong_bs, batch_accum_count_errors_with_failed_bs, batch_accum_count_disagreements_with_failed_bs;
    //double batch_accum_time_classification, batch_accum_time_bootstrappings;
    //bool failed_bs;



    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    
    //Attempt to incorporate CFAR10 CNN Block 1 here


    /*std::ifstream file("/home/pi/tfhe/build/test/cfar10_l1_output.txt"); // Open the file
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file!---/home/pi/tfhe/build/test/cfar10_l1_output.txt" << std::endl;
        return 1;
    }
    std::string line0;
    std::vector<int> numbers;
    

    while (std::getline(file, line0)) {
        std::stringstream ss(line0);
        std::string value;
     
        while (std::getline(ss, value, ',')) { // Split by comma
            numbers.push_back(std::stoi(value)); // Convert to integer and store
        }

    }

    file.close();
     // Display the parsed numbers
    std::cout << "Parsed quantized inputs/numbers: ";
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    */





    //=====================================================================
    // LAYER 2 -- Distributed Conv2D (32 output channels / 12 Pi nodes)
    //   Mirrors lyer2_masterM.cpp: reads ALL 128 Layer-1 output channels,
    //   applies a real 3x3 kernel (weights from layer2_weights.txt) with
    //   zero-padding at the spatial edges ("fixed global offset"), adds
    //   the Layer-2 bias, then bootstraps (PBS/sign activation, OMP
    //   4 threads) this node's slice of the 32 output channels.
    //=====================================================================

    int32_t pheight   = 32;
    int32_t pwidth    = 32;
    int32_t n_strides = pheight * pwidth;          // 1024

    const int32_t num_channels_in  = L2_NUM_CHANNELS_IN;   // 128 (all read)
    const int32_t num_channels_out = L2_NUM_CHANNELS_OUT;  // 32
    const int32_t kernel_size      = L2_KERNEL_SIZE;       // 3
    const int32_t half_k           = kernel_size / 2;      // 1

    // Output-channel partition across the 12 nodes, taken from the
    // architecture diagram (sums to 32; every node owns >=1 channel):
    //   Pi0..Pi8 -> 3 ch each (ch 0-26), Pi9/Pi10 -> 2 ch, Pi11 -> 1 ch
    const int32_t l2_channels_per_node[12] = {3,3,3,3,3,3,3,3,3,2,2,1};
    int32_t l2_start_channel = 0;
    for (int32_t i = 0; i < node_id-1; ++i) l2_start_channel += l2_channels_per_node[i];
    const int32_t num_channels_nde = l2_channels_per_node[node_id-1]; // this node's channel count

    std::cout << "[L2] node_id=" << node_id << " owns output channels ["
              << l2_start_channel << ", " << (l2_start_channel+num_channels_nde) << ")\n";

    LweSample *conv2d_mac_sum, *bias_sum, *l1_channel;
    conv2d_mac_sum       = new_LweSample_array(n_strides*num_channels_nde, in_out_params);
    bootstrapped_samples = new_LweSample_array(n_strides*num_channels_nde, in_out_params);
    bias_sum             = new_LweSample_array(n_strides*num_channels_nde, in_out_params);

    int32_t wrong_boos=0;
    std::vector<int> wrongbsvec;

    std::ofstream outFile("/home/pi10/conf_bs_errors.txt");
    if (!outFile.is_open()) {
        throw std::runtime_error("Error: Could not open file for writing.");
    }
    else
        std::cout << "Output file opened successfully---";
    outFile << "Hello, Soumik! (Layer 2, node " << node_id << ")\n";

    //----------------------------------------------------------------
    // Load Layer-2 biases (32 values, one per output channel)
    //----------------------------------------------------------------
    std::cout << "Loading Layer 2 biases from: " << FILE_L2_BIAS << endl;
    std::ifstream file_1(FILE_L2_BIAS);
    std::vector<int> l2_biases_active;
    if (file_1.is_open()) {
        std::string line;
        while (std::getline(file_1, line)) {
            std::istringstream stream(line);
            std::string word;
            while (stream >> word) {
                uint32_t number;
                if (stringToUint32(word, number))
                    l2_biases_active.push_back(int(number));
                else
                    std::cout << "Out of range/invalid value in layer2_bias.txt\n";
            }
        }
    } else {
        std::cerr << "Error: Unable to open file " << FILE_L2_BIAS << std::endl;
    }
    file_1.close();
    std::cout << "# Size l2_biases_active array==" << l2_biases_active.size() << endl;
    if ((int)l2_biases_active.size() < num_channels_out) {
        std::cerr << "WARNING: layer2_bias.txt has " << l2_biases_active.size()
                  << " entries, expected " << num_channels_out << " -- padding with 0\n";
        while ((int)l2_biases_active.size() < num_channels_out) l2_biases_active.push_back(0);
    }

    //----------------------------------------------------------------
    // Load Layer-2 3x3 kernel weights (32*128*3*3 = 36864 ints)
    //   Format: flat [out_ch][in_ch][kh][kw], same as lyer2_masterM.cpp
    //----------------------------------------------------------------
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
        std::cerr << "Error: Unable to open " << FILE_L2_WEIGHTS << std::endl;
    }
    int32_t expected_weights = num_channels_out * num_channels_in * kernel_size * kernel_size;
    std::cout << "Layer 2 weights loaded: " << l2_weights_flat.size()
              << " (expected " << expected_weights << ")" << endl;

    // Weight accessor: w[out_ch][in_ch][kh][kw]
    auto get_l2_weight = [&](int out_ch, int in_ch, int kh, int kw) -> int32_t {
        int idx = out_ch * (num_channels_in * kernel_size * kernel_size)
                + in_ch  * (kernel_size * kernel_size)
                + kh * kernel_size + kw;
        if (idx < 0 || idx >= (int)l2_weights_flat.size()) return 0;
        return l2_weights_flat[idx];
    };

    //----------------------------------------------------------------
    // Initialise this node's MAC accumulators with the Layer-2 bias,
    // encoded at space_after_bs to match the scale of the Layer-1
    // bootstrapped ciphertexts being summed into them (same convention
    // as lyer2_masterM.cpp).
    //----------------------------------------------------------------
    for (int32_t oc_local = 0; oc_local < num_channels_nde; ++oc_local)
    {
        int32_t oc_global = l2_start_channel + oc_local;
        Torus32 mu_l2_bias = modSwitchToTorus32(l2_biases_active[oc_global], space_after_bs);
        std::cout << "Layer2 Bias ch " << oc_global << "==" << l2_biases_active[oc_global] << endl;
        for (int32_t s = 0; s < n_strides; ++s)
            lweNoiselessTrivial(conv2d_mac_sum + (oc_local*n_strides) + s, mu_l2_bias, in_out_params);
    }

    Torus32 mub = modSwitchToTorus32(1, space_msg);

    //----------------------------------------------------------------
    // Output files
    //----------------------------------------------------------------
    int32_t lay_id=3;  // Layer 2 uses lay_id=3 to distinguish output files from Layer 1 (lay_id=2)

    stringstream sss1;
    sss1 << "lconv_node" << node_id << "_l2" << lay_id << "_12node_memoptout.txt";
    string convout_path = sss1.str();
    std::ofstream outputFile1(convout_path);

    stringstream sss_1;
    sss_1 << "img3_lconv_node" << node_id << "_" << lay_id << "_memopt_l2nodeimap.txt";
    string convindex_path0 = sss_1.str();
    std::ofstream indexFile1(convindex_path0);

    //----------------------------------------------------------------
    // Conv2D MAC: stream through the 128 Layer-1 input channels one at
    // a time (memory-optimised) and accumulate each one's 3x3-kernel
    // contribution into every (output channel, ph, pw) this node owns.
    // Boundary handling = zero-padding (skip out-of-range kh/kw), this
    // is the "fixed global offset" from the architecture diagram.
    //----------------------------------------------------------------
    std::cout << "About 2 run Layer 2 Conv2D (128 in-ch, 3x3 kernel, "
              << num_channels_nde << " out-ch on node " << node_id << ")\n";

    l1_channel = new_LweSample_array(n_strides, in_out_params);
    int32_t l1_load_ok = 0, l1_load_fail = 0;

    for (int32_t nc = 0; nc < num_channels_in; ++nc)
    {
        // Load this Layer-1 output channel's 32x32 bootstrapped map
        for (int32_t ph = 0; ph < pheight; ++ph) {
            for (int32_t pw = 0; pw < pwidth; ++pw) {
                int32_t s = ph*pwidth + pw;
                uint64_t output_i = get_output_i_l1(ph, pw, nc);
                stringstream ss_l1;
                ss_l1 << L1_OUTPUT_BASE_PATH << nc << "_" << ph << "_" << pw << "_" << output_i;
                string l1_path = ss_l1.str();

                FILE *fl1_read = fopen(l1_path.c_str(), "rb");
                if (fl1_read) {
                    import_lweSample_fromFile(fl1_read, l1_channel + s, in_out_params);
                    fclose(fl1_read);
                    l1_load_ok++;
                } else {
                    std::cerr << "Layer2 ERR: Cannot open L1 input: " << l1_path << endl;
                    lweNoiselessTrivial(l1_channel + s, modSwitchToTorus32(0, space_after_bs), in_out_params);
                    l1_load_fail++;
                }
            }
        }

        // Accumulate this channel's contribution into every (oc,ph,pw) this node owns
        for (int32_t oc_local = 0; oc_local < num_channels_nde; ++oc_local)
        {
            int32_t oc_global = l2_start_channel + oc_local;
            for (int32_t ph = 0; ph < pheight; ++ph) {
                for (int32_t pw = 0; pw < pwidth; ++pw) {
                    int32_t out_s = (oc_local*n_strides) + ph*pwidth + pw;
                    for (int32_t kh = 0; kh < kernel_size; ++kh) {
                        int32_t iph = ph + (kh - half_k);
                        if (iph < 0 || iph >= pheight) continue; // zero-padding
                        for (int32_t kw = 0; kw < kernel_size; ++kw) {
                            int32_t ipw = pw + (kw - half_k);
                            if (ipw < 0 || ipw >= pwidth) continue; // zero-padding
                            int32_t in_s = iph*pwidth + ipw;
                            int32_t w = get_l2_weight(oc_global, nc, kh, kw);
                            if (w != 0)
                                lweAddMulTo(conv2d_mac_sum + out_s, w, l1_channel + in_s, in_out_params);
                        }
                    }
                }
            }
        }
        std::cout << "[L2] node " << node_id << " accumulated input channel " << nc
                  << " / " << num_channels_in << "\n";
    }
    delete_LweSample_array(n_strides, l1_channel);

    std::cout << "L1 samples loaded OK: " << l1_load_ok
              << " | Failed (zero-filled): " << l1_load_fail << "\n\n";

    for (int32_t oc_local = 0; oc_local < num_channels_nde; ++oc_local) {
        int32_t oc_global = l2_start_channel + oc_local;
        Torus32 conv2D_res = lweSymDecrypt(conv2d_mac_sum + (oc_local*n_strides), user_sk, space_after_bs);
        std::cout << "Layer2 Accumulated (ch=" << oc_global << ", ph=0, pw=0, before 2X+1): "
                  << modSwitchFromTorus32(conv2D_res, space_after_bs) << endl;
    }

    //----------------------------------------------------------------
    // 2x+1 sign transform + OMP-parallel PBS / sign activation (4 threads)
    //----------------------------------------------------------------
    int n_threads = 4;
    auto t1 = high_resolution_clock::now();
    std::cout << "Layer 2 Multicore PBS/Sign Activation about to start now\n";

    for (int32_t oc_local = 0; oc_local < num_channels_nde; ++oc_local)
    {
        int32_t oc_global = l2_start_channel + oc_local;
        for (int32_t ph = 0; ph < pheight; ++ph)
        {
            for (int32_t pw = 0; pw < pwidth; pw += n_threads)
            {
                std::vector<Torus32> boos_out_tid;
                std::vector<Torus32> convoutres_tid;
                std::mutex mtx;
                #pragma omp parallel for shared(bias_sum,boos_out_tid,convoutres_tid,wrong_boos,bootstrapped_samples,conv2d_mac_sum) firstprivate(space_msg,space_after_bs,bs_key)
                for (int tid = 0; tid < n_threads; ++tid)
                {
                    int t_id = omp_get_thread_num();
                    if ((pw + t_id) >= pwidth) continue; // guard against edge

                    int32_t boos_offset = (oc_local*n_strides) + ph*pwidth + pw + t_id;

                    // 2x + 1 transform to avoid the 0-sign ambiguity (same as Layer 1)
                    lweNoiselessTrivial(bias_sum + boos_offset, mub, in_out_params);
                    lweAddMulTo(bias_sum + boos_offset, 2, conv2d_mac_sum + boos_offset, in_out_params);

                    LweSample *LWE_in = bias_sum + boos_offset;
                    LweSample *LWE_bs = bootstrapped_samples + boos_offset;

                    Torus32 convoutres;
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        convoutres = lweSymDecrypt(bias_sum + boos_offset, user_sk, space_msg);
                        convoutres_tid.push_back(convoutres);
                        std::cout << "L2 MAC(2x+1) ch=" << oc_global << " ph=" << ph
                                  << " pw=" << (pw+t_id) << " => "
                                  << modSwitchFromTorus32(convoutres, space_msg) << "\n";
                    }

                    tfhe_bootstrap_FFT(LWE_bs, bs_key, mu_boot, LWE_in);

                    Torus32 boos_out;
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        boos_out = lweSymDecrypt(bootstrapped_samples + boos_offset, user_sk, space_after_bs);
                        boos_out_tid.push_back(boos_out);
                    }
                    uint64_t output_i = get_output_i(ph, pw+t_id, oc_global);
                    std::cout << "Thread=" << t_id << " ch=" << oc_global
                              << " L2 BS output index=" << output_i << "\n";
                } // end OMP parallel
            } // end pw stride loop
        } // end ph
        std::cout << "[L2] node " << node_id << " bootstrapped output channel "
                  << oc_global << " (" << (oc_local+1) << "/" << num_channels_nde << ")\n";
    } // end oc_local

    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    duration<double, std::milli> ms_double = t2 - t1;

    //----------------------------------------------------------------
    // Validation, error logging, export to Layer 3
    //----------------------------------------------------------------
    for (int32_t oc_local = 0; oc_local < num_channels_nde; ++oc_local)
    {
        int32_t oc_global = l2_start_channel + oc_local;
        for (int32_t ph = 0; ph < pheight; ++ph)
        {
            for (int32_t pw = 0; pw < pwidth; ++pw)
            {
                int32_t samp_offset = (oc_local*n_strides) + ph*pwidth + pw;

                Torus32 convoutres = lweSymDecrypt(bias_sum + samp_offset, user_sk, space_msg);
                Torus32 boos_out   = lweSymDecrypt(bootstrapped_samples + samp_offset, user_sk, space_after_bs);
                uint64_t output_i  = get_output_i(ph, pw, oc_global);

                bool wrong =
                    (modSwitchFromTorus32(convoutres,space_msg) > (space_msg/2) && modSwitchFromTorus32(boos_out,space_after_bs)==1) ||
                    (modSwitchFromTorus32(convoutres,space_msg) > 0 && modSwitchFromTorus32(convoutres,space_msg) <= (space_msg/2) && modSwitchFromTorus32(boos_out,space_after_bs)==(space_after_bs-1)) ||
                    (modSwitchFromTorus32(convoutres,space_msg) == 0 && modSwitchFromTorus32(boos_out,space_after_bs)==(space_after_bs-1));

                if (wrong) {
                    wrong_boos++;
                    wrongbsvec.push_back(samp_offset);
                    outFile << oc_global << " " << ph << " " << pw << "\n";
                    std::cout << "L2 WRONG BS ch=" << oc_global << " ph=" << ph << " pw=" << pw << "\n";
                } else {
                    std::cout << "L2 Correct Sign/Bootstrapping ch=" << oc_global << " ph=" << ph << " pw=" << pw << "\n";
                }

                outputFile1 << oc_global << "-" << ph << "-" << pw << "-"
                            << modSwitchFromTorus32(convoutres,space_msg) << "-"
                            << modSwitchFromTorus32(boos_out,space_after_bs) << "\n";
                indexFile1  << oc_global << "-" << ph << "-" << pw << "-" << output_i << "-"
                            << modSwitchFromTorus32(convoutres,space_msg) << "-"
                            << modSwitchFromTorus32(boos_out,space_after_bs) << "\n";

                // Export this Layer-2 output sample for Layer 3
                stringstream ssio;
                ssio << L2_OUTPUT_BASE_PATH << oc_global << "_" << ph << "_" << pw << "_" << output_i;
                FILE *flwesample_write = fopen(ssio.str().c_str(), "wb");
                if (flwesample_write) {
                    export_lweSample_toFile(flwesample_write, bootstrapped_samples + samp_offset, in_out_params);
                    fclose(flwesample_write);
                } else {
                    std::cerr << "WARN: Cannot write L2 LWE sample to " << ssio.str() << "\n";
                }
            }
        }
        std::cout << "Wrong Sign/Bootstrapping Computations so far===" << wrong_boos << "\n";
    }

    outFile.close();
    outputFile1.close();
    indexFile1.close();

    int returnCode = system("/home/pi10/sendaccbatchsamples_12nodes");
    if (returnCode == 0) {
        cout << "Command for Syncing Node-wise Layer-2 Conv2D+Act Output executed\n" << endl;
    } else {
        cout << "Command execution failed or returned non-zero: " << returnCode << endl;
    }

    std::cout << "Layer2 Multicore Conv2D+PBS Duration--Node " << node_id << "=="
              << ms_int.count() << "ms\n";
    std::cout << "Layer2 Multicore Conv2D+PBS Duration--Node " << node_id << "=="
              << ms_double.count() << "ms\n";
    std::cout << "Layer2: Wrong Bootstrappings (node " << node_id << ")==" << wrong_boos << "\n";
    printf("\n");

    delete_LweSample_array(n_strides*num_channels_nde, conv2d_mac_sum);
    delete_LweSample_array(n_strides*num_channels_nde, bias_sum);
    delete_LweSample_array(n_strides*num_channels_nde, bootstrapped_samples);

    return 0;
}


void my_signal_handler(int32_t sig)
{
    cerr << "*** Received SIGint32_t *** (Stopping & finishing threads gracefully...)\n"; //  << sig
    got_sigterm = 1;
    return;
}


void writeStatistics()
{
    if (MODE_STATS)
    {
        ostringstream stats;
        stats << "\n[STAT] STATISTICS of inputs 1, 2, ..., " << count_images<< " of " << num_images << endl;
        stats << "Errors: " << count_errors << " / " << count_images << " (" << rel_error_percent << " %) 'Misclassifications'" << endl;
        stats << "Disagreements: " << count_disagreements;
        stats << " (pro-clear/pro-hom: " << count_disag_pro_clear << " / " << count_disag_pro_hom << ")" << endl;
        stats << "Wrong bootstrappings: " << count_wrong_bs << endl;
        stats << "Errors with failed bootstrapping: " << count_errors_with_failed_bs << endl;
        stats << "Disagreements with failed bootstrapping: " << count_disagreements_with_failed_bs << endl;
        stats << "Avg. time for the evaluation of the network (seconds): " << avg_time_per_classification << endl;
        stats << "Avg. time per bootstrapping (seconds): " << avg_time_per_bootstrapping << endl;
        stats << "[PROF] Bootstrapping took: " << rel_time_bs_percent << " % of the time!" << endl;

        // To Console
        string str = stats.str();
        cout << str;

        ofstream stat(FILE_STATISTICS);
        stat << str;
        stat.close();
        cout << "\n[SAVE] Wrote statistics to file:   " << FILE_STATISTICS << endl;
    }

    if (MODE_LATEX)
    {
        ostringstream latex;
        latex << "%\\input{"<<FILE_LATEX<<"}" << endl;

        latex << "% Experiments detailed" << endl;
        latex << "\\newcommand{\\EXPnumBS}{$"      <<total_num_hidden_neurons<<"$}" << endl;
        latex << "\\newcommand{\\EXPbsEXACT}{$"    <<avg_time_per_bootstrapping<<"$\\ [sec/bootstrapping]}" << endl;
        latex << "\\newcommand{\\EXPtimeEXACT}{$"  <<avg_time_per_classification<<"$\\ [sec/classification]}" << endl;

        latex << "\\newcommand{\\EXPnumERRabs}{$"  <<count_errors<<"$}" << endl;
        latex << "\\newcommand{\\EXPnumERRper}{$"  <<rel_error_percent<<"\\ \\%$}" << endl;
        latex << "\\newcommand{\\EXPwrongBSabs}{$" <<count_wrong_bs<<"$}" << endl;
        latex << "\\newcommand{\\EXPwrongDISabs}{$"<<count_disagreements_with_failed_bs<<"$}" << endl;
        latex << "\\newcommand{\\EXPdis}{$"        <<count_disagreements<<"$}" << endl;
        latex << "\\newcommand{\\EXPclear}{$"      <<count_disag_pro_clear<<"$}" << endl;
        latex << "\\newcommand{\\EXPhom}{$"        <<count_disag_pro_hom<<"$}" << endl << endl;

        latex << "\\begin{Verbatim}[frame=single,numbers=left,commandchars=+\\[\\]%" << endl;
        latex << "]" << endl;
        latex << "### Classified samples: +EXPtestset" << endl;
        latex << "Time per bootstrapping: +EXPbsEXACT" << endl;
        latex << "Errors: +EXPnumERRabs / +EXPtestset (+EXPnumERRper)" << endl;
        latex << "Disagreements: +EXPdis" << endl;
        latex << "(pro-clear/pro-hom: +EXPclear / +EXPhom)" << endl;
        latex << "Wrong bootstrappings: +EXPwrongBSabs" << endl;
        latex << "Disagreements with wrong bootstrapping: +EXPwrongDISabs" << endl;
        latex << "Avg. time for the evaluation of the network: +EXPtimeEXACT" << endl;
        latex << "\\end{Verbatim}" << endl;

        ofstream lat(FILE_LATEX);
        lat << latex.str();
        lat.close();
        cout << "[SAVE] Wrote LaTeX_result to file: " << FILE_LATEX << endl;
    }
}


void deleteMatrix(int32_t** matrix, int32_t dim_mat)
{
    for (int32_t i=0; i<dim_mat; ++i)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
}


void deleteTensor(int32_t*** tensor, int32_t dim_tensor, const int32_t* dim_vec)
{
    int32_t** matrix;
    int32_t dim_mat;
    for (int32_t i=0; i<dim_tensor; ++i)
    {
        matrix =  tensor[i];
        dim_mat = dim_vec[i];
        deleteMatrix(matrix, dim_mat);
    }
    delete[] tensor;
}


/**
 * main
 * \brief gives argc, argv, runs fully homomorphic evaluation of Neural Net and writes results.
 */
int32_t main(int32_t argc, char *argv[])
{
    time_t timestamp = time(nullptr);

    ostringstream test;
    test << asctime(localtime(&timestamp));
    if (!batched) cout << "un";
        test << "batchedMultiProcessing Testrun with call: " << argv[0] << endl;

    //string output = test.str();

    // The signal handler needs to be registered in the main.
    if (signal((int) SIGINT, my_signal_handler) == SIG_ERR)
    {
        cerr << "Registering my_signal_handler failed!"<< endl;
    }


    if (VERBOSE)
    {
        cout << test.str();
        cout << "Starting experiment to classify " << num_images;
        if (!noisyLWE) cout << " noiseless";
        cout << " encrypted MNIST images." << endl << "(";
        cout << "Execution with parameters... alpha = " << alpha << ", number of processes: " << num_processes << endl;
        // if (!VERBOSE) cout << "Print32_t unused variables: "<< torus_slices_upper_half << torus_slices_all << torus_slices_allowed << endl;
        cout << "\n*** Graceful exit on CTRL+C, exit on  CTRL+Z. ***\n\n";
    }

    processNeuralNet();

    // Some output
    test << avg_time_per_bootstrapping  << " [sec/bootstrapping]  ... Avg. time per bootstrapping (seconds)\n";
    test << avg_time_per_classification << " [sec/classification] ... Avg. time for one classification (seconds)\n\n";

    ofstream testf(FILE_TEST, ios_base::app);
    testf << test.str();
    testf.close();
    cout << "[SAVE] Appended timestamped test results: " << FILE_TEST << endl;

    return 0;
}
