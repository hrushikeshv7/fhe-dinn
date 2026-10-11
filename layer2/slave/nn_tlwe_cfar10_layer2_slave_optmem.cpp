
// -------------------------------------------------------------------------------|
// Title      : Fast Homomorphic Evaluation of Deep Discretized Neural Networks
// Project    : Showcase Fast Fully Homomorphic Evaluation of Encrypted Inputs
//              using Deep Discretized Neural Networks hence preserving Privacy
// -------------------------------------------------------------------------------|
// File       : nn_tlwe_cfar10_lyer1_slave2_optmem.cpp
// Authors    : Florian Bourse      <Florian.Bourse@ens.fr>
//              Michele Minelli     <Michele.Minelli@ens.fr>
//              Matthias Minihold   <Matthias.Minihold@RUB.de>
//              Pascal Paillier     <Pascal.Paillier@cryptoexperts.com>
//
// Reference  : TFHE: Fast Fully Homomorphic Encryption Library over the Torus
//              https://github.com/tfhe
// -------------------------------------------------------------------------------|
// Description:
//     Slave Node 2 (pi3): Processes Layer-1 Conv2D channels [3*32, 4*32) = [96, 127].
//     Mirrors slave node 1 (pi2, channels [64,95]) but offset by one additional
//     block of num_channels_q1=32.  Output LWE samples are serialized to
//     /home/pi3/helloworld/ and used as accumulation inputs by the master.
// -------------------------------------------------------------------------------|
// Revisions  :
// Date        Version  Description
// 2017-11-16  0.3.0    Version for github, referenced by ePrint32_t paper
// -------------------------------------------------------------------------------|


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
//   This node already points at /home/pi3/key_user_secret.bin and
//   /home/pi3/tfhe/src/test/key_cloud.bin -- the SAME files M1569
//   generates on this same Pi. No file copy is needed for this node,
//   but GENERATE MUST be false so it doesn't overwrite M1569's key with
//   a new random one (which would desync it from every other node).
//   Make sure M1569 has been run at least once before this node runs.
// -----------------------------------------------------------------------
#define GENERATE   false /// If true: Generates fresh keys, else reuses saved ones.
#define MODE_STATS true /// If true: Writes a file with some statistics, else not.
#define MODE_LATEX true /// If true: Writes a file with some LaTeX formatted listing, else not.
#define MODE_BATCH true /// If true: Batching multiple inputs into one polynomial, else individual.
#define MODE_NOISE true /// If true: Adds Gaussian noise to samples, else trivially broken or for testing.

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
// ---------------------------------------------------------------------------
// SLAVE 2 node paths  (pi3)
// ---------------------------------------------------------------------------
#define FILE_USER_KEY       "/home/pi3/key_user_secret.bin"
//#define FILE_CLOUD_KEY      "/home/soumik/fhe-dinn/tfhe/src/test/key_cloud.bin"
#define FILE_CLOUD_KEY      "/home/pi3/tfhe/src/test/key_cloud.bin"
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
const int32_t total_num_hidden_neurons = CARD_TESTSET * NUM_NEURONS_HIDDEN;


const int32_t space_msg = MSG_SLOTS;
const int32_t space_after_bs = TORUS_SLOTS;

const int32_t space_after_l2=TORUS_SLOTS;
const int32_t space_after_n2=TORUS_SLOTS;


TFheGateBootstrappingParameterSet *our_default_bootstrapping_parameters(int32_t minimum_lambda)
{
    if ( minimum_lambda != 80 )
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
    uint32_t OutDepth=128;
    uint32_t lay_dim_hw_w=32;

    //uint32_t lay_dim_hw_w=8;
    return (((ph)*lay_dim_hw_w + pw)*OutDepth + od) ;
}


int32_t processNeuralNet()
{
    // Temporary variables
    string line;

    if (VERBOSE) cout << "[GEN] Generate parameters for a security level (according to  CGGI16a) of at least " << minimum_lambda << " [bit]." << endl;
    TFheGateBootstrappingParameterSet *params = our_default_bootstrapping_parameters(minimum_lambda);

    TFheGateBootstrappingSecretKeySet *secret;

    if(GENERATE)
    {
        // Cloud sets up keys
        if (VERBOSE) cout << "[GEN] Generate the secret keyset." << endl;
        secret = new_random_gate_bootstrapping_secret_keyset(params);

        // Save keys to file
        if (VERBOSE) cout << "[GEN] User's secret key is exported to file: " << FILE_USER_KEY << endl;
        FILE* sk = fopen(FILE_USER_KEY,"wb");
        export_tfheGateBootstrappingSecretKeySet_toFile(sk, secret);
        fclose(sk);
    }
    else
    {
        // Read keys from file
        if (VERBOSE) cout << "[READ] User's secret key is read from file: " << FILE_USER_KEY << endl;
        FILE* sk = fopen(FILE_USER_KEY,"rb");
        secret = new_tfheGateBootstrappingSecretKeySet_fromFile(sk);
        fclose(sk);
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
    delete_TGswSampleFFT(tmp);
    delete_TGswSampleFFT(tmp2);

    const LweKey* user_sk = secret->lwe_key;
    const LweParams* user_in_out_params = params->in_out_params;
    const LweParams* in_out_params = params->in_out_params;

    // Program the wheel to value(s) after Bootstrapping
    const Torus32 mu_boot = modSwitchToTorus32(1, space_after_bs);

    // Temporary variables
    Torus32 mu, phase;

    LweSample      *bootstrapped_samples;
    LweSample *bootstrapped_samples_n2, *wgt_sum_l2, *wgt_sum_l3, *bootstrapped_samples_n3;

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;


    //Execution of First Layer code

    int32_t pheight=32;
    int32_t pwidth=32;
    int32_t n_strides=pheight*pwidth;

    int32_t strides=10;
    //int32_t num_channels=10;
    int32_t num_channels=128;
    // ---------------------------------------------------------------------------
    // SLAVE 2: each slave handles num_channels_q1=32 channels.
    // Slave 1 (pi2) : channels [2*32, 3*32) = [64, 96)   (index multiplier = 2)
    // Slave 2 (pi3) : channels [3*32, 4*32) = [96, 128)  (index multiplier = 3)
    // ---------------------------------------------------------------------------
    int32_t num_channels_q1=32;   // channels per slave partition

    //Before lwe/tfhe operations first parsing the inputs

    LweSample     *enc_pixels_LWE, *conv2d_mac_sum, *mac_sum, *bias_sum;
    conv2d_mac_sum = new_LweSample_array(n_strides*num_channels_q1, in_out_params);
    bootstrapped_samples = new_LweSample_array(n_strides*num_channels_q1, in_out_params);
    bias_sum=new_LweSample_array(n_strides*num_channels_q1, in_out_params);

    int32_t wrong_boos=0;

    std::vector<int> wrongbsvec;


    // ---------------------------------------------------------------------------
    // Slave-2 specific output/error log
    // ---------------------------------------------------------------------------
    std::ofstream outFile("/home/pi3/conf_bs_errors.txt");

    // Check if file opened successfully
    if (!outFile.is_open()) {
        throw std::runtime_error("Error: Could not open file for writing.");
    }
    else
        std::cout<<"Output file opened succsessfully---";
    outFile << "Hello, Soumik!\n";

    //HEre injecting code block to process/scan layer-1 bias data from filesystem

    // ---------------------------------------------------------------------------
    // Layer-1 bias loading  (pi3 paths)
    // ---------------------------------------------------------------------------
    std::string file_path = "/home/pi3/tfhe/build/layer1_bias.txt";
    std::ifstream file_1(file_path);
    std::vector<int32_t> numbers_1;
    std::vector<int> l1_biases;

    if (file_1.is_open()) {
        std::string line;
        while (std::getline(file_1, line)) {
            std::istringstream stream(line);
            std::string word;

          while (stream >> word) {
          std::cout << ", String: " << word << std::endl;
          std::cout<<"Converting to Integer...\n";
          uint32_t number;

         if(stringToUint32(word, number)) {
             std::cout << "Converted uint32_t value: =" << number << " as well as typecasted (int) value=="<<int(number)<<"\n";
             l1_biases.push_back(int(number));
         }
         else
            std::cout<<"Out of range/invalid input tostringtoUint32 function\n";

          }
         }

    } else {
        std::cerr << "Error: Unable to open file " << file_path << std::endl;
    }

    file_1.close();

    std::cout<<"# Size l1 bias array=="<<l1_biases.size()<<endl;
    std::cout<<"Log to print l1 biases value vector";
    for (int num : l1_biases) {
        std::cout << num << " ";
    }
    std::cout<<"\n\n\n\n\n";


    std::cout<<"Bias processing for 2nd Layer..."<<endl;

    // ---------------------------------------------------------------------------
    // Layer-2 bias loading  (pi3 paths)
    // ---------------------------------------------------------------------------
    std::string file_path2 = "/home/pi3/tfhe/build/layer2_bias.txt";
    std::ifstream file_2(file_path2);
    std::vector<int32_t> numbers_2;
    std::vector<int> l2_biases;

    if (file_2.is_open()) {
        std::string line2;
        while (std::getline(file_2, line2)) {
            std::istringstream stream(line2);
            std::string word;

          while (stream >> word) {
          std::cout << ", String: " << word << std::endl;
          std::cout<<"Converting to Integer...\n";
          uint32_t numberr;

         if(stringToUint32(word, numberr)) {
             std::cout << "Converted uint32_t value: =" << numberr << " as well as typecasted (int) value=="<<int(numberr)<<"\n";
             l2_biases.push_back(int(numberr));
         }
         else
            std::cout<<"Out of range/invalid input tostringtoUint32 function\n";

          }
         }

    } else {
        std::cerr << "Error: Unable to open file " << file_path2 << std::endl;
    }

    file_2.close();

    auto maxIt = std::max_element(l2_biases.begin(), l2_biases.end());
    auto minIt = std::min_element(l2_biases.begin(), l2_biases.end());

    std::cout<<"# Size/Dimension of l2 bias array=="<<l1_biases.size()<<endl;
    std::cout<<"Log to print l2 biases value vector";
    for (int num : l2_biases) {
        std::cout << num << " ";
    }
    std::cout<<"The maximum and minimum bias values of l2_biases are resp "<<int(*maxIt)<<" and  "<<int(*minIt)<<endl;
    std::cout<<"\n\n\n\n\n";


    std::cout<<"Maximum Modulus of 2nd Layer weightsum/wixi=="<<(space_after_bs/2)<<endl;

    //Initializing Layer-1 Conv2D MAC Outputs with bias Values
    for(int p=0;p<num_channels_q1;p++)
    {
        //Initializing MAC output with layer-wise bias value...
        mu = modSwitchToTorus32(0, space_msg);
        for(int q=0;q<strides;q++){
                lweNoiselessTrivial(conv2d_mac_sum + (p*strides)+q, mu, in_out_params);
                lweNoiselessTrivial(bias_sum + (p*strides)+q, mu, in_out_params);
        }
    }

    Torus32 mub;
    mub= modSwitchToTorus32(1, space_msg);
    lweNoiselessTrivial(bias_sum, mub, in_out_params);

    int32_t lay_id=2;

    // ---------------------------------------------------------------------------
    // Output log files (slave 2 / pi3 filenames)
    // ---------------------------------------------------------------------------
    stringstream sss1;
    sss1<<"lconv_slave2_core1"<<lay_id<<"_memoptout.txt";
    string convout_path=sss1.str();
    std::ofstream outputFile1(convout_path);

    stringstream sss2;
    sss2<<"lconv_slave2_core2"<<lay_id<<"_memoptput.txt";
    string convout_path2=sss2.str();
    std::ofstream outputFile2(convout_path2);

    stringstream sss3;
    sss3<<"lconv_slave2_core3"<<lay_id<<"_memoptput.txt";
    string convout_path3=sss3.str();
    std::ofstream outputFile3(convout_path3);

    stringstream sss4;
    sss4<<"lconv_slave2_core4"<<lay_id<<"_memoptput.txt";
    string convout_path4=sss4.str();
    std::ofstream outputFile4(convout_path4);

    stringstream sss_1;
    sss_1<<"img3_lconv_slave2_core1_"<<lay_id<<"memopt_indexmap.txt";
    string convindex_path0=sss_1.str();
    std::ofstream indexFile1(convindex_path0);

    stringstream sss_2;
    sss_2<<"img3_lconv_slave2_core2_"<<lay_id<<"memopt_indexmap.txt";
    string convindex_path1=sss_2.str();
    std::ofstream indexFile2(convindex_path1);

    stringstream sss_3;
    sss_3<<"img3_lconv_slave2_core3_"<<lay_id<<"memopt_indexmap.txt";
    string convindex_path2=sss_3.str();
    std::ofstream indexFile3(convindex_path2);

    stringstream sss_4;
    sss_4<<"img3_lconv_slave2_core4_"<<lay_id<<"memopt_indexmap.txt";
    string convindex_path3=sss_4.str();
    std::ofstream indexFile4(convindex_path3);

    // ---------------------------------------------------------------------------
    // Layer-1 Conv2D MAC Accumulation
    // Slave 2 handles channels [3*num_channels_q1, 4*num_channels_q1) = [96, 128)
    //
    // The local storage index is offset by (3*num_channels_q1):
    //   ptn2_offset = n_strides*(p - 3*num_channels_q1) + ph*pheight + pw
    //
    // This mirrors slave-1 which uses (p - 2*num_channels_q1).
    // ---------------------------------------------------------------------------
    std::cout<<"About 2 run 1st Layer Conv2D Layer (Slave-2: channels 96-127)"<<"\n";
    for(int p=(3*num_channels_q1);p<(4*num_channels_q1);p++)
    {
        for(int ph=0;ph<pheight;ph++)
        {
            for(int pw=0;pw<pwidth;pw++)
            {
                std::cout<<"\n\n\n"<<"Channel-ID=="<<p<<"  and Stride Num=="<<(ph*pheight)+pw<<endl;
                mac_sum= new_LweSample(in_out_params);
                std::cout<<"Conv2D Filter/Stride-id"<<(ph*pheight)+pw<<endl;
                stringstream ss;
                std::vector<int> weights;
                std::vector<int> pixels;
                // ---------------------------------------------------------------------------
                // Pixel file path: /home/pi3/img_enc_pixels_3/cx_<lay_id>_<p>_<ph>_<pw>.txt
                // ---------------------------------------------------------------------------
                ss << "/home/pi3/img_enc_pixels_3/cx_" <<lay_id<<"_"<< p << "_"<<ph<<"_"<<pw<<".txt";
                string cx_path=ss.str();

                std::ifstream file1(cx_path);
                if (!file1.is_open()){
                    std::cerr << "Error: Could not open the file!--with file name/path==" <<cx_path<< std::endl;
                    continue;
                }
                std::string line1;
                while (std::getline(file1, line1)) {
                    std::string value;
                    std::stringstream sss1(line1);
                    while (std::getline(sss1, value, ','))
                    {
                        pixels.push_back(std::stoi(value));
                    }
                }
                file1.close();

                enc_pixels_LWE = new_LweSample_array(pixels.size(), user_in_out_params);
                for (int32_t ii = 0; ii < static_cast<int>(pixels.size()); ++ii)
                {
                     mu = modSwitchToTorus32(pixels[ii], space_msg);
                     if (noisyLWE)
                     {
                                lweSymEncrypt(enc_pixels_LWE + ii, mu, alpha, user_sk);
                     }
                     else
                     {
                                lweNoiselessTrivial(enc_pixels_LWE + ii, mu, user_in_out_params);
                     }
                }

                mu = modSwitchToTorus32(l1_biases[p], space_msg);
                std::cout<<"Bias added for this Layer MAC=="<<l1_biases[p]<<endl;
                lweNoiselessTrivial(mac_sum, mu, in_out_params);

                for (int ip=0; ip<static_cast<int>(pixels.size()); ip++)
                {
                                int32_t w=1;
                                lweAddMulTo(mac_sum, w, enc_pixels_LWE + ip, in_out_params);
                }

                std::cout << "Parsed quantized image pixels/inputs: ";
                std::cout<<"Size of parsed input from file--"<<pixels.size()<<endl;
                for (int s=0;s<static_cast<int>(pixels.size());s++) {
                    std::cout << pixels[s] << " ";
                }
                std::cout << std::endl;
                for (int weight : weights) {
                    std::cout << weight << " ";
                }
                std::cout << std::endl;

                Torus32 convout=lweSymDecrypt(mac_sum, user_sk, space_msg);
                std::cout<<"Logs 2 print/report filter-wise Conv2D output for channel-id=="<<p<<",pix-height=="<<ph<<",pix_width=="<<pw<<endl;
                printf("%d   ",modSwitchFromTorus32(convout,space_msg));

                // ---------------------------------------------------------------------------
                // Local storage offset: subtract start of slave-2's channel partition
                // Slave-1: p - 2*num_channels_q1   (channels 64-95)
                // Slave-2: p - 3*num_channels_q1   (channels 96-127)
                // ---------------------------------------------------------------------------
                int32_t ptn2_offset=(n_strides*(p-(3*num_channels_q1)))+(ph*pheight)+pw;
                lweAddTo(conv2d_mac_sum + ptn2_offset , mac_sum, in_out_params);

                Torus32 conv2D_res=lweSymDecrypt(conv2d_mac_sum + ptn2_offset, user_sk, space_msg);
                std::cout<<"SOUMIK::Just Logging Conv2D Result once before (2X+1) Transform--will then proceed to multicore bootstrapping cores\n";
                std::cout<<"Conv2D Result (X)=="<<modSwitchFromTorus32(conv2D_res,space_msg)<<endl;

                delete_LweSample_array(static_cast<int>(pixels.size()),enc_pixels_LWE);
           }
        }
    }


               //Break the loop here to start the openmp pragma/parallelization block here...

    int n_threads=4;
    auto t1 = high_resolution_clock::now();
    std::cout<<"Multicore Conv2D Execution about to start now\n";

    // ---------------------------------------------------------------------------
    // Multi-threaded bootstrapping  (Slave-2: channels [3*32, 4*32) = [96, 128))
    // ---------------------------------------------------------------------------
    for(int p=(3*num_channels_q1);p<(4*num_channels_q1);p++)
    {
        for(int ph=0;ph<pheight;ph++)
        {
            for(int pw=0;pw<(pwidth);pw=pw+n_threads)
            {

            //Trivial Initialization of Thread Variables
            std::vector<Torus32> boos_out_tid;
            std::vector<Torus32> convoutres_tid;
            std::mutex mtx;
            #pragma omp parallel for shared(bias_sum,boos_out_tid,convoutres_tid,wrong_boos,bootstrapped_samples,conv2d_mac_sum) firstprivate(space_msg,space_after_bs,bs_key)
            for (int tid=0; tid<n_threads; tid++) {

                int t_id = omp_get_thread_num();

                // Local buffer offset for this slave partition
                // Slave-1: (p - 2*num_channels_q1)
                // Slave-2: (p - 3*num_channels_q1)
                int32_t boos_offset=(n_strides*(p-(3*num_channels_q1)))+(ph*pheight)+pw+tid;

                LweSample *LWE_bs = bootstrapped_samples + boos_offset;
                lweNoiselessTrivial(bias_sum+boos_offset, mub, in_out_params);
                lweAddMulTo(bias_sum+boos_offset,2,conv2d_mac_sum + boos_offset, in_out_params);

                LweSample *LWE_in= bias_sum+boos_offset;
                Torus32 convoutres;
                {std::lock_guard<std::mutex> lock(mtx);
                convoutres=lweSymDecrypt(bias_sum+boos_offset, user_sk, space_msg);
                convoutres_tid.push_back(convoutres);
                std::cout<<"Outputting MAC result after 2x+1 transformation i.e. 2(wX)+1 to eliminate 0's=="<<modSwitchFromTorus32(convoutres,space_msg)<<"\n\n";}
                phase = lwePhase(conv2d_mac_sum + boos_offset, user_sk);
                std::cout<<"Conv2D Sample Id/Offset-id=="<<boos_offset<<endl;
                std::cout<<"SMK::DBG Entry--The lwephase computation output intermediate Conv2D outputs (BFORE Bootstrapping Invocation)-is =="<<phase<<"\n\n";
                std::cout<<"NOTE::The error term i.e. e =b-a.s-delta.m (BFORE bootstrapping)=="<<(phase-convoutres)<<endl;
                tfhe_bootstrap_FFT(LWE_bs, bs_key, mu_boot, LWE_in);
                phase = lwePhase(bootstrapped_samples+boos_offset,user_sk);
                std::cout<<"Bootstrapped Sample Id/Offset-id=="<<boos_offset<<endl;
                std::cout<<"SMK::DBG Entry--The lwephase computation output intermediate Conv2D outputs-is =="<<phase<<"\n\n";
                Torus32 boos_out;
                {std::lock_guard<std::mutex> lock(mtx);
                boos_out=lweSymDecrypt(bootstrapped_samples+boos_offset,user_sk, space_after_bs);
                boos_out_tid.push_back(boos_out);
                }
                std::cout<<"The error after bootstrapping/ebs=="<<(phase-boos_out)<<"\n\n";
                std::cout<<"Bootstrapped Sample Id/Offset-id=="<<boos_offset<<"\n";
                std::cout<<"\n\n\nTrying to print bootstrapped output/signed output for channel-id=="<<p<<"and stride=="<<(ph*pheight)+pw<<endl;
                printf("%d   ",modSwitchFromTorus32(boos_out,space_after_bs));
                std::cout<<"Cross checking with corresponding Conv2D Value=="<<modSwitchFromTorus32(convoutres,space_msg)<<"\n";
                uint64_t output_i=get_output_i(ph,pw+tid,p);
                std::cout<<"ALERT Thread-id=="<<t_id<<"--Bootstrapped Output Index=="<<output_i<<"\n\n\n\n\n";

                } //Pragma Thread Block ends here
                }
        }
    }

    // ---------------------------------------------------------------------------
    // Verification loop + LWE sample serialisation
    // Slave-2 writes to /home/pi3/helloworld/
    // (Slave-1 wrote to /home/pi2/helloworld/)
    // ---------------------------------------------------------------------------
    for(int p=(3*num_channels_q1);p<(4*num_channels_q1);p++)
    {
        for(int ph=0;ph<pheight;ph++)
        {
            for(int pw=0;pw<pwidth;pw++)
            {
                int32_t samp_offset=(n_strides*(p-(3*num_channels_q1)))+(ph*pheight)+pw;

                Torus32 convoutres;
                convoutres=lweSymDecrypt(bias_sum+samp_offset, user_sk, space_msg);
                Torus32 boos_out;
                boos_out=lweSymDecrypt(bootstrapped_samples+samp_offset,user_sk,space_after_bs);
                uint64_t output_i=get_output_i(ph,pw,p);

                if(modSwitchFromTorus32(convoutres,space_msg)>(space_msg/2) && modSwitchFromTorus32(boos_out,space_after_bs)==1)
                {
                    std::cout<<"Computed phase is -ve--WITH Wrong Bootstrapping Computation\n\n\n";
                    wrong_boos++;
                    wrongbsvec.push_back(samp_offset);
                    outputFile1<<p<<"-"<<ph<<"-"<<pw<<"-"<<modSwitchFromTorus32(boos_out,space_after_bs)<<"\n";
                    indexFile1<< p<<"-"<<ph<<"-"<<pw<<"-"<<output_i<<"-"<<modSwitchFromTorus32(convoutres,space_msg)<<"-"<<modSwitchFromTorus32(boos_out,space_after_bs)<<endl;
                    std::cout<<"Error Configuration (ch-ph-pw) format=="<<p<<"--"<<ph<<"--"<<pw<<endl;
                }
                else if(modSwitchFromTorus32(convoutres,space_msg)>0 && modSwitchFromTorus32(convoutres,space_msg)<=(space_msg/2) && modSwitchFromTorus32(boos_out,space_after_bs)==(space_after_bs-1))
                {
                    std::cout<<"Computed phase is +ve--WITH Wrong Bootstrapping Computation\n\n\n";
                    wrong_boos++;
                    wrongbsvec.push_back(samp_offset);
                    outputFile1<<p<<"-"<<ph<<"-"<<pw<<"-"<<modSwitchFromTorus32(boos_out,space_after_bs)<<"\n";
                    indexFile1<<p<<"-"<<ph<<"-"<<pw<<"-"<<output_i<<"-"<<modSwitchFromTorus32(convoutres,space_msg)<<"-"<<modSwitchFromTorus32(boos_out,space_after_bs)<<"\n";
                    std::cout<<"Error Configuration (ch-ph-pw) format=="<<p<<"--"<<ph<<"--"<<pw<<endl;
                }
                else if(modSwitchFromTorus32(convoutres,space_msg)==0 && (modSwitchFromTorus32(boos_out,space_after_bs)==(space_after_bs-1)))
                {
                    std::cout<<"Computed phase is -ve for 0 output--case report. WITH Wrong Bootstrapping Computation\n\n\n";
                    wrong_boos++;
                    wrongbsvec.push_back(samp_offset);
                    outputFile1<<p<<"-"<<ph<<"-"<<pw<<"-"<<modSwitchFromTorus32(boos_out,space_after_bs)<<"\n";
                    indexFile1<< p<<"-"<<ph<<"-"<<pw<<"-"<<output_i<<"-"<<modSwitchFromTorus32(convoutres,space_msg)<<"-"<<modSwitchFromTorus32(boos_out,space_after_bs)<<endl;
                    std::cout<<"Error Configuration (ch-ph-pw) format (Again 0-Sign Bootstrapping ErrorCase)=="<<p<<"--"<<ph<<"--"<<pw<<endl;
                }
                else
                {
                    std::cout<<"Correct Sign/Bootsrapping Computation";
                    outputFile1<<p<<"-"<<ph<<"-"<<pw<<"-"<<modSwitchFromTorus32(convoutres,space_msg)<<"-"<<modSwitchFromTorus32(boos_out,space_after_bs)<<"\n";
                    indexFile1<<p<<"-"<<ph<<"-"<<pw<<"-"<<output_i<<"-"<<modSwitchFromTorus32(convoutres,space_msg)<<"-"<<modSwitchFromTorus32(boos_out,space_after_bs)<<"\n";
                }

                std::cout<<"\n\n\n";
                std::cout<<"Wrong Sign/Boostrapping Computations done so far==="<<wrong_boos<<"\n\n\n";

                // ---------------------------------------------------------------------------
                // Serialise the bootstrapped LWE sample to filesystem.
                // Slave-1 path: /home/pi2/helloworld/image3_l1out_lwesample_<p>_<ph>_<pw>_<output_i>
                // Slave-2 path: /home/pi3/helloworld/image3_l1out_lwesample_<p>_<ph>_<pw>_<output_i>
                // ---------------------------------------------------------------------------
                stringstream ssio;
                ssio << "/home/pi3/helloworld/image3"<<"_l1out_lwesample_" << p<<"_"<<ph<<"_"<<pw<<"_"<<output_i;
                string accumsample_path=ssio.str();
                FILE *flwesample_write=fopen(accumsample_path.c_str(),"wb");
                export_lweSample_toFile(flwesample_write, bootstrapped_samples+samp_offset, in_out_params) ;
                fclose(flwesample_write);

                std::cout<<"Serialized Bootstrapped Samples/Layer-wise Output to filesystem\n";
            }
        }
    }

    outFile.close();
    outputFile1.close();
    outputFile2.close();
    outputFile3.close();
    outputFile4.close();

    auto t2 = high_resolution_clock::now();

     auto ms_int = duration_cast<milliseconds>(t2 - t1);
     duration<double, std::milli> ms_double = t2 - t1;
     std::cout <<"Multicore Conv2D Duration--Single Node=="<< ms_int.count() << "ms\n";
     std::cout << "Multicore Conv2D Duration--Single Node=="<<ms_double.count() << "ms\n";
     std::cout<<"Amount of Wrong Bootstrappings done so far in Layer-1 Conv2D (Slave-2)=="<<wrong_boos<<"\n";
     printf("\n");

    return 0;
    // Multicore Slave-2 Layer-1 Conv2D Block execution stops here


    std::cout<<"Indices of incorrect computed/wrong bootstrapped samples\n\n";
    for (int i = 0; i < static_cast<int>(wrongbsvec.size()); ++i) {
        std::cout << wrongbsvec[i] << " ";
    }
    std::cout<<"\n\n\n";


    //Code Snippet to test/verify the correctness of 2nd Layer Computation
    mu = modSwitchToTorus32(0, space_after_l2);

    int corr_bs_l2=0,wrong_bs_l2=0,corr_wwrongbs=0,wrong_wwrongbs=0,corr_wcorrbs=0,wrong_wcorrbs=0;

    int wrong_0bs_l2=0,wrong_0bs_l3=0;

    int pxheight=32,pxwidth=32, corr_bstrap=0,incorr_bstrap=0,num_channels_2=12;

    corr_bs_l2=0;wrong_bs_l2=0;
    wgt_sum_l2= new_LweSample_array(num_channels_2*pheight*pwidth,in_out_params);

    Torus32 mu_boot222=modSwitchToTorus32(1, space_after_n2);

    bootstrapped_samples_n2=new_LweSample_array(num_channels_2*pxheight*pxwidth, in_out_params);

    //Code Block for Accumulating Layer-2 wixi's values/wgtsums
    for(int nc2=0;nc2<num_channels_2;nc2++)
    {
    for(int ph=0;ph<pxheight;ph++)
    {
        for(int pw=0;pw<pxwidth;pw++)
        {
            int32_t s_offset=(nc2*pxheight*pxwidth)+ph*pxheight+pw;
            std::cout<<"Enlistting Previous Layer Items(Cleartext) Accumulated--==";
            Torus32 mu_l2=modSwitchToTorus32(l2_biases[nc2], space_after_bs);
            lweNoiselessTrivial(wgt_sum_l2+s_offset, mu_l2, in_out_params);
            for(int nc=0;nc<num_channels;nc++)
            {
                int32_t ch_offset=nc*(pxwidth)*pxheight;
                int index=ch_offset+(ph*pxheight)+pw;

                Torus32 dec_acc_inp=lweSymDecrypt(bootstrapped_samples+index,user_sk, space_after_bs);
                int bs_sign0=modSwitchFromTorus32(dec_acc_inp,space_after_bs);
                std::cout<<bs_sign0<<"  ";
                std::cout<<"Accessing Prev Laayer Cipher Index=="<<s_offset<<" and trying 2 populate next layer BS_SAmple index=="<<index<<"\n\n";

                lweAddTo(wgt_sum_l2+s_offset, bootstrapped_samples+index,in_out_params);
            }

            std::cout<<"\n\n";

            Torus32 result_conv2D=lweSymDecrypt(wgt_sum_l2+s_offset,user_sk, space_after_bs);
            printf("Conv2D Result==%d   \n\n",modSwitchFromTorus32(result_conv2D,space_after_bs));
            phase=lwePhase(wgt_sum_l2+s_offset,user_sk);

            std::cout<<"Again checking/debugging phase of 2nd layer/Layer-2 output before bootstrapping---="<<lwePhase(wgt_sum_l2+s_offset, user_sk)<<endl;
            std::cout<<"Also Checking /Logging Error for 2nd Layer neuron BFORE bootstrapping=="<<(phase-result_conv2D)<<"\n\n";
            //Block to extract sign by bootstrapping
            LweSample *LWE_sum_l2= wgt_sum_l2+s_offset;
            LweSample *LWE_boos_l2=bootstrapped_samples_n2+s_offset;
            tfhe_bootstrap_FFT(LWE_boos_l2, bs_key, mu_boot222, LWE_sum_l2);

            std::cout<<"NOW checking/debugging phase of 2nd layer/Layer-2 AFTER bootstrapping---="<<lwePhase(bootstrapped_samples_n2+s_offset, user_sk)<<endl;

            Torus32 resout_0=lweSymDecrypt(bootstrapped_samples_n2+s_offset,user_sk, space_after_n2);
            phase=lwePhase(bootstrapped_samples_n2+s_offset,user_sk);
            std::cout<<"Also Checking /Logging Ciphertext Error for 2nd Layer neuron AFTER invoking bootstrapping units=="<<(phase-resout_0)<<"\n\n";
            if(modSwitchFromTorus32(resout_0,space_after_n2)==1 && modSwitchFromTorus32(result_conv2D,space_after_bs)>(space_after_bs/2))
            {
                incorr_bstrap++;
                std::cout<<"Incorrect/WRONG Bootstrappings in 2ND Layer (so far)=="<<incorr_bstrap;
                std::cout<<"Error in Bootstrappings for -ve Conv2D outputs being reported"<<"\n\n";
            }
            else if(modSwitchFromTorus32(resout_0,space_after_n2)==(space_after_n2-1) && modSwitchFromTorus32(result_conv2D,space_after_bs)>0 && modSwitchFromTorus32(result_conv2D,space_after_bs)<=(space_after_bs/2))
            {
                incorr_bstrap++;
                std::cout<<"Incorrect/WRONG Bootstrappings in 2ND Layer (so far)=="<<incorr_bstrap;
                std::cout<<"Errors in Bootstrapping (Non-zero Case) for +ve Conv2D Outputs being Reported"<<endl;
            }
            else if(modSwitchFromTorus32(resout_0,space_after_n2)==(space_after_n2-1) && modSwitchFromTorus32(result_conv2D,space_after_bs)==0)
            {
                incorr_bstrap++;
                wrong_0bs_l2++;
                std::cout<<"Incorrect/WRONG Bootstrappings in 2ND Layer (so far)=="<<incorr_bstrap;
                std::cout<<"Logging Errors (2ND Layer):::Again 0's Sign Misclassification Error Case"<<endl;
            }
            else
                corr_bstrap++;
            std::cout<<"# of Wrong/Erroneous Bootstrappings(Sign COmputations) in 2ND Layer so far=="<<incorr_bstrap<<"\n\n";
            std::cout<<"\nDecrypted Sign of bootstrapped sample/Conv2D putput for 2nd lyer (for this neuron) is %d  for sample index=="<<s_offset<<"=="<<modSwitchFromTorus32(resout_0,space_after_n2)<<"\n\n";
        int set_flag=0;
        if (lwePhase(bootstrapped_samples_n2+s_offset, user_sk)<0 && (modSwitchFromTorus32(resout_0,space_after_n2)==1))
        {
             std::cout<<"2ND Layer Sign Bootstrapping ---Misclassification!!!";
             if(set_flag==1)
             {
                 std::cout<<"ERRORs in Previous Layr Accumulation Ciphers";
                 wrong_wwrongbs++;
             }
             else
             {
             std::cout<<"Correct cipheetexts accumulated from previous layer---no errors previous layer";
             wrong_wcorrbs++;
             }
             wrong_bs_l2++;
         }
         else if (lwePhase(bootstrapped_samples_n2+s_offset,user_sk)>=0 && (modSwitchFromTorus32(resout_0,space_after_n2)==(space_after_n2-1)))
         {
             std::cout<<"2ND Layer Sign Bootstrapping ---Misclassification!!!";
             if(set_flag==1)
             {
                 std::cout<<"ERRORs in Previous Layr Accumulation Ciphers";
                 wrong_wwrongbs++;
             }
             else
             {
                 std::cout<<"Correct cipheetexts accumulated from previous layer---no errors previous layer";
                 wrong_wcorrbs++;
             }
             wrong_bs_l2++;
         }
         else
         {
             std::cout<<"Correct Sign Computation after Bootstrapping...";
             if(set_flag==1)
             {
                 std::cout<<"ERRORs in Previous Layr Accumulation Ciphers";
                 corr_wwrongbs++;
             }
             else
             {
                 std::cout<<"Correct cipheetexts accumulated from previous layer---no errors previous layer";
                 corr_wcorrbs++;
             }
             corr_bs_l2++;
         }

         }
    }
    }

    std::cout<<"DBG::SOUMIK--Amount of CORRECT Conv2D result/Sign Computations i.e. Correct Bootstrappings in 2ND Layer=="<<corr_bstrap<<"\n\n\n";
    std::cout<<"DBG::SOUMIK--Amount of WRONG Conv2D result/Sign Computations i.e. Incorrect Bootstrappings in 2ND Layer=="<<incorr_bstrap<<"\n\n\n";
    std::cout<<"DBG::SOUMIK--Amount of WRONG Sign Computations for Conv2D Output=0 Case, in 2ND Layer=="<<wrong_0bs_l2<<"\n\n\n";

    std::cout<<"Done with status outputs of 2nd Layer Bootstrapped Result outputs\n\n\n";

    std::cout<<"Total Sign Computation Missclassifications after 2ND Layer Conv2D Activations(Bootstrapping)---Ignore=="<<wrong_bs_l2<<endl;
    std::cout<<"Total Sign Computation Correct classifications after 2ND Layer Conv2D Activations(Bootstrapping)---Ignore=="<<corr_bs_l2<<endl;
    std::cout<<"Again checking 2nd layer Conv2D output status for channel-id, pixel height, and pixel width for accumulating 128 neurons/previous layer"<<endl;
    std::cout<<"Correct Sign Bootstrapping COmputation with Previous Layer Erronerous Accumulations=="<<corr_wwrongbs<<endl;
    std::cout<<"Wrong Sign Bootstrapping Computation with Corrrect Previous Layer O/P Accumulations=="<<wrong_wcorrbs<<"\n\n";
    std::cout<<"Wrong Sign Bootstrapping Computation with Erroneous Previous Layer O/P Accumulations=="<<wrong_wwrongbs<<"\n\n";
    std::cout<<"Correct Sign Bootstrapping Computation with Correct Previous Layer O/P Accumulations=="<<corr_wcorrbs<<"\n\n";

    std::cout<<"Indices of incorrect computed/wrong bootstrapped samples\n\n";
    for (int i = 0; i < static_cast<int>(wrongbsvec.size()); ++i) {
        std::cout << wrongbsvec[i] << " ";
    }
    std::cout<<"\n\n\n";


    //Code Block for Accumulating Layer-3 wixi's values/wgtsums

    Torus32 mu_nu2=modSwitchToTorus32(0, space_after_n2);
    Torus32 mu_boot2222=modSwitchToTorus32(1, space_after_n2);
    int32_t corr_bstrap_l3=0,incorr_bstrap_l3=0;
    corr_wwrongbs=0,wrong_wwrongbs=0,corr_wcorrbs=0,wrong_wcorrbs=0;

    wgt_sum_l3= new_LweSample_array(pxheight*pxwidth,in_out_params);

    for(int ii=0;ii<pxwidth*pxheight;ii++)
        lweNoiselessTrivial(wgt_sum_l3+ii, mu_nu2, in_out_params);

    bootstrapped_samples_n3=new_LweSample_array(pxheight*pxwidth,in_out_params);

    for(int ph=0;ph<pxheight;ph++)
    {
        for(int pw=0;pw<pxwidth;pw++)
        {
            int32_t s_offset2=ph*pxheight+pw;
            std::cout<<"Enlistting Previous Layer Items(Cleartext) Accumulated--==";
            for(int nc=0;nc<num_channels_2;nc++)
            {
                int32_t ch_offset=nc*(pxwidth)*pxheight;
                int32_t index=ch_offset+s_offset2;

                int32_t dec_acc_inp=lweSymDecrypt(bootstrapped_samples_n2+index, user_sk, space_after_n2);
                int bs_sign1=modSwitchFromTorus32(dec_acc_inp,space_after_n2);
                std::cout<<bs_sign1<<"  ";

                lweAddTo(wgt_sum_l3+s_offset2, bootstrapped_samples_n2+index,in_out_params);
            }

            std::cout<<"\n\n";

            Torus32 result_conv2D_3=lweSymDecrypt(wgt_sum_l3+s_offset2, user_sk, space_after_n2);
            printf("Conv2D Result==%d   \n\n",modSwitchFromTorus32(result_conv2D_3,space_after_n2));
            phase=lwePhase(wgt_sum_l3+s_offset2, final_key);
            std::cout<<"Also Checking /Logging Error for 2nd Layer neuron BFORE bootstrapping=="<<(phase-result_conv2D_3)<<"\n\n";
            std::cout<<"Again checking/debugging phase of 3nd layer/Layer-3 output before bootstrapping---="<<lwePhase(wgt_sum_l3+s_offset2, user_sk)<<endl;
            //Block to extract sign by bootstrapping
            LweSample *LWE_sum_l3= wgt_sum_l3+s_offset2;
            LweSample *LWE_boos_l3=bootstrapped_samples_n3+s_offset2;
            tfhe_bootstrap_FFT(LWE_boos_l3, bs_key, mu_boot2222, LWE_sum_l3);

            std::cout<<"NOW checking/debugging phase of 3rd layer/Layer-3 AFTER bootstrapping---="<<lwePhase(bootstrapped_samples_n3+s_offset2, user_sk)<<endl;
            phase=lwePhase(bootstrapped_samples_n3+s_offset2,user_sk);

            Torus32 resout00=lweSymDecrypt(bootstrapped_samples_n3+s_offset2, user_sk, space_after_n2);
            std::cout<<"Also Checking /Logging Error for 2nd Layer neuron AFTER bootstrapping=="<<(phase-resout00)<<"\n\n";
            if(modSwitchFromTorus32(resout00,space_after_n2)==1 && modSwitchFromTorus32(result_conv2D_3,space_after_n2)>num_channels)
            {
                incorr_bstrap_l3++;
                std::cout<<"DBG::3RD Layer Misclassification/Sign Computation Error for -ve Conv2D outputs\n";
            }
            else if(modSwitchFromTorus32(resout00,space_after_n2)==(space_after_n2-1) && modSwitchFromTorus32(result_conv2D_3,space_after_n2)>0 && modSwitchFromTorus32(result_conv2D_3,space_after_n2)<=num_channels)
            {
                incorr_bstrap_l3++;
                std::cout<<"DBG:: 3RD Layer Sign COmputation Error for +ve Conv2D outputs"<<endl;
            }
            else if(modSwitchFromTorus32(resout00,space_after_n2)==(space_after_n2-1) && modSwitchFromTorus32(result_conv2D_3,space_after_n2)==0)
            {
                incorr_bstrap_l3++;
                std::cout<<"DBG:: 3RD Layer Sign COmputation Error---Again 0's Sign Bootstrapping Error Case..."<<endl;
                wrong_0bs_l3++;
            }
            else
                corr_bstrap_l3++;
            std::cout<<"\nDecrypted Sign of bootstrapped sample/Conv2D putput for 3rd lyer (for this neuron) is %d  for sample index=="<<s_offset2<<"=="<<modSwitchFromTorus32(resout00,space_after_n2)<<"\n\n";
        int set_flag=0;
        if (lwePhase(bootstrapped_samples_n3+s_offset2, user_sk)<0 && (modSwitchFromTorus32(resout00,space_after_n2)==1))
        {
             std::cout<<"3RD Layer Sign Bootstrapping ---Misclassification!!!";
             if(set_flag==1)
             {
                 std::cout<<"ERRORs in Previous Layr Accumulation Ciphers";
                 wrong_wwrongbs++;
             }
             else
             {
             std::cout<<"Correct cipheetexts accumulated from previous layer---no errors previous layer";
             wrong_wcorrbs++;
             }
             wrong_bs_l2++;
         }
         else if (lwePhase(bootstrapped_samples_n3+s_offset2, user_sk)>=0 && (modSwitchFromTorus32(resout00,space_after_n2)==(space_after_n2-1)))
         {
             std::cout<<"3RD Layer Sign Bootstrapping ---Misclassification!!!";
             if(set_flag==1)
             {
                 std::cout<<"ERRORs in Previous Layr Accumulation Ciphers";
                 wrong_wwrongbs++;
             }
             else
             {
                 std::cout<<"Correct cipheetexts accumulated from previous layer---no errors previous layer";
                 wrong_wcorrbs++;
             }
             wrong_bs_l2++;
         }
         else
         {
             std::cout<<"Correct Sign Computation after Bootstrapping...";
             if(set_flag==1)
             {
                 std::cout<<"ERRORs in Previous Layr Accumulation Ciphers";
                 corr_wwrongbs++;
             }
             else
             {
                 std::cout<<"Correct cipheetexts accumulated from previous layer---no errors previous layer";
                 corr_wcorrbs++;
             }
             corr_bs_l2++;
         }

         }
    }


    std::cout<<"DBG::SOUMIK--Amount of CORRECT Conv2D result/Sign Computations (2ND Layer)=="<<corr_bstrap<<"\n\n\n";
    std::cout<<"DBG::SOUMIK--Amount of WRONG Conv2D result/Sign Computations== (2ND Layer)"<<incorr_bstrap<<"\n\n\n";

    std::cout<<"DBG::SOUMIK--Amount of CORRECT Conv2D result/Sign Computations (3rd Layer)=="<<corr_bstrap_l3<<"\n\n\n";
    std::cout<<"DBG::SOUMIK--Amount of WRONG Conv2D result/Sign Computations== (3rd Layer)"<<incorr_bstrap_l3<<"\n\n\n";
    std::cout<<"DBG::SOUMIK--Amount of WRONG Sign Computations for Conv2D Output=0 Error Case, in 3RD Layer=="<<wrong_0bs_l3<<"\n\n\n";

    std::cout<<"Multicore Implementation:::Amount of Wrong Sign/Boostrapping Computations done so far in (1ST Layer)-==="<<wrong_boos;
    std::cout <<"Multicore 1st LayerConv2D Duration--Single Node (Slave-2)=="<< ms_int.count() << "ms\n";
    std::cout << "Multicore 1st Layer Conv2D Duration--Single Node (Slave-2)=="<<ms_double.count()<< "ms\n";
    printf("\n");

    std::cout<<"#/Amount of Correct Bootstrappings/SIgn Computations=="<<corr_bs_l2<<"\n\n";
    std::cout<<"#/Amount of Wrong Bootstrappings/SIgn Computations=="<<wrong_bs_l2<<"\n\n\n\n\n";

    return 0;
}


void my_signal_handler(int32_t sig)
{
    cerr << "*** Received SIGint32_t *** (Stopping & finishing threads gracefully...)\n";
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
        cout << "\n*** Graceful exit on CTRL+C, exit on  CTRL+Z. ***\n\n";
    }

    processNeuralNet();

    test << avg_time_per_bootstrapping  << " [sec/bootstrapping]  ... Avg. time per bootstrapping (seconds)\n";
    test << avg_time_per_classification << " [sec/classification] ... Avg. time for one classification (seconds)\n\n";

    ofstream testf(FILE_TEST, ios_base::app);
    testf << test.str();
    testf.close();
    cout << "[SAVE] Appended timestamped test results: " << FILE_TEST << endl;

    return 0;
}
