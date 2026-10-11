/* distributed_cnn_orchestrator.c
 *
 * Simulated distributed Conv2D run.
 * Inputs:
 *   - number of layers
 *   - number of nodes
 *   - channel counts per layer
 *   - CIFAR-10 binary file path
 *
 * Output:
 *   - one text log file per batch/layer
 *   - one binary output file per batch/layer
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Section 1: Configuration                                           */
/* ------------------------------------------------------------------ */

typedef struct {
    int total_nodes;
    int num_conv_layers;
    int *channels;
    int *spatial_h;
    int *spatial_w;
    int *kernel_h;
    int *kernel_w;
} ModelConfig;

static void setup_configuration(ModelConfig *config,
                                int total_nodes,
                                int num_conv_layers,
                                int *channels,
                                int *spatial_h,
                                int *spatial_w,
                                int *kernel_h,
                                int *kernel_w) {
    config->total_nodes = total_nodes;
    config->num_conv_layers = num_conv_layers;
    config->channels = channels;
    config->spatial_h = spatial_h;
    config->spatial_w = spatial_w;
    config->kernel_h = kernel_h;
    config->kernel_w = kernel_w;
}

static void free_configuration(ModelConfig *config) {
    free(config->channels);
    free(config->spatial_h);
    free(config->spatial_w);
    free(config->kernel_h);
    free(config->kernel_w);
}

/* ------------------------------------------------------------------ */
/* Section 2: Binary file formats                                     */
/* ------------------------------------------------------------------ */

#define BIN_MAGIC 0x46484531 /* "FHE1" */

typedef struct {
    int32_t magic;
    int32_t total_nodes;
    int32_t total_layers;
    int32_t batch_id;
    int32_t layer_id;
    int32_t assigned_channels;
    int32_t total_channels;
} BinFileHeader;

typedef struct {
    int32_t batch_id;
    int32_t layer_id;
    int32_t node_id;
    int32_t channel_id;
    int32_t spatial_h;
    int32_t spatial_w;
} BinRecordHeader;

/* Checkpoint struct written after each channel's data */
typedef struct {
    int32_t batch_id;
    int32_t layer_id;
    int32_t node_id;
    int32_t channel_id;
    int32_t status;
} BinCheckpoint;

/* ------------------------------------------------------------------ */
/* Section 3: CIFAR-10 input loading                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    unsigned char label;
    unsigned char pixels[3072];
} Cifar10Record;

static int load_cifar10_record(const char *path, Cifar10Record *record) {
    FILE *fp = fopen(path, "rb");
    size_t read_bytes;

    if (!fp) {
        perror("fopen");
        return 0;
    }

    read_bytes = fread(&record->label, sizeof(unsigned char), 1, fp);
    if (read_bytes != 1) {
        fprintf(stderr, "Failed to read CIFAR-10 label from %s\n", path);
        fclose(fp);
        return 0;
    }

    read_bytes = fread(record->pixels, sizeof(unsigned char), 3072, fp);
    if (read_bytes != 3072) {
        fprintf(stderr, "Failed to read CIFAR-10 image pixels from %s\n", path);
        fclose(fp);
        return 0;
    }

    fclose(fp);
    return 1;
}

/* ------------------------------------------------------------------ */
/* Section 4: Per-node worker (writes into layer-level files)         */
/* ------------------------------------------------------------------ */

static void execute_node_chunk(int node_id,
                               int batch_id,
                               int layer_id,
                               int start_channel,
                               int local_channels,
                               int spatial_h,
                               int spatial_w,
                               int kernel_h,
                               int kernel_w,
                               const unsigned char *image_pixels,
                               unsigned char image_label,
                               FILE *text_log,
                               FILE *bin_file) {
    int channel_id, ph, pw;
    int n_strides = spatial_h * spatial_w;
    int kernel_elements = kernel_h * kernel_w;
    int32_t *conv2d_output;

    if (local_channels <= 0) return;

    conv2d_output = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n_strides * local_channels));
    if (!conv2d_output) {
        fprintf(stderr, "FATAL: allocation failure in execute_node_chunk\n");
        exit(EXIT_FAILURE);
    }

    for (channel_id = 0; channel_id < local_channels; channel_id++) {
        int global_channel_id = start_channel + channel_id;
        int local_channel_index = channel_id;
        int ip;

        for (ph = 0; ph < spatial_h; ph++) {
            for (pw = 0; pw < spatial_w; pw++) {
                int32_t acc = 0;

                for (ip = 0; ip < kernel_elements; ip++) {
                    int kh = ip / kernel_w;
                    int kw = ip % kernel_w;
                    int img_h = ph + kh;
                    int img_w = pw + kw;
                    if (img_h >= spatial_h) img_h = spatial_h - 1;
                    if (img_w >= spatial_w) img_w = spatial_w - 1;
                    int color_channel = (global_channel_id + layer_id + node_id) % 3;
                    int pixel_index = ((img_h * spatial_w + img_w) * 3 + color_channel);
                    acc += (int32_t)image_pixels[pixel_index];
                }

                acc += global_channel_id + layer_id + node_id + image_label;
                conv2d_output[(n_strides * local_channel_index) + (ph * spatial_w) + pw] = acc;
            }
        }

        fprintf(text_log,
                "batch=%d layer=%d node=%d channel=%d image_label=%u status=DONE\n",
                batch_id, layer_id, node_id, global_channel_id, image_label);
        fflush(text_log);

        {
            BinRecordHeader rec;
            int32_t out_val;
            rec.batch_id = batch_id;
            rec.layer_id = layer_id;
            rec.node_id = node_id;
            rec.channel_id = global_channel_id;
            rec.spatial_h = spatial_h;
            rec.spatial_w = spatial_w;
            fwrite(&rec, sizeof(BinRecordHeader), 1, bin_file);

            for (ph = 0; ph < spatial_h; ph++) {
                for (pw = 0; pw < spatial_w; pw++) {
                    int32_t offset = (n_strides * local_channel_index) + (ph * spatial_w) + pw;
                    out_val = conv2d_output[offset];
                    fwrite(&out_val, sizeof(int32_t), 1, bin_file);
                }
            }

            BinCheckpoint chk;
            chk.batch_id = batch_id;
            chk.layer_id = layer_id;
            chk.node_id = node_id;
            chk.channel_id = global_channel_id;
            chk.status = 1;
            fwrite(&chk, sizeof(BinCheckpoint), 1, bin_file);
            fflush(bin_file);
        }
    }

    free(conv2d_output);
}

/* ------------------------------------------------------------------ */
/* Section 5: Channel distribution                                     */
/* ------------------------------------------------------------------ */

static void distribute_channels(int total_channels,
                                int total_nodes,
                                int *node_channel_counts) {
    int base = total_channels / total_nodes;
    int remainder = total_channels % total_nodes;
    int node_id;

    for (node_id = 0; node_id < total_nodes; node_id++) {
        node_channel_counts[node_id] = base;
        if (node_id < remainder) node_channel_counts[node_id]++;
    }
}

/* ------------------------------------------------------------------ */
/* Section 6: Main scheduler (one file per batch/layer)               */
/* ------------------------------------------------------------------ */

static void simulate_cluster_execution(ModelConfig *config,
                                       const Cifar10Record *record) {
    int batch_id;
    int node_id;

    for (batch_id = 0; batch_id < config->num_conv_layers; batch_id++) {
        int layer_index = batch_id;
        int layer_id = batch_id + 1;
        int batch_num = batch_id + 1;
        int total_channels = config->channels[layer_index];
        int spatial_h = config->spatial_h[layer_index];
        int spatial_w = config->spatial_w[layer_index];
        int kernel_h = config->kernel_h[layer_index];
        int kernel_w = config->kernel_w[layer_index];
        int *node_channel_counts = (int *)malloc(sizeof(int) * config->total_nodes);
        int current_channel = 0;

        if (!node_channel_counts) {
            fprintf(stderr, "FATAL: allocation failure in simulate_cluster_execution\n");
            exit(EXIT_FAILURE);
        }

        distribute_channels(total_channels, config->total_nodes, node_channel_counts);

        printf("Batch %d (Layer %d): total_channels=%d\n", batch_num, layer_id, total_channels);
        for (node_id = 0; node_id < config->total_nodes; node_id++) {
            printf("  node %d -> %d channels\n", node_id, node_channel_counts[node_id]);
        }

        char text_path[256];
        char bin_path[256];
        snprintf(text_path, sizeof(text_path),
                 "batch_%d_layer_%d_conv2d_log.txt",
                 batch_num, layer_id);
        snprintf(bin_path, sizeof(bin_path),
                 "batch_%d_layer_%d_conv2d_data.bin",
                 batch_num, layer_id);

        FILE *text_log = fopen(text_path, "w");
        if (!text_log) {
            fprintf(stderr, "FATAL: could not open %s\n", text_path);
            free(node_channel_counts);
            exit(EXIT_FAILURE);
        }

        FILE *bin_file = fopen(bin_path, "wb");
        if (!bin_file) {
            fprintf(stderr, "FATAL: could not open %s\n", bin_path);
            fclose(text_log);
            free(node_channel_counts);
            exit(EXIT_FAILURE);
        }

        {
            BinFileHeader header;
            header.magic = BIN_MAGIC;
            header.total_nodes = (int32_t)config->total_nodes;
            header.total_layers = (int32_t)config->num_conv_layers;
            header.batch_id = (int32_t)batch_num;
            header.layer_id = (int32_t)layer_id;
            header.assigned_channels = 0;
            header.total_channels = (int32_t)total_channels;
            fwrite(&header, sizeof(BinFileHeader), 1, bin_file);
            fflush(bin_file);
        }

        fprintf(text_log,
                "batch=%d layer=%d total_channels=%d status=START image_label=%u\n",
                batch_num, layer_id, total_channels, record->label);
        fflush(text_log);

        for (node_id = 0; node_id < config->total_nodes; node_id++) {
            int start_channel = current_channel;
            int local_channels = node_channel_counts[node_id];
            current_channel += local_channels;

            fprintf(text_log,
                    "batch=%d layer=%d node=%d assigned_channels=%d start_channel=%d status=NODE_START\n",
                    batch_num, layer_id, node_id, local_channels, start_channel);
            fflush(text_log);

            execute_node_chunk(node_id,
                               batch_num,
                               layer_id,
                               start_channel,
                               local_channels,
                               spatial_h,
                               spatial_w,
                               kernel_h,
                               kernel_w,
                               record->pixels,
                               record->label,
                               text_log,
                               bin_file);

            fprintf(text_log,
                    "batch=%d layer=%d node=%d status=NODE_DONE\n",
                    batch_num, layer_id, node_id);
            fflush(text_log);
        }

        fprintf(text_log,
                "batch=%d layer=%d status=DONE\n",
                batch_num, layer_id);
        fflush(text_log);

        fclose(text_log);
        fclose(bin_file);
        free(node_channel_counts);
    }
}

/* ------------------------------------------------------------------ */
/* Section 7: User input                                              */
/* ------------------------------------------------------------------ */

int main(void) {
    ModelConfig config;
    int total_nodes = 0;
    int num_conv_layers = 0;
    int *channels = NULL;
    int *spatial_h = NULL;
    int *spatial_w = NULL;
    int *kernel_h = NULL;
    int *kernel_w = NULL;
    int i;
    Cifar10Record record;
    char image_path[512];

    printf("Enter number of layers: ");
    if (scanf("%d", &num_conv_layers) != 1 || num_conv_layers <= 0) {
        fprintf(stderr, "Invalid number of layers\n");
        return EXIT_FAILURE;
    }

    printf("Enter number of nodes: ");
    if (scanf("%d", &total_nodes) != 1 || total_nodes <= 0) {
        fprintf(stderr, "Invalid number of nodes\n");
        return EXIT_FAILURE;
    }

    channels = (int *)malloc(sizeof(int) * num_conv_layers);
    spatial_h = (int *)malloc(sizeof(int) * num_conv_layers);
    spatial_w = (int *)malloc(sizeof(int) * num_conv_layers);
    kernel_h = (int *)malloc(sizeof(int) * num_conv_layers);
    kernel_w = (int *)malloc(sizeof(int) * num_conv_layers);

    if (!channels || !spatial_h || !spatial_w || !kernel_h || !kernel_w) {
        fprintf(stderr, "FATAL: allocation failure in main\n");
        free(channels);
        free(spatial_h);
        free(spatial_w);
        free(kernel_h);
        free(kernel_w);
        return EXIT_FAILURE;
    }

    printf("Enter channel counts for %d layers: ", num_conv_layers);
    for (i = 0; i < num_conv_layers; i++) {
        if (scanf("%d", &channels[i]) != 1 || channels[i] < 0) {
            fprintf(stderr, "Invalid channel count for layer %d\n", i + 1);
            free(channels);
            free(spatial_h);
            free(spatial_w);
            free(kernel_h);
            free(kernel_w);
            return EXIT_FAILURE;
        }
    }

    printf("Enter CIFAR-10 file path: ");
    if (scanf("%511s", image_path) != 1) {
        fprintf(stderr, "Invalid CIFAR-10 file path\n");
        free(channels);
        free(spatial_h);
        free(spatial_w);
        free(kernel_h);
        free(kernel_w);
        return EXIT_FAILURE;
    }

    for (i = 0; i < num_conv_layers; i++) {
        spatial_h[i] = 32;
        spatial_w[i] = 32;
        kernel_h[i] = 3;
        kernel_w[i] = 3;
    }

    if (!load_cifar10_record(image_path, &record)) {
        free(channels);
        free(spatial_h);
        free(spatial_w);
        free(kernel_h);
        free(kernel_w);
        return EXIT_FAILURE;
    }

    setup_configuration(&config,
                        total_nodes,
                        num_conv_layers,
                        channels,
                        spatial_h,
                        spatial_w,
                        kernel_h,
                        kernel_w);

    simulate_cluster_execution(&config, &record);

    printf("Cluster simulation complete: %d layers across %d nodes.\n",
           config.num_conv_layers, config.total_nodes);
    printf("CIFAR-10 image label = %u\n", record.label);
    printf("Outputs written as:\n");
    printf("  - batch_<batch>_layer_<layer>_conv2d_log.txt\n");
    printf("  - batch_<batch>_layer_<layer>_conv2d_data.bin\n");

    free_configuration(&config);

    return 0;
}
