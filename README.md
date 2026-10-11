# FHE-DINN: Distributed Encrypted CNN Inference

Privacy-preserving CNN inference on CIFAR-10 using TFHE, distributed across
multiple nodes. Done during my research internship at ATDC, IIT Kharagpur.

## Overview
The network runs on encrypted inputs, so the nodes never see the image in
plaintext. Conv2D layers are computed on LWE ciphertexts, and the sign
activation is applied through TFHE programmable bootstrapping (PBS). Since
bootstrapping is slow, the channels of each layer are split across nodes
and computed in parallel.

## Contents
- `layer2/master/`  Single-node reference implementation of Layer 2
  (reads Layer 1 ciphertexts, Conv2D 3x3, PBS activation, exports outputs)
- `layer2/slave/`   Worker node code with reduced memory use, processes its
  assigned block of channels and writes the output ciphertexts to disk
- `layer2/12node/`  Layer 2 version for a 12-node setup
- `orchestrator/`   C program that simulates the distributed Conv2D run
  (channel assignment per node, per-batch logs and binary output)
- `results/`        Log and output from batch 2, layer 2

## My contributions
- Adapted the base DiNN code to Layer 2 
- Master/slave split of channels across nodes
- Memory optimisation for running on low-RAM devices
- Extension to a 12-node setup
- Orchestrator for channel assignment and logging

## Credits
Base code is from "Fast Homomorphic Evaluation of Deep Discretized Neural
Networks" (Bourse, Minelli, Minihold, Paillier), built on the TFHE library.
The distributed implementation and modifications are my work.

## Build
Requires the TFHE library.
(add your compile commands here)

## Results
In the batch 2 / layer 2 run, all channels completed across the assigned
nodes (see `results/batch_2_layer_2_conv2d_log.txt`).
(add timing and memory numbers if you have them)
