#ifndef VOLVOXAI_TEST_FAKE_ANDROID_NEURAL_NETWORKS_H
#define VOLVOXAI_TEST_FAKE_ANDROID_NEURAL_NETWORKS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ANeuralNetworksModel ANeuralNetworksModel;
typedef struct ANeuralNetworksCompilation ANeuralNetworksCompilation;
typedef struct ANeuralNetworksExecution ANeuralNetworksExecution;

typedef struct {
    int32_t type;
    uint32_t dimensionCount;
    const uint32_t* dimensions;
    float scale;
    int32_t zeroPoint;
} ANeuralNetworksOperandType;

enum {
    ANEURALNETWORKS_NO_ERROR = 0,
    ANEURALNETWORKS_TENSOR_FLOAT32 = 3,
    ANEURALNETWORKS_INT32 = 1,
    ANEURALNETWORKS_FUSED_NONE = 0,
    ANEURALNETWORKS_FULLY_CONNECTED = 9,
};

int ANeuralNetworks_getDeviceCount(uint32_t* numDevices);
int ANeuralNetworksModel_create(ANeuralNetworksModel** model);
void ANeuralNetworksModel_free(ANeuralNetworksModel* model);
int ANeuralNetworksModel_addOperand(
    ANeuralNetworksModel* model, const ANeuralNetworksOperandType* type);
int ANeuralNetworksModel_setOperandValue(
    ANeuralNetworksModel* model, int32_t index, const void* buffer,
    size_t length);
int ANeuralNetworksModel_addOperation(
    ANeuralNetworksModel* model, int32_t type, uint32_t inputCount,
    const uint32_t* inputs, uint32_t outputCount, const uint32_t* outputs);
int ANeuralNetworksModel_identifyInputsAndOutputs(
    ANeuralNetworksModel* model, uint32_t inputCount,
    const uint32_t* inputs, uint32_t outputCount, const uint32_t* outputs);
int ANeuralNetworksModel_finish(ANeuralNetworksModel* model);
int ANeuralNetworksCompilation_create(
    ANeuralNetworksModel* model, ANeuralNetworksCompilation** compilation);
void ANeuralNetworksCompilation_free(
    ANeuralNetworksCompilation* compilation);
int ANeuralNetworksCompilation_finish(
    ANeuralNetworksCompilation* compilation);
int ANeuralNetworksExecution_create(
    ANeuralNetworksCompilation* compilation,
    ANeuralNetworksExecution** execution);
void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution);
int ANeuralNetworksExecution_setInput(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const void* buffer,
    size_t length);
int ANeuralNetworksExecution_setOutput(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, void* buffer, size_t length);
int ANeuralNetworksExecution_compute(ANeuralNetworksExecution* execution);

#ifdef __cplusplus
}
#endif

#endif
