#ifndef TRT_DequantizeAndLinear_PLUGIN_H
#define TRT_DequantizeAndLinear_PLUGIN_H

#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include <cstdlib>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <string>
#include <vector>

#define LOG_ERROR(status)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cout << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

namespace nvinfer1
{
namespace plugin
{
class DequantizeAndLinear : public IPluginV2Ext
{
public:
    DequantizeAndLinear() {};

    DequantizeAndLinear(const void* data, size_t length);

    ~DequantizeAndLinear();

    int getNbOutputs() const noexcept;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept;

    int initialize() noexcept;

    void terminate() noexcept;

    size_t getWorkspaceSize(int) const noexcept;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) noexcept;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept;

    size_t getSerializationSize() const noexcept;

    void serialize(void* buffer) const noexcept;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept;

    void detachFromContext() noexcept;

    const char* getPluginType() const noexcept;

    const char* getPluginVersion() const noexcept;

    void destroy() noexcept;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept;
    IPluginV2Ext* clone() const noexcept;

    void setPluginNamespace(const char* pluginNamespace) noexcept;

    const char* getPluginNamespace() const noexcept;

private:
    Weights copyToDevice(const void* hostData, size_t count);

    void serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const;

    Weights deserializeToDevice(const char*& hostBuffer, size_t count);

    size_t* mCopySize = nullptr;
    bool mIgnoreBatch{false};
    int mConcatAxisID{0}, mOutputConcatAxis{0}, mNumInputs{0};
    int* mInputConcatAxis = nullptr;
    nvinfer1::Dims mCHW;
    const char* mPluginNamespace;
    cublasHandle_t mCublas;
};

class DequantizeAndLinearPluginCreator : public IPluginCreator
{
public:
    DequantizeAndLinearPluginCreator();

    ~DequantizeAndLinearPluginCreator() override = default;

    const char* getPluginName() const noexcept;

    const char* getPluginVersion() const noexcept;

    const PluginFieldCollection* getFieldNames() noexcept;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept;

private:
    static PluginFieldCollection mFC;
    bool mIgnoreBatch{false};
    int mConcatAxisID;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_DequantizeAndLinear_PLUGIN_H