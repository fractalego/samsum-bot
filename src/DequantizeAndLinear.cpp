#include "DequantizeAndLinear.h"
#include <algorithm>
#include <assert.h>
#include <cstring>
#include <cudnn.h>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::DequantizeAndLinear;
using nvinfer1::plugin::DequantizeAndLinearPluginCreator;

static const char* DequantizeAndLinear_PLUGIN_VERSION{"1"};
static const char* DequantizeAndLinear_PLUGIN_NAME{"DequantizeAndLinear"};

PluginFieldCollection DequantizeAndLinearPluginCreator::mFC{};
std::vector<PluginField> DequantizeAndLinearPluginCreator::mPluginAttributes;


DequantizeAndLinear::DequantizeAndLinear(const void* data, size_t length)
{
}

DequantizeAndLinear::~DequantizeAndLinear()
{
}

int DequantizeAndLinear::getNbOutputs() const noexcept
{
    return 1;
}

Dims DequantizeAndLinear::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    return DimsCHW(1, 1, 1);
}

int DequantizeAndLinear::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void DequantizeAndLinear::terminate() noexcept
{
    LOG_ERROR(cublasDestroy(mCublas));
}

size_t DequantizeAndLinear::getWorkspaceSize(int) const noexcept
{
    return 0;
}

int DequantizeAndLinear::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) noexcept
{
    return 0;
}

size_t DequantizeAndLinear::getSerializationSize() const noexcept
{
    return sizeof(bool) + sizeof(int) * (3 + mNumInputs) + sizeof(nvinfer1::Dims) + (sizeof(mCopySize) * mNumInputs);
}

void DequantizeAndLinear::serialize(void* buffer) const noexcept
{
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void DequantizeAndLinear::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void DequantizeAndLinear::detachFromContext() noexcept {}

// Return true if output tensor is broadcast across a batch.
bool DequantizeAndLinear::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool DequantizeAndLinear::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

// Set plugin namespace
void DequantizeAndLinear::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const char* DequantizeAndLinear::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
DataType DequantizeAndLinear::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return DataType::kFLOAT;
}

void DequantizeAndLinear::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    assert(nbOutputs == 1);
    mCHW = inputDims[0];
    mNumInputs = nbInputs;
    assert(inputDims[0].nbDims == 3);

    if (mInputConcatAxis == nullptr)
    {
        LOG_ERROR(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
    }

    for (int i = 0; i < nbInputs; ++i)
    {
        int flattenInput = 0;
        assert(inputDims[i].nbDims == 3);
        if (mConcatAxisID != 1)
        {
            assert(inputDims[i].d[0] == inputDims[0].d[0]);
        }
        if (mConcatAxisID != 2)
        {
            assert(inputDims[i].d[1] == inputDims[0].d[1]);
        }
        if (mConcatAxisID != 3)
        {
            assert(inputDims[i].d[2] == inputDims[0].d[2]);
        }
        flattenInput = inputDims[i].d[0] * inputDims[i].d[1] * inputDims[i].d[2];
        mInputConcatAxis[i] = flattenInput;
        mOutputConcatAxis += mInputConcatAxis[i];
    }

    for (int i = 0; i < nbInputs; ++i)
    {
        mCopySize[i] = inputDims[i].d[0] * inputDims[i].d[1] * inputDims[i].d[2] * sizeof(float);
    }
}

bool DequantizeAndLinear::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}
const char* DequantizeAndLinear::getPluginType() const noexcept
{
    return "DequantizeAndLinear";
}

const char* DequantizeAndLinear::getPluginVersion() const noexcept
{
    return "1";
}

void DequantizeAndLinear::destroy() noexcept
{
    delete this;
}

IPluginV2Ext* DequantizeAndLinear::clone() const noexcept
{
    auto* plugin = new DequantizeAndLinear();
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

DequantizeAndLinearPluginCreator::DequantizeAndLinearPluginCreator()
{
}

const char* DequantizeAndLinearPluginCreator::getPluginName() const noexcept
{
    return DequantizeAndLinear_PLUGIN_NAME;
}

const char* DequantizeAndLinearPluginCreator::getPluginVersion() const noexcept
{
    return DequantizeAndLinear_PLUGIN_VERSION;
}

const PluginFieldCollection* DequantizeAndLinearPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* DequantizeAndLinearPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    auto* plugin = new DequantizeAndLinear();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* DequantizeAndLinearPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call Concat::destroy()
    IPluginV2Ext* plugin = new DequantizeAndLinear();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

REGISTER_TENSORRT_PLUGIN(DequantizeAndLinearPluginCreator);