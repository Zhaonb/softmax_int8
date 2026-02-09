/*
 ============================================================================
 Name        : softmax_int8.c
 Author      : vip
 Version     :
 Copyright   : Your copyright notice
 Description : 
 ============================================================================
 */
#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_helper.h>
#include <VX/vx_ext_program.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_cnn.h>
#include <VX/viv_nn_compatibility.h>
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include "vx_lib_softmax_int8.h"



//#define USE_SOFTMAX_INT8_VXC
//#define USE_OFFLINE_COMPILER
vx_status VX_CALLBACK vxsoftmax_int8Validator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;
    vx_uint32 index = 0;
#ifndef UINT16_MAX
   #define UINT16_MAX       ((unsigned short)0xffff)
#endif
    for(index = 0; index < num; index++)
    {
    // Validator
        if(index == 0) //tensor
        {
            vx_tensor tensor = (vx_tensor)parameters[index];
            if(tensor != NULL)
            {
                vx_uint32     num_of_dim;
                vx_enum       data_format;

                // num_of_dim
                status |= vxQueryTensor(tensor, VX_TENSOR_NUM_OF_DIMS, &num_of_dim, sizeof(num_of_dim));
                if (num_of_dim<2 || num_of_dim > 3)
                {
                    printf("[%s : %d] Warning num_of_dim<2 || num_of_dim > 3 \n", __FILE__,__LINE__);
                }
                // data_format
                status |= vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &data_format, sizeof(data_format));
                if (data_format != VX_TYPE_INT8)                    
                    status |= VX_ERROR_INVALID_TYPE;

            }
            else
            {
                printf("[%s : %d] Warning parameter %d is NULL \n", __FILE__,__LINE__,index);
            }       

        }
        else if(index == 1) //tensor
        {
            vx_tensor tensor = (vx_tensor)parameters[index];
            if(tensor != NULL)
            {
                    vx_uint32    num_of_dim;
                    vx_enum      data_format;

                    // num_of_dim
                    status |= vxQueryTensor(tensor, VX_TENSOR_NUM_OF_DIMS, &num_of_dim, sizeof(num_of_dim));
                    if (num_of_dim<2 || num_of_dim > 3)
                    {
                        printf("[%s : %d] Warning num_of_dim<2 || num_of_dim > 3 \n", __FILE__,__LINE__);
                    }
                    // data_format
                    status |= vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &data_format, sizeof(data_format));
                    if (data_format != VX_TYPE_INT8)
                        status |= VX_ERROR_INVALID_TYPE;

            }
            else
            {
                printf("[%s : %d] Warning parameter %d is NULL \n", __FILE__,__LINE__,index);
            }   
        }

        else
        {
            printf("[%s : %d] Validator  failure! invalid index = %d\n", __FILE__,__LINE__,index);  
        }

        if(status < 0)
        {
            printf("[%s : %d] Validator  failure! index = %d\n",__FILE__, __LINE__,index);  
            break;
        }       
    }
    return status;
}


#ifndef USE_SOFTMAX_INT8_VXC
#ifdef WIN32_
static vx_float32 round(vx_float32 x)
{
#if defined(_M_X64)
    return (vx_float32) _copysignf(floorf(fabsf(x) + 0.5f), x);
#else
    return (vx_float32) _copysign(floorf(fabsf(x) + 0.5f), x);
#endif
}
#endif
vx_int8 Fp32toInt8(vx_float32 val, vx_int8 fixedPointPos)
{
    vx_int8 result = 0;

    if (fixedPointPos > 0)
    {
        vx_int32 data = (vx_int32) round(val * (vx_float32)(1 << fixedPointPos));
        result = (vx_int8)((data > 127) ? 127 : (data < -128) ? -128 : data);

    }
    else
    {
        vx_int32 data = (vx_int32) round(val * (1.0f / (vx_float32)(1 << -fixedPointPos)));
        result = (vx_int8)((data > 127) ? 127 : (data < -128) ? -128 : data);
    }

    return result;
}

vx_float32 Int8toFp32(vx_int8 val, vx_int8 fixedPointPos)
{
    vx_float32 result = 0.0f;

    if (fixedPointPos > 0)
    {
        result = (vx_float32)val * (1.0f / ((vx_uint32) (1 << fixedPointPos)));
    }
    else
    {
        result = (vx_float32)val * ((vx_float32) (1 << -fixedPointPos));
    }

    return result;
}

static vx_status get_stride_size(vx_enum data_format, vx_uint32 *stride)
{
	vx_status status = VX_FAILURE;

	switch(data_format)
	{
	case VX_TYPE_INT8:
	case VX_TYPE_UINT8:
		stride[0] = 1;
		status = VX_SUCCESS;
		break;
	case VX_TYPE_INT16:
	case VX_TYPE_UINT16:
	case VX_TYPE_FLOAT16:
		stride[0] = 2;
		status = VX_SUCCESS;
		break;
	case VX_TYPE_INT32:
	case VX_TYPE_UINT32:
	case VX_TYPE_FLOAT32:
		stride[0] = 4;
		status = VX_SUCCESS;
		break;
	default:
		printf("can not support this type: %d\n", data_format);
		break;
	}

	return status;
}
vx_status VX_CALLBACK vxsoftmax_int8Kernel(vx_node node, const vx_reference* paramObj, vx_uint32 paramNum)
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;

    if(paramNum == 2)
    {
        status = VX_SUCCESS;
        // tensor
        void* TensorData[2] = { NULL, NULL };
        vx_tensor tensorObj[2] = { NULL, NULL };
        vx_uint32 num_of_dim = 0, input_num = 0, i;
        vx_uint32 input_size[4] = {1, 1, 1, 1};
        vx_uint32 input_stride_size[4] = {1, 1, 1, 1};
        vx_enum   data_format[2] = {0, 0};
        vx_int8   fl[2] = {0, 0};
        vx_tensor_addressing input_user_addr[2] = {NULL, NULL};

        vx_context context = vxGetContext((vx_reference)node);

        // input tensor
        tensorObj[0] = (vx_tensor)paramObj[0];
        status |= vxQueryTensor(tensorObj[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dim, sizeof(num_of_dim));
        status |= vxQueryTensor(tensorObj[0], VX_TENSOR_DATA_TYPE, &data_format[0], sizeof(data_format[0]));
        status |= vxQueryTensor(tensorObj[0], VX_TENSOR_DIMS, input_size, sizeof(input_size));
        status |= vxQueryTensor(tensorObj[0], VX_TENSOR_FIXED_POINT_POSITION, &fl[0], sizeof(fl[0]));
        status |= get_stride_size(data_format[0], &input_stride_size[0]);

        input_num = input_size[0] * input_size[1] * input_size[2];
        for (i=1; i<num_of_dim; i++)
            input_stride_size[i] = input_stride_size[i-1] * input_size[i];
        input_user_addr[0] = vxCreateTensorAddressing(context, input_size, input_stride_size, num_of_dim);
        TensorData[0] = (void *)malloc(input_num * input_stride_size[0]);
        status |= vxCopyTensorPatch(tensorObj[0], NULL, input_user_addr[0],  TensorData[0], VX_READ_ONLY, 0);
        vx_float32 *input_data_fp32 = (vx_float32 *)malloc(input_num * sizeof(vx_float32));
        vx_float32 *output_data_fp32 = (vx_float32 *)malloc(input_num * sizeof(vx_float32));
		vx_int8 *output_data_int8 = (vx_int8 *)malloc(input_num * sizeof(vx_int8));

        vx_float32 fmax = 0.f;
        for (i=0; i<input_num; i++)
        {
        	input_data_fp32[i] = Int8toFp32(((vx_int8*)(TensorData[0]))[i], fl[0]);
        	fmax = input_data_fp32[i] > fmax ? input_data_fp32[i] : fmax;
        }
        vx_float32  fSum = 0.0f;
        for (i=0; i<input_num; i++)
        {
        	output_data_fp32[i] = expf(input_data_fp32[i] - fmax);
        	fSum += output_data_fp32[i];
        }
        for (i=0; i<input_num; i++)
        {
        	output_data_fp32[i] = output_data_fp32[i] / fSum;
        }
        status |= vxReleaseTensorAddressing(&input_user_addr[0]);

        // output tensor
        tensorObj[1] = (vx_tensor)paramObj[1];
		input_size[0] = 1;input_size[1] = 1;input_size[2] = 1;input_size[3] = 1;
		input_stride_size[0] = 1;input_stride_size[1] = 1;input_stride_size[2] = 1;input_stride_size[3] = 1;
		
        status |= vxQueryTensor(tensorObj[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dim, sizeof(num_of_dim));
        status |= vxQueryTensor(tensorObj[1], VX_TENSOR_DATA_TYPE, &data_format[1], sizeof(data_format[1]));
        status |= vxQueryTensor(tensorObj[1], VX_TENSOR_DIMS, input_size, sizeof(input_size));
        status |= vxQueryTensor(tensorObj[1], VX_TENSOR_FIXED_POINT_POSITION, &fl[1], sizeof(fl[1]));
        status |= get_stride_size(data_format[1], &input_stride_size[0]);

        input_num = input_size[0] * input_size[1] * input_size[2];
        for (i=1; i<num_of_dim; i++)
            input_stride_size[i] = input_stride_size[i-1] * input_size[i];
        input_user_addr[1] = vxCreateTensorAddressing(context, input_size, input_stride_size, num_of_dim);
		
		// convert fp32 to int8
        for (i=0; i<input_num; i++)
        {
        	output_data_int8[i] = Fp32toInt8(output_data_fp32[i], fl[1]);
		}	
		status |= vxCopyTensorPatch(tensorObj[1], NULL, input_user_addr[1], output_data_int8, VX_WRITE_ONLY, 0);
		status |= vxReleaseTensorAddressing(&input_user_addr[1]);
		

        free(input_data_fp32);
        free(output_data_fp32);
		free(output_data_int8);
    }

    return status;
}
#endif

vx_status VX_CALLBACK vxsoftmax_int8Initializer(vx_node nodObj, const vx_reference *paramObj, vx_uint32 paraNum)
{
    vx_status status = VX_SUCCESS;
#ifdef USE_SOFTMAX_INT8_VXC
// Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_image imgObj = (vx_image)paramObj[0];
    vx_int32 imgWid = 0;
    vx_int32 imgHei = 0;
    status |= vxQueryImage(imgObj, VX_IMAGE_WIDTH, &imgWid, sizeof(vx_uint32));
    status |= vxQueryImage(imgObj, VX_IMAGE_HEIGHT, &imgHei, sizeof(vx_uint32));

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkScale[0]  = 16;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.localWorkSize[0]    = 8;
    shaderParam.localWorkSize[1]    = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((imgWid + shaderParam.globalWorkScale[0] - 1) / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
    shaderParam.globalWorkSize[1]   = gcmALIGN((imgHei + shaderParam.globalWorkScale[1] - 1) / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
    /*
    {
        // uniforms
        vx_uint32 uni4x8_A[16] = {
            0x9a291606, 0x9a291606, // TCfg
            0xe2200242, 0x30524304, 0x0284a4e4, 0x85056640, 0xb568505a, // BinSelect
            0x00007400, // AccumType, ConstantType, and PostShift
            0x00000402, 0x00040403, 0x00020201, 0x02080202, 0x00000402, 0x00040403, 0x00020201, 0x02080202 // Constant
        };
        status |= vxSetNodeUniform(nodObj, "uni_A", 1, uni4x8_A);
    }
    */
    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS, &shaderParam, sizeof(vx_kernel_execution_parameters_t));
#endif
    if(status < 0)
    {
            printf("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);     
    }
    return status;
}

vx_status VX_CALLBACK vxsoftmax_int8Deinitializer(vx_node nodObj, const vx_reference *paraObj, vx_uint32 paraNum)
{
    return VX_SUCCESS;
}

static vx_param_description_t vxsoftmax_int8KernelParam[] = 
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};

vx_kernel_description_t vxsoftmax_int8KernelInfo = 
{
    VX_KERNEL_ENUM_SOFTMAX_INT8,
    VX_KERNEL_NAME_SOFTMAX_INT8,
#ifdef USE_SOFTMAX_INT8_VXC
    NULL,
#else
    vxsoftmax_int8Kernel,
#endif
    vxsoftmax_int8KernelParam,
    (sizeof(vxsoftmax_int8KernelParam) / sizeof(vxsoftmax_int8KernelParam[0])),
    vxsoftmax_int8Validator,
    NULL,
    NULL,
    vxsoftmax_int8Initializer,
    vxsoftmax_int8Deinitializer
};

#ifdef USE_SOFTMAX_INT8_VXC
vx_char *programSource = NULL;
static vx_char* loadSources(const vx_char *filename, vx_size *programSize)
{
    FILE *pFile = NULL;

    pFile = fopen(filename, "rb");
    if (pFile != NULL && programSize)
    {
        vx_int32 size = 0;
        /* obtain file size:*/
        fseek(pFile, 0, SEEK_END);
        *programSize = ftell(pFile);
        rewind(pFile);

        size = (int)(*programSize + 1);
        programSource = (char*)malloc(sizeof(char)*(size));
        if (programSource == NULL)
        {
            fclose(pFile);
            free(programSource);
            return NULL;
        }

        fread(programSource, sizeof(char), *programSize, pFile);
        programSource[*programSize] = '\0';
        fclose(pFile);
    }
    else
    {
        printf("[%s : %d] Open %s failed.\n", __FILE__, __LINE__, filename);
    }

    return programSource;
}

vx_uint8 *binaryBuf = NULL;
vx_size vxLoadVxcBinarySource(vx_char *binaryFile)
{
    vx_size programLength = 0;
    int count = 0;
    printf("Creating program with binary file...\n");
    FILE *fp = fopen(binaryFile, "rb");
    if(!fp)
    {
        printf("Cannot open binary program file.\n");
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    programLength = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    binaryBuf = (vx_uint8 *) malloc(programLength + 1);
    count = fread(binaryBuf, 1, programLength, fp);
    binaryBuf[count] = '\0';
    if (fp)
        fclose(fp);
    return programLength;
}
#endif

static vx_kernel_description_t* kernels[] = 
{
    &vxsoftmax_int8KernelInfo
};

#ifdef __cplusplus
extern "C"
#endif
    vx_status VX_API_CALL vxPublishKernels(vx_context ContextVX)
{
    vx_status status = VX_FAILURE;
    vx_kernel kernelObj = NULL;

#ifdef USE_SOFTMAX_INT8_VXC
    vx_program programObj = NULL;

    const vx_char* programSrc[1];
    vx_size programLen = 0;
    unsigned int i = 0, j = 0;
#ifndef USE_OFFLINE_COMPILER
    programSrc[0] = loadSources(VXC_KERNEL_SOFTMAX_INT8_PATH"softmax_int8/softmax_int8.vx", &programLen); // +
    
    programObj = vxCreateProgramWithSource(ContextVX, 1, programSrc, &programLen);
#else
    programLen = vxLoadVxcBinarySource((vx_char *)VXC_KERNEL_ADDER_PATH"Adder/Adder.vxgcSL");

         if(programLen <= 0)
         {
         printf("[%s : %d]  vxLoadVxcBinarySource() Error!\n", __FILE__, __LINE__);
                   return VX_FAILURE;
         }
         programObj = vxCreateProgramWithBinary(ContextVX, binaryBuf, programLen);
         if(binaryBuf)
                   free(binaryBuf);
#endif
    status = vxBuildProgram(programObj, "-cl-viv-vx-extension");
    if(VX_SUCCESS != status)
    {
        printf("[%s : %d] vxBuildProgram() Error!\n", __FILE__, __LINE__);
        return status;
    }
    if(programSource != NULL)
    {
        free(programSource);
        programSource = NULL;
        programSrc[0] = NULL;
    }
    for(i=0; i<(sizeof(kernels)/sizeof(kernels[0])); i++)
    {
        kernelObj = vxAddKernelInProgram(programObj,
            kernels[i]->name,
            kernels[i]->enumeration,
            kernels[i]->numParams,
            kernels[i]->validate,         
            kernels[i]->initialize,
            kernels[i]->deinitialize
            );
        if(kernelObj)
        {
            status = VX_SUCCESS; 
            for(j=0; j < kernels[i]->numParams; j++)
            {
                status = vxAddParameterToKernel(kernelObj, 
                    j,
                    kernels[i]->parameters[j].direction,
                    kernels[i]->parameters[j].data_type,
                    kernels[i]->parameters[j].state
                    );
                if(VX_SUCCESS != status)
                {
                    printf("[%s : %d] Failed to add parameter %d to kernel %s.\n", __FILE__, __LINE__, j, kernels[i]->name);
                    break;
                }
            }

            if(VX_SUCCESS == status)
            {
                status = vxFinalizeKernel(kernelObj);
                if(VX_SUCCESS != status)
                {
                    printf("[%s : %d] Failed to finalize kernel[%u]=%s\n", __FILE__, __LINE__, i, kernels[i]->name);

                    status = vxRemoveKernel(kernelObj);
                    if(VX_SUCCESS != status)
                        printf("[%s : %d] Failed to remove kernel[%u]=%s\n", __FILE__, __LINE__, i, kernels[i]->name);
                }
            }
            else
            {
                status = vxRemoveKernel(kernelObj);
                if(VX_SUCCESS != status)
                    printf("[%s : %d] Failed to remove kernel[%u]=%s\n", __FILE__, __LINE__, i, kernels[i]->name);
            }
        }
        else
        {
            printf("[%s : %d] Failed to add kernel %s\n", __FILE__, __LINE__, kernels[i]->name);
        }
    }
#else
    for(unsigned int i=0; i<(sizeof(kernels)/sizeof(kernels[0])); i++)
    {
        kernelObj = vxAddUserKernel(ContextVX,
            kernels[i]->name,
            kernels[i]->enumeration,
            kernels[i]->function,
            kernels[i]->numParams,
            kernels[i]->validate,          
            kernels[i]->initialize,
            kernels[i]->deinitialize
            );
        if(kernelObj)
        {
            status = VX_SUCCESS; 
            for(unsigned int j=0; j<kernels[i]->numParams; j++)
            {
                status |= vxAddParameterToKernel(kernelObj,
                    j,
                    kernels[i]->parameters[j].direction,
                    kernels[i]->parameters[j].data_type,
                    kernels[i]->parameters[j].state
                    );
                if(VX_SUCCESS != status)
                {
                    printf("[%s : %d] Failed to add parameter %d to kernel %s.\n", __FILE__, __LINE__, j, kernels[i]->name);
                    break;
                }
            }

            if(VX_SUCCESS == status)
            {
                status |= vxFinalizeKernel(kernelObj);
                if(VX_SUCCESS != status)
                {
                    printf("[%s : %d] Failed to finalize kernel[%u]=%s\n", __FILE__, __LINE__, i, kernels[i]->name);

                    status |= vxRemoveKernel(kernelObj);
                    if(VX_SUCCESS != status)
                        printf("[%s : %d] Failed to remove kernel[%u]=%s\n", __FILE__, __LINE__, i, kernels[i]->name);
                }
            }
            else
            {
                status |= vxRemoveKernel(kernelObj);
                if(VX_SUCCESS != status)
                    printf("[%s : %d] Failed to remove kernel[%u]=%s\n", __FILE__, __LINE__, i, kernels[i]->name);
            }
        }
        else
        {
            printf("[%s : %d] Failed to add kernel %s\n", __FILE__, __LINE__, kernels[i]->name);
        }
    }
#endif
    return status;
}
