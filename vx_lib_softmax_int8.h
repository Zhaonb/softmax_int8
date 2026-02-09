/****************************************************************************
*
*    Copyright 2012 - 2016 Vivante Corporation, Santa Clara, California.
*    All Rights Reserved.
*
*    Permission is hereby granted, free of charge, to any person obtaining
*    a copy of this software and associated documentation files (the
*    'Software'), to deal in the Software without restriction, including
*    without limitation the rights to use, copy, modify, merge, publish,
*    distribute, sub license, and/or sell copies of the Software, and to
*    permit persons to whom the Software is furnished to do so, subject
*    to the following conditions:
*
*    The above copyright notice and this permission notice (including the
*    next paragraph) shall be included in all copies or substantial
*    portions of the Software.
*
*    THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
*    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
*    IN NO EVENT SHALL VIVANTE AND/OR ITS SUPPLIERS BE LIABLE FOR ANY
*    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
*    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
*    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
#pragma once
#ifndef _OPENVX_EXT_SOFTMAX_INT8_H_
#define _OPENVX_EXT_SOFTMAX_INT8_H_
#include <VX/vx.h>

/*! if there are more than 1 kernel in solutionï¼?
the KERNEL_ENUM_SOFTMAX_INT8_OFFSET must be modifiedï¼Œkeep different for any kernel  
*/
#define KERNEL_ENUM_SOFTMAX_INT8_OFFSET 2 



//! [KERNEL NAME]
#define VX_KERNEL_NAME_SOFTMAX_INT8 "com.vivantecorp.extension.softmax_int8VXC"
/*! \brief The Example Library Set
 * \ingroup group_example_ext
 */
#define VX_LIBRARY_SOFTMAX_INT8 (0x3) // assigned from Khronos, vendors control their own

/*! \brief The list of Example Kernels.
 * \ingroup group_xyz_ext
 */
//! [KERNEL ENUM]
enum vx_kernel_softmax_int8_ext_e {
    /*! \brief The Example Kernel */
    VX_KERNEL_ENUM_SOFTMAX_INT8 = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SOFTMAX_INT8) + KERNEL_ENUM_SOFTMAX_INT8_OFFSET,
    // up to 0xFFF kernel enums can be created.
};

#ifdef LINUX
   #define VXC_KERNEL_SOFTMAX_INT8_PATH "../../"
#else
   #define VXC_KERNEL_SOFTMAX_INT8_PATH "../"
#endif	
#ifndef gvxOBJ_CHECK
#define gvxOBJ_CHECK(ref) \
    do \
    { \
        status = vxGetStatus((vx_reference)ref); \
        if (ref == 0 || status != VX_SUCCESS) \
        { \
            printf("Obj ERROR: status=%d @ %s(%d)\n", status, __FUNCTION__, __LINE__); \
        } \
    } \
    while (0)
#endif
#ifndef gvxSTATUS_CHECK
#define gvxSTATUS_CHECK(status) \
    do \
    { \
        if (status != VX_SUCCESS) \
        { \
            printf("status ERROR: status=%d @ %s(%d)\n", status, __FUNCTION__, __LINE__); \
        } \
    } \
    while (0)
#endif

#ifdef __cplusplus
extern "C" {
#endif

//! [node]
/*! \brief [Graph] This is an example
 * in the Graph to call the example kernel.
 * \param [in] graph The handle to the graph in which to instantiate the node.
 * \param [in] input The input image.
 * \param [in] input The input scalar.
 * \param [in] input The input array.
 * \param [in] input The input tensor.
 * \param [out] output The output image.
 * \param [out] output The output scalar.
 * \param [out] output The output array.
 * \param [out] output The output tensor.
 * \ingroup group_example_kernel
 */

 static vx_node vxcsoftmax_int8Node(vx_graph graph, vx_tensor input_tensor, vx_tensor output_tensor)
{
    vx_context context = NULL;
    vx_kernel kernel = NULL;
    vx_node node = NULL;
    vx_status status = VX_FAILURE; 

    context = vxGetContext((vx_reference)graph);
    status = vxLoadKernels(context, "softmax_int8");
    gvxSTATUS_CHECK(status);
    if(status == VX_SUCCESS)
    {
    	kernel = vxGetKernelByName(context, VX_KERNEL_NAME_SOFTMAX_INT8);
    	gvxOBJ_CHECK(kernel);
    	if( kernel )
        {
    		node = vxCreateGenericNode(graph, kernel);
    		gvxOBJ_CHECK(node);
            if(node)
            {
            	vx_int32 index = 0;
                status |= vxSetParameterByIndex(node, index++, (vx_reference)input_tensor);
                status |= vxSetParameterByIndex(node, index++, (vx_reference)output_tensor);


                vx_border_t border;
                border.mode = VX_BORDER_REPLICATE;
                border.constant_value.U32 = 0;
                status |= vxSetNodeAttribute(node, VX_NODE_BORDER, &border, sizeof(border));

                gvxSTATUS_CHECK(status);
                if(status != VX_SUCCESS)
				{
					vxReleaseNode(&node);
					vxReleaseKernel(&kernel);
					return node;
				}
            }
            else
            {
                vxReleaseKernel(&kernel);
            }
        }
    }
    return node;
}
//! [node]

//! [vxu]
/*! \brief [Immediate] This is an immediate mode version of the example node.
 * \param [in] context The overall context of the implementation.
 * \param [in] input The input image.
 * \param [in] input The input scalar.
 * \param [in] input The input array.
 * \param [in] input The input tensor.
 * \param [out] output The output image.
 * \param [out] output The output scalar.
 * \param [out] output The output array.
 * \param [out] output The output tensor.
 * \ingroup group_example_kernel
 */
 static vx_status vxusoftmax_int8 (vx_context context, vx_tensor input_tensor, vx_tensor output_tensor)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    gvxOBJ_CHECK(graph);
    if (graph)
    {
        vx_node node = vxcsoftmax_int8Node(graph, input_tensor, output_tensor);
        gvxOBJ_CHECK(node);
        if (node)
        {
            status = vxVerifyGraph(graph);
            gvxSTATUS_CHECK(status);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }


    return status;
}
#ifdef __cplusplus
}
#endif
#endif
