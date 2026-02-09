# softmax_int8

下面是根据你提供的内容，按 **Markdown** 规范整理后的版本：

---

# OpenVX Int8 Softmax 扩展实现（支持 VIVANTE 加速）

## 功能概述

### **Int8 精度 Softmax**

* 实现 **定点 INT8 输入/输出** 的 Softmax 函数
* 适用于资源受限或需要量化推理的场景

### **OpenVX 兼容**

* 接口设计遵循 **OpenVX 标准规范**
* 可无缝集成到现有 OpenVX 图计算框架中

### **双模式支持**

1. **CPU 参考实现**

   * 纯软件版本
   * 便于验证算法正确性
2. **PPU（VIVANTE）硬件加速实现**

   * 调用 Vivante NPU/PPU 加速单元
   * 显著提升 Softmax 运算速度

---
