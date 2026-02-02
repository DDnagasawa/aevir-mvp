核心功能范畴 (Scope)
L1 边缘节点的核心功能：
* 节点管理：节点注册、心跳维持、任务获取、梯度上传及工作量证明（Proof）。
* 本地训练引擎：集成 Transformer 与 LoRA 。
* 资源监控与治理：监控 CPU/内存/GPU 指标，并执行。
* 数据沙箱机制：文件路径隔离、操作审计日志记录，以及待处理队列的持久化存储。
* 退避重试机制（Backoff Retry）、任务轮询及离线恢复能力。支持 Flower 客户端模式运行，具备 L3 链上提交的扩展性。

当前进展
* L1 本地流程闭环：沙箱初始化、资源前置校验、模型训练及证明生成流程均
* 训练： Transformer + LoRA 架构下的反向传播与梯度采集功能。
* 数据：审计日志与待上传队列实现本地沙箱存储（路径：l1_edge/sandbox/）。
* 连不上重试：针对 L2/L3 接口调用的重试/退避策略及轮询逻辑已部署。


1. 完成：本地流程
* 资源检查与sandbox已执行。
* 训练已完成：[L1] gradients bytes=...
* 证明已生成：[L1] proof_hash=...

2. 未完成（空 endpoint ）
* 注册/心跳：skip L2 registration + retry register/heartbeat 无 L2 时跳过并重试后放弃。
* 上传梯度：retry upload_gradients 无 L2 仍在重试。
* 提交证明：skip L3 proof submit + retry submit_proof 无 L3 失败。


剩下约 10% 主要是这些：
强隔离（容器/OS 级沙箱）
L2/L3 对接的幂等与去重保障
安全（证书/签名/安全存储）