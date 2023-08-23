---
title: Service Mesh 架构
date: 2023-08-22 10:23:01
tag: 
   - service mesh
   - Cloud Native
category:
   - Service Mesh
---

微服务架构的特性

- 围绕业务构建团队
{% asset_img service_mesh_1.png %}

- 去中心化的数据管理
{% asset_img service_mesh_2.png %}
<!-- more -->
如何管理和控制服务间的通信

- 服务注册和发现
- 路由，流量转移
- 弹性能力（熔断、超时、重试）
- 安全（身份认证、授权）
- 可观察性（可视化）

##### Service Mesh 演进

{% asset_img service_mesh_3.png %}

##### Service Mesh 定义

{% asset_img service_mesh_4.png %}

##### Service Mesh 产品形态

{% asset_img service_mesh_5.png %}
`service mesh` 是`sidecar`的网络拓扑模式

##### Service Mesh 主要功能

- 流量控制：路由:（负载均衡、蓝绿部署、灰度发布、AB测试）、流量转移、超时重试、熔断、故障注入、流量镜像
- 策略：黑、白名单、流量限制
- 网络安全：授权、身份认证
- 可观测性：指标收集和展示、日志收集、分布式追踪

##### Service Mesh 与 Kubernetes的关系

- kubernetes
    1. 解决容器编排和调度的问题
    2. 本质上是管理应用的生命周期（调度器）
    3. 给予`service mesh` 支持和帮助

- service mesh
    1. 解决服务间网络通信的问题
    2. 本质上是管理服务通信（代理）
    3. 是对`kubernetes`网络功能方面的扩展和延伸

##### Service Mesh 与 API网关的异同点

{% asset_img asset_img service_mesh_6.png %}
- 功能有重叠，但角色不同
- `Service Mesh` 在应用内，API网关在应用之上（边界）

##### Service Mesh产品发展史

{% asset_img service_mesh_7.png  %}

##### Istio 的流量控制能力

- 路由(负载均衡)、流量转移
- 流量进出
- 网络弹性能力（熔断、超时控制）
- 测试相关

###### 核心资源
- 虚拟服务（`Virtual Service`）
- 目标规则（`Destination Rule`）
- 网关（`Gateway`）
- 服务入口（`Service Entry`）
- `SideCar`

###### 虚拟服务
- 将流量路由到给定目标地址
- 请求地址与真实的工作负载解耦
- 包含一组路由规则
- 通常和目标规则（`destnation rule`）成堆出现
- 丰富的路由匹配规则
{% asset_img service_mesh_8.png  %}

###### 目标规则
- 定义虚拟服务路由目标地址的真实地址，即子集（subset）
- 设置负载均衡的方式（随机、轮询、权重、最小请求数）
{% asset_img service_mesh_9.png  %}

###### 网关
- 管理进出网格的流量
- 处在网格的边界
{% asset_img service_mesh_10.png  %}

###### 服务入口
- 把外部服务注册到网格中
- 功能（为外部目标转发请求；添加超时重试等策略；扩展网格）
{% asset_img service_mesh_11.png  %}

###### SideCar
- 调整Envoy代理接管的端口和协议
- 限制Envoy代理可访问的服务
{% asset_img service_mesh_12.png  %}

###### 网络弹性和测试
- 弹性能力：超时、重试、熔断
- 测试能力：故障注入、流量镜像

##### 可观测性

- 从开发者的角度探究系统的运行状态
- 组成：指标、日志、追踪
{% asset_img service_mesh_13.png  %}

###### 指标（Metrics）
- 以聚合的方式监控和理解系统的行为
- Istio指标的分类（1.**代理级别的指标**,收集目标是sidecar代理，资源粒度上的网格监控，容许指定收集的代理，主要用于针对性的调试；2.**服务级别的指标**，用于监控服务通信，四个基本的服务监控需求是延迟、流量、错误、饱和，默认指标导出到Prometheus，并且可自定更改。可根据需求开启和关闭；3.**控制平面指标**，对自身组件行为的监控，用于了解网络的健康情况）

###### 访问日志（Access Logs）
- 通过应用产生的事件来了解系统
- 包括了完整的元数据信息（目标、源）
- 生成位置可选（本地，远端，如filebeat）
- 日志内容
  - 应用日志
  - Envoy日志

###### 分布式追踪（Distributed tracing）
- 通过追踪请求，了解服务的调用关系
- 常用于调用链路的问题排查、性能分析
- 支持多种追踪系统（jeager、Zipkin、DataDog）

##### 网络安全
{% asset_img service_mesh_14.png  %}

- 证书管理
- 身份认证
- 访问授权

###### 认证
- 认证方式
  - 对等认证：用于服务间身份认证。Mutual（mTLS）
  - 请求认证：用于终端用户身份认证；Json Web Token(JWT)
- 认证策略
  - 配置方式：yaml配置文件
  - 配置生效范围：网格-全网格生效、按命名空间生效、按服务（工作负载）生效
  - 策略更新：
- 支持兼容模式
{% asset_img service_mesh_15.png  %}

认证策略同步到数据面的方式为：
{% asset_img service_mesh_16.png  %} {% asset_img service_mesh_17.png  %}

###### 访问授权
- 授权级别
- 策略分发
- 授权引擎
- 无需显示启用
{% asset_img service_mesh_18.png  %}

大致分为三种方式：
- 按网格授权
- 按命名空间授权
- 按服务（工作负载）授权

授权策略：配置通过AuthorizationPolicy实现；组成部分：选择器（selector）;行为(Action); 规则列表(Rules)：来源（from）、操作(to)、匹配条件(when);范围设置：metadata/namespace,selector; 值匹配：精确、模糊、前缀、后缀；全部允许和拒绝；自定义条件。
