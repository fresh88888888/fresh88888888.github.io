---
title: Kubernetes 架构
date: 2023-07-31 11:23:01
tag: 
   - kubernetes
   - Cloud Native
category:
   - Kubernetes
---

Kubernetes(简称：k8s) 最初源于谷歌内部的 Borg，提供了面向应用的容器集群部署和管理系统。Kubernetes 的目标旨在消除编排物理 / 虚拟计算，网络和存储基础设施的负担，并使应用程序运营商和开发人员完全将重点放在以容器为中心的原语上进行自助运营。Kubernetes 也提供稳定、兼容的基础（平台），用于构建定制化的 workflows 和更高级的自动化任务。

Kubernetes 具备完善的**集群管理能力**，包括**多层次的安全防护和准入机制**、**多租户应用支撑能力**、透明的**服务注册和服务发现机制**、内建**负载均衡器、故障发现和自我修复能力**、**服务滚动升级和在线扩容**、**可扩展的资源自动调度机制、多粒度的资源配额管理能力**。Kubernetes 还提供**完善的管理工具，涵盖开发、部署测试、运维监控**等各个环节。

{% asset_img kubernetes_architecture.png "Kubernetes 架构" %}

Kubernetes是一个分布式服务，所有从架构上可以为Control Plane(Master) Node 和Worker Node，这也是分布式架构的主要特点。
<!-- more -->

控制面节点（Control Plane Node）负责容器编排并维护集群的所需状态，它具有以下组件：
1. `kube-apiserver`
2. `etcd`
3. `kube-scheduler`
4. `kube-controller-manager`
5. `cloud-controller-manager`

{% asset_img kube-api-server.drawio-1.png "kube-apiserver" %}

工作面节点（Worker Node）负责运行容器化应用程序,它具有以下组件。
1. `kubelet`
2. `kube-proxy`
3. `container runtime`

#### Control Plane Node 组件

##### kube-apiserver

kube-apiserver 是实现API Server主要组件，它可以进行水平扩展，API Server 开放了Kubernetes API，它是Kubernetes 控制面节点的前端组件。当您使用 kubectl 管理集群时，在后端，您实际上是通过HTTP REST API与 API 服务器进行通信。然而，内部集群组件（如调度程序、控制器等）使用gRPC与 API 服务器通信。API 服务器与集群中其他组件之间的通信通过 TLS 进行，以防止对集群进行未经授权的访问。kube-apiserver负责以下工作：

1. API管理：公开集群API并处理所有API请求。
2. 身份验证（使用客户端证书、不记名令牌和 HTTP 基本身份验证）和授权（ABAC 和 RBAC 评估）
3. 处理 API 请求并验证 API 对象（如 Pod、服务等）的数据（验证和变更准入控制器）
4. 它是唯一与 etcd 通信的组件，采用gRPC框架通信。
5. kube-apiserver 协调控制面节点与工作节点之间的所有进程。
6. kube-apiserver 有一个内置的代理，它是kube-apiserver进程的一部分。

##### etcd

Kubernetes 是一个分布式系统，它需要像 etcd 这样高效的分布式数据库来支持其分布式特性。它既充当后端服务发现又充当数据库。etcd是一个开源的强一致性、分布式k-v数据库。etcd具有以下特性：

- 强一致性：如果对一个节点进行更新，强一致性将确保它立即更新到集群中的所有其他节点。
- 分布式：etcd 被设计为作为集群在多个节点上运行，而不牺牲一致性。
- 键值存储：将数据存储为键和值的非关系数据库。它还公开了一个键值 API。该数据存储构建在BboltDB之上，BboltDB 是 BoltDB 的一个分支。

etcd 使用raft 共识算法 来实现强一致性和可用性。它以Leader-follower的方式工作，以实现高可用性并承受节点故障。
1. etcd 存储 Kubernetes 对象的所有配置、状态和元数据（pod、秘密、守护进程集、部署、配置映射、状态集等）。
2. etcd 允许客户端使用 API 订阅事件Watch() 。kube-apiserver 使用 etcd 的监视功能来跟踪对象状态的变化。
3. etcd使用gRP开方调用获取键/值的API 。此外，gRPC 网关是一个 RESTful 代理，它将所有 HTTP API 调用转换为 gRPC 消息。它使其成为 Kubernetes 的理想数据库。
4. etcd 以键值格式存储/registry目录键下的所有对象。etcd 是控制面节点中唯一的Statefulset（部署有状态应用和将数据保存到永久性存储空间的聚簇应用）组件。

{% asset_img etcd-component.png %}

##### kube-scheduler

kube-scheduler 负责调度工作节点上的 pod。部署 Pod 时，需要指定 Pod 指标要求，例如 CPU、内存、关联性、优先级、持久卷 (PV) 等。调度程序的主要任务是识别创建请求并为 Pod 选择最佳节点。下图显示了调度程序如何工作的：

{% asset_img kube-scheduler.png %}

kube-scheduler的工作原理：
1. 为了选择最佳节点，Kube 调度程序使用过滤和评分操作。
2. 在过滤中，调度程序找到最适合调度 Pod 的节点。例如，如果有五个具有资源可用性的工作节点来运行 pod，则它会选择所有五个节点。如果没有节点，则 Pod 不可调度并移至调度队列。如果它是一个大型集群，假设有 100 个工作节点，并且调度程序不会迭代所有节点。有一个名为 的调度程序配置参数percentageOfNodesToScore。默认值通常为50%。
3. 在评分阶段，调度程序通过为过滤后的工作节点分配分数来对节点进行排名。调度器通过调用多个调度插件来进行评分。最后，将选择排名最高的工作节点来调度 Pod。如果所有节点的等级相同，则将随机选择一个节点。
4. 一旦选择了节点，调度程序就会在 API 服务器中创建一个绑定事件。意思是绑定 pod 和节点的事件。

**它是一个监听 kube-apiserver中 pod 创建事件的控制器。调度程序有两个阶段。调度周期 和 绑定周期。它们一起被称为调度上下文。 调度 周期选择一个工作节点，绑定周期将该更改应用于集群。调度程序始终将高优先级 pod 放在低优先级 pod 之前进行调度。此外，在某些情况下，Pod 开始在所选节点中运行后，Pod 可能会被驱逐或移动到其他节点。您可以创建自定义调度程序并在集群中与本机调度程序一起运行多个调度程序。部署 Pod 时，您可以在 Pod 清单中指定自定义调度程序。因此，将根据自定义调度程序逻辑做出调度决策。调度器有一个可插拔的调度框架。这意味着，您可以将自定义插件添加到调度工作流程中**。

##### kube-controller-manager

> 在 Kubernetes 中，控制器是控制循环，用于监视集群的状态，然后在需要时进行或请求更改。每个控制器都会尝试使当前集群状态更接近所需状态。

假设您想要创建一个部署，您可以在清单 YAML 文件中指定所需的状态（声明性方法）。例如，2 个副本、1 个卷挂载、configmap 等。内置的部署控制器可确保部署始终处于所需状态。如果用户使用 5 个副本更新部署，部署控制器会识别它并确保所需状态为 5 个副本。

kube-controller-manager 是管理所有Kubernetes控制器的组件。Kubernetes 资源/对象（例如 pod、命名空间、作业、副本集）由各自的控制器管理。另外，kube调度器也是一个由Kube控制器管理器管理的控制器。kube调度器也是一个由kube-controller-manager管理的控制器。

{% asset_img kube-controller-manager.png %}

内置 Kubernetes 控制器的列表:
1. `Deployment controller`
2. `Replicaset controller`
3. `DaemonSet controller `
4. `Job Controller (Kubernetes Jobs)`
5. `CronJob Controller`
6. `endpoints controller`
7. `namespace controller`
8. `service accounts controller`
9. `Node controller`

kube-controller-manager 管理所有控制器，控制器将集群保持在所需的状态, 可以使用与自定义资源定义关联的自定义控制器来扩展 Kubernetes。

##### cloud-controller-manager

当kubernetes部署在云环境中时，云控制器管理器充当云平台API和Kubernetes集群之间的桥梁。这样，kubernetes 核心组件就可以独立工作，并允许云提供商使用插件与 kubernetes 集成。kube-controller-manager 允许 Kubernetes 集群配置云资源，例如实例（用于节点）、负载均衡器（用于服务）和存储卷（用于持久卷）。

{% asset_img cloud-controller-manager.png %}

kube-controller-manager 包含一组特定于云平台的控制器，可确保特定于云的组件（节点、负载均衡器、存储等）的所需状态。以下是属于云控制器管理器一部分的三个主要控制器。

1. `Node controller`: 该控制器通过与云提供商 API 对话来更新节点相关信息。例如，节点标记和注释、获取主机名、CPU 和内存可用性、节点健康状况等。
2. `Route controller`: 负责在云平台上配置网络路由。这样不同节点中的 Pod 就可以互相通信。
3. `Service controller`: 它负责为 kubernetes 服务部署负载均衡器、分配 IP 地址等。

> 部署负载均衡器类型的 Kubernetes 服务。这里 Kubernetes 提供了一个特定于云的负载均衡器并与 Kubernetes 服务集成。为云存储解决方案支持的 Pod 供应存储卷 (PV)。


#### Worker Node 组件

##### Kubelet

Kubelet 是一个代理组件，运行在集群中的每个节点上。Kubelet 不作为容器运行，而是作为守护进程运行，由 systemd 管理。

它负责向 kube-apiserver注册工作节点，并主要使用来自 kube-apiserver的 podSpec（Pod 规范 - YAML 或 JSON）。podSpec 定义了应该在 Pod 内运行的容器、它们的资源（例如 CPU 和内存限制）以及其他设置，例如环境变量、卷和标签。然后，通过创建容器将 podSpec 调整到所需的状态。

kubelet工作范围：
1. 创建、修改和删除 Pod 的容器。
2. 负责处理活跃度、就绪度和启动探测。
3. 负责通过读取 pod 配置并在主机上创建相应的目录来挂载卷。
4. 通过调用 kube-apiserver 来收集和报告节点和 Pod 状态。

Kubelet 也是一个控制器，它监视 Pod 更改并利用节点的容器运行时来拉取镜像、运行容器等。除了来自 API 服务器的 PodSpec 之外，kubelet 还可以接受来自文件、HTTP endpoint和 HTTP 服务器的 podSpec。

**Kubelet 使用 CRI（容器运行时接口）gRPC 接口与容器运行时进行通信。它还公开一个 HTTP endpoint来收集日志并为客户端提供执行会话。使用CSI（容器存储接口）gRPC 配置块存储卷。使用集群中配置的 CNI 插件来分配 Pod IP 地址并为 Pod 设置必要的网络路由和防火墙规则。**

{% asset_img kubelet-architecture.png %}

##### kube-proxy

Kubernetes 中的服务是一种向内部或外部流量公开一组 Pod 的方法。当您创建服务对象时，它会获得分配给它的虚拟 IP。它被称为 clusterIP。它只能在 Kubernetes 集群内访问。

Endpoint对象包含Service对象下所有Pod组的IP地址和端口。Endpoint Controller 负责维护 Pod IP 地址（端点）列表。Service controller 负责配置服务的Endpoint。

您无法 ping ClusterIP，因为它仅用于服务发现，与可 ping 通的 pod IP 不同。Kube-proxy 是一个守护进程，作为daemonset在每个节点上运行。它是一个代理组件，为 Pod 实现 Kubernetes 服务概念。（一组具有负载平衡功能的 Pod 的单个 DNS）。它主要代理 UDP、TCP 和 SCTP，不支持HTTP。当您使用服务 (ClusterIP) 公开 Pod 时，Kube-proxy 会创建网络规则以将流量发送到分组在 Service 对象下的后端 Pod（Endpoint）。这意味着，所有负载平衡和服务发现都由 Kube 代理负责。

Kube-proxy 工作原理：

kube-proxy 与 kube-apiserver 通信以获取有关服务 (ClusterIP) 以及相应 pod IP 和端口（Endpoint）的详细信息。它还监视服务和Endpoint的变化。Kube-proxy 使用以下任一模式来创建/更新规则，以将流量路由到服务后端的 Pod。
1. `IPTables`：这是默认模式。在 IPTables 模式下，流量由 IPtable 规则处理。在这种模式下，kube-proxy 会随机选择后端 pod 进行负载均衡。一旦建立连接，请求就会发送到同一个 pod，直到连接终止。
2. `IPVS`: 对于服务超过1000个的集群，IPVS提供性能提升。它支持后端负载均衡算法: 1. `rr：round-robin` ：这是默认模式; 2. `lc`：最少连接（打开连接的最小数量）; 3. `dh`: 目的地哈希; `sh`: 源哈希; `sed`：最短的预期延迟; `nq`: 从不排队。
3. 用户空间（遗留且不推荐）
4. Kernelspace：此模式仅适用于 Windows 系统。

{% asset_img kube-proxy.png %}

##### container runtime

您可能了解Java 运行时 (JRE)。它是在主机上运行Java程序所需的软件。同样，容器运行时是运行容器所需的软件组件。容器运行时运行在 Kubernetes 集群中的所有节点上。它负责从容器注册表中提取镜像、运行容器、为容器分配和隔离资源以及管理主机上容器的整个生命周期。

- **容器运行时接口（CRI）**：它是一组 API，允许 Kubernetes 与不同的容器运行时交互。它允许不同的容器运行时与 Kubernetes 互换使用。CRI 定义了用于创建、启动、停止和删除容器以及管理镜像和容器网络的 API。

- **开放容器倡议（OCI）**：它是一组容器格式和运行时的标准。

Kubernetes 支持多种符合容器运行时接口(CRI)的容器运行时（CRI-O、Docker Engine、containerd 等）。这意味着，所有这些容器运行时都实现 CRI 接口并公开 gRPC CRI API（运行时和图像服务端点）。正如我们在 Kubelet 部分中了解到的，kubelet 代理负责使用 CRI API 与容器运行时交互，以管理容器的生命周期。它还从容器运行时获取所有容器信息并将其提供给控制面Node。让我们以CRI-O容器运行时接口为例。以下是容器运行时如何与 kubernetes 配合使用的高级概述。

{% asset_img cri-o.png %}

1. 当 kube-apiserver 对 pod 发出新请求时，kubelet 与 CRI-O 守护进程通信，通过 Kubernetes 容器运行时接口启动所需的容器。
2. CRI-O 检查并从配置的容器注册表中提取所需的容器映像containers/image。
3. 然后，CRI-O 为容器生成 OCI 运行时规范 (JSON)。
4. 最后，CRI-O 启动与 OCI 兼容的运行时 (runc)，以根据运行时规范来启动容器进程。


#### Kubernetes 集群插件组件

除了核心组件之外，kubernetes 集群还需要附加组件才能完全运行。选择插件取决于项目要求和用例。以下是集群上可能需要的一些流行插件组件。
1. CNI插件（容器网络接口）
2. CoreDNS（用于 DNS 服务器）： CoreDNS 充当 Kubernetes 集群内的 DNS 服务器。通过启用此插件，您可以启用基于 DNS 的服务发现。
3. Metrics Server（用于资源指标）：此插件可帮助您收集集群中节点和 Pod 的性能数据和资源使用情况。
4. Web UI（Kubernetes 仪表板）：此插件使 Kubernetes 仪表板能够通过 Web UI 管理对象。

##### CNI插件

CNI（容器网络接口）， 云原生计算基金会项目包含用于编写插件以在 Linux 和 Windows 容器中配置网络接口的规范和库，以及许多受支持的插件。CNI 只关心容器的网络连接，并在删除容器时删除分配的资源。由于这一重点，CNI 拥有广泛的支持，并且规范易于实现。它是一个基于插件的架构，具有供应商中立的规范和库，用于为容器创建网络接口。它并非特定于 Kubernetes。通过 CNI，容器网络可以在 Kubernetes、Mesos、CloudFoundry、Podman、Docker 等容器编排工具之间实现标准化。

当谈到容器网络时，企业可能有不同的需求，如网络隔离、安全、加密等。随着容器技术的进步，许多网络提供商为容器创建了基于 CNI 的解决方案，并具有广泛的网络功能。您可以将其称为 CNI-Plugins。这使得用户可以从不同的提供商中选择最适合其需求的网络解决方案。CNI 插件如何与 Kubernetes 配合使用？
1. `Kube-controller-manager` 负责为每个节点分配 pod CIDR。每个 Pod 从 Pod CIDR 获取唯一的 IP 地址。
2. `Kubelet` 与容器运行时交互以启动`预定的 pod。CRI 插件是容器运行时的一部分，它与 CNI 插件交互来配置 Pod 网络。
3. `CNI` 插件支持使用覆盖网络在相同或不同节点上分布的 Pod 之间进行联网。

CNI 插件的高级功能：
- `Pod Networking`
- `Pod` 网络安全和隔离使用网络策略来控制 Pod 之间以及命名空间之间的流量。

一些流行的 CNI 插件包括：
- `Calico`
- `Flannel`
- `Weave Net`
- `Cilium` (Uses eBPF)
- `Amazon VPC CNI` (For AWS VPC)
- `Azure CNI`(For Azure Virtual network)Kubernetes networking is a big topic and it differs based on the hosting platforms.
