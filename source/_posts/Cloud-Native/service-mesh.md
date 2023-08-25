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
- 定义虚拟服务路由目标地址的真实地址，即子集（`subset`）
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
- 调整`Envoy`代理接管的端口和协议
- 限制`Envoy`代理可访问的服务
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
- `Istio`指标的分类（1.**代理级别的指标**,收集目标是`sidecar`代理，资源粒度上的网格监控，容许指定收集的代理，主要用于针对性的调试；2.**服务级别的指标**，用于监控服务通信，四个基本的服务监控需求是延迟、流量、错误、饱和，默认指标导出到`Prometheus`，并且可自定更改。可根据需求开启和关闭；3.**控制平面指标**，对自身组件行为的监控，用于了解网络的健康情况）

###### 访问日志（Access Logs）
- 通过应用产生的事件来了解系统
- 包括了完整的元数据信息（目标、源）
- 生成位置可选（本地，远端，如`filebeat`）
- 日志内容
  - 应用日志
  - Envoy日志

###### 分布式追踪（Distributed tracing）
- 通过追踪请求，了解服务的调用关系
- 常用于调用链路的问题排查、性能分析
- 支持多种追踪系统（`jeager、Zipkin、DataDog`）

##### 网络安全
{% asset_img service_mesh_14.png  %}

- 身份认证（authentication）：证书管理、证书认证，`You are who you say you are`
- 访问授权（authoriztaion）：授权策略, `You can do what you want to do`
- 验证（vaildation）：输入验证, `Input is correct`

###### 认证
- 认证方式
  - 对等认证：用于服务间身份认证。 `Mutual（mTLS）`
  - 请求认证：用于终端用户身份认证；`Json Web Token(JWT)`
- 认证策略
  - 配置方式：`yaml`配置文件
  - 配置生效范围：网格-全网格生效、按命名空间生效、按服务（工作负载）生效
  - 策略更新：
- 支持兼容模式
{% asset_img service_mesh_15.png  %}

认证策略同步到数据面的方式为：
阶段一：
```
对等认证-----设置mTLS----兼容模式
                          
                          
                       严格模式

                (新)jwt
              /      \
             /        \
身份认证                 策略
             \        /
              \      /
                (旧)jwt
```
阶段二：
```
对等认证-----设置mTLS----严格模式
                          
                          
                       严格模式

                (新)jwt
            /      \
           /        \
身份认证                 策略
           \         
            \     
                (旧)jwt
```

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

授权策略：配置通过`AuthorizationPolicy`实现；组成部分：选择器（`selector`）;行为(`Action`); 规则列表(`Rules`)：来源（`from`）、操作(`to`)、匹配条件(`when`);范围设置：`metadata/namespace,selector`; 值匹配：精确、模糊、前缀、后缀；全部允许和拒绝；自定义条件。
###### 安全发现服务（SDS）
- 身份和证书管理
- 实现安全配置自动化
- 中心化`SDS Server`
- 优点：无需挂载secret卷；动态更新证书，无需重启；可监视多个证书密钥对
{% asset_img service_mesh_31.png  %}
###### 双向验证 (mTLS)
- `TLS`：客户端根据服务端证书验证其身份
- `mTLS`：客户端、服务器端彼此都验证对方身份
###### 身份认证及授权（JWT）
- `Json Web Token`
- 以`json`格式传递信息
- 应用场景：授权和信息交换
- 组成部分：`Header、Payload、Signature`
{% asset_img service_mesh_32.png  %}
##### 调试及测试
###### 故障注入
- `Netflix`的 `Chaos Monkey`
- 混沌工程（`Chaos engineering`）
{% asset_img service_mesh_22.png  %}
配置延迟故障
{% asset_img service_mesh_23.png  %}
###### 流量镜像（Traffic Mirroring）
- 实时复制请求到镜像服务
- 应用场景：1.线上问题排查(`troubleshooting`)；2.观察生产环境的请求处理能力(压力测试)；3.复制请求信息用于分析

##### Istio 安装部署

1. 安装K8s
2. 安装 `Istio`
3. 部署官方 `Demo` 体验 `Istio`

###### 安装K8S
请参见[`minikubte download`](https://minikube.sigs.k8s.io/docs/start/) 略

###### 下载 Istio
- 在线下载：`curl -L https://istio.io/downloadIstio | sh -`
- 离线下载：
```bash
sudo wget https://github.com/istio/istio/releases/download/1.18.2/istio-1.18.2-linux-amd64.tar.gz
...
tar -zxvf istio-1.18.2-linux-amd64.tar.gz
```
压缩包里包含以下内容:
```bash
[root@vela istio-1.18.1]# ll
drwxr-x---  2 root root    22 Jul 14 04:37 bin
-rw-r--r--  1 root root 11348 Jul 14 04:37 LICENSE
drwxr-xr-x  5 root root    52 Jul 14 04:37 manifests
-rw-r-----  1 root root   986 Jul 14 04:37 manifest.yaml
-rw-r--r--  1 root root  6595 Jul 14 04:37 README.md
drwxr-xr-x 24 root root  4096 Jul 14 04:37 samples
drwxr-xr-x  3 root root    57 Jul 14 04:37 tools

```
各个目录的作用：
- `bin`：存放的是 `istioctl` 工具
- `manifests`：相关 `yaml` 用于部署 `Istio`
- `samples`：一些 `Demo` 用的 `yaml`
- `tools`：一些工具，暂时使用不到
先将`istioctl`工具 `cp` 到 `bin` 目录下，便于后续使用 `istioctl` 命令。
```bash
cp bin/istioctl /usr/local/bin/
```

###### 安装 Istio
>由于易用性的问题，Istio 废弃了以前的 Helm 安装方式，现在使用 istioctl 即可一键安装。
`Istio` 提供了以下配置档案（`configuration profile`）供不同场景使用，查看当前内置的 `profile`：
```bash
$ istioctl profile list
Istio configuration profiles:
    default
    demo
    empty
    external
    minimal
    openshift
    preview
    remote

```
具体每个 `profile` 包含哪些组件，可以使用`istioctl profile dump`命令查看：
```bash
$ istioctl profile dump demo
```
对于演示环境，我们直接安装 demo 版本就可以了 
```bash
$ istioctl install --set profile=demo -y

✔ Istiod installed                                       
✔ Ingress gateways installed                             
✔ Egress gateways installed                             
✔ Installation complete
Making this installation the default for injection and validation.
```
部署完成后,这里k8s先需要创建一个命名空间, 然后给命名空间打上 label，告诉 Istio 在部署应用的时候，自动注入 Envoy 边车代理：
```bash
$ kubectl create namspace `k8s-istio`
...
$ kubectl label namespace `k8s-istio` istio-injection=enabled
```
安装后可以验证是否安装正确。
```bash
# 先根据安装的profile导出manifest
istioctl manifest generate --set profile=demo > $HOME/generated-manifest.yaml
# 然后根据验证实际环境和manifest文件是否一致
istioctl verify-install -f $HOME/generated-manifest.yaml
# 出现下面信息则表示验证通过
✔ Istio is installed and verified successfully
```
查看一下安装了些什么东西：
```bash
$ kubectl get pods -n istio-system

NAME                                    READY   STATUS    RESTARTS   AGE
istio-egressgateway-75db994b58-jlc28    1/1     Running   0          38m
istio-ingressgateway-79bb75ddbb-dmm87   1/1     Running   0          38m
istiod-68cb9f5cb6-h5fcv                 1/1     Running   0          39m
```
可以看到只安装了出入站网关以及最重要的 Istiod 服务。再看下 CRD 情况:
```bash
$ kubectl get crds |grep istio

authorizationpolicies.security.istio.io    2023-08-24T05:12:04Z
destinationrules.networking.istio.io       2023-08-24T05:12:04Z
envoyfilters.networking.istio.io           2023-08-24T05:12:04Z
gateways.networking.istio.io               2023-08-24T05:12:04Z
istiooperators.install.istio.io            2023-08-24T05:12:04Z
peerauthentications.security.istio.io      2023-08-24T05:12:04Z
proxyconfigs.networking.istio.io           2023-08-24T05:12:04Z
requestauthentications.security.istio.io   2023-08-24T05:12:04Z
serviceentries.networking.istio.io         2023-08-24T05:12:04Z
sidecars.networking.istio.io               2023-08-24T05:12:04Z
telemetries.telemetry.istio.io             2023-08-24T05:12:04Z
virtualservices.networking.istio.io        2023-08-24T05:12:04Z
wasmplugins.extensions.istio.io            2023-08-24T05:12:05Z
workloadentries.networking.istio.io        2023-08-24T05:12:05Z
workloadgroups.networking.istio.io         2023-08-24T05:12:05Z
```
这些就是 istio 需要用到的 CRD 了，比较常见的比如：
- `gateways`
- `virtualservices`
- `destinationrules`

###### 部署 bookinfo 应用
官方提供了 `bookinfo` 应用来演示 `Istio` 相关功能。
```bash
$ kubectl apply -f samples/bookinfo/platform/kube/bookinfo.yaml

service/details created
serviceaccount/bookinfo-details created
deployment.apps/details-v1 created
service/ratings created
serviceaccount/bookinfo-ratings created
deployment.apps/ratings-v1 created
service/reviews created
serviceaccount/bookinfo-reviews created
deployment.apps/reviews-v1 created
deployment.apps/reviews-v2 created
deployment.apps/reviews-v3 created
service/productpage created
serviceaccount/bookinfo-productpage created
deployment.apps/productpage-v1 created
```

**部署应用**
在 `k8s-istio` 命名空间创建了应用对应的 `service` 和 `deployment`。服务启动需要一定时间，可通过以下命令进行查看：
```bash
$ kubectl get services
NAME          TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
details       ClusterIP   10.106.81.196   <none>        9080/TCP   54m
kubernetes    ClusterIP   10.96.0.1       <none>        443/TCP    19h
productpage   ClusterIP   10.103.139.13   <none>        9080/TCP   54m
ratings       ClusterIP   10.102.135.80   <none>        9080/TCP   54m
reviews       ClusterIP   10.103.83.28    <none>        9080/TCP   54m

$ kubectl get pods
NAME                              READY   STATUS              RESTARTS   AGE
details-v1-698b5d8c98-p5kwm       0/1     ContainerCreating   0          37s
productpage-v1-75875cf969-frn4z   0/1     ContainerCreating   0          35s
ratings-v1-5967f59c58-zbldg       0/1     ContainerCreating   0          36s
reviews-v1-9c6bb6658-kztz7        0/1     ContainerCreating   0          36s
reviews-v2-8454bb78d8-jzghc       0/1     ContainerCreating   0          36s
reviews-v3-6dc9897554-qvn7g       0/1     ContainerCreating   0          36s
```
等 `pod` 都启动后，通过以下命令测试应用是否正常启动了：
```bash
kubectl exec "$(kubectl get pod -l app=ratings -o jsonpath='{.items[0].metadata.name}')" -c ratings -- curl -s productpage:9080/productpage | grep -o "<title>.*</title>"
```
输出以下内容就算成功:
```html
<title>Simple Bookstore App</title>
```

**部署网关**
此时，BookInfo 应用已经部署，但还不能被外界访问。需要借助网关才行
```bash
$ kubectl apply -f samples/bookinfo/networking/bookinfo-gateway.yaml
```
输出如下：
```bash
gateway.networking.istio.io/bookinfo-gateway created
virtualservice.networking.istio.io/bookinfo created
```
可以看到， 这里部署了一个网关（`gateway`）和一个虚拟服务（`virtualservice`）。此时在浏览器中，输入`http://localhost/productpage`应该可以访问到具体页面了。
确保配置文件没有问题：
```bash
$ istioctl analyze
✔ No validation issues found when analyzing namespace: default.
```
确定ingress的ip和端口，外部访问则需要通过 `NodePort` 访问:
```bash
$ kubectl -n  istio-system get svc istio-ingressgateway
NAME                   TYPE           CLUSTER-IP       EXTERNAL-IP   PORT(S)                                                                      AGE
istio-ingressgateway   LoadBalancer   10.108.211.196   <pending>     15021:31160/TCP,80:32096/TCP,443:30055/TCP,31400:31682/TCP,15443:30083/TCP   22m
```
可以看到，80 端口对应的 `NodePort` 为 32096,那么直接访问的 `URL` 就是：`http://$IP:32096/productpage`。

在新的终端窗口中运行此命令以启动一个 Minikube 隧道，将流量发送到 Istio Ingress Gateway。 这将为 service/istio-ingressgateway 提供一个外部负载均衡器 EXTERNAL-IP。
```bash
$ minikube tunnel

Status:	
	machine: minikube
	pid: 1462709
	route: 10.96.0.0/12 -> 192.168.49.2
	minikube: Running
	services: [istio-ingressgateway]
    errors: 
		minikube: no errors
		router: no errors
		loadbalancer emulator: no errors
```
设置入站主机和端口：
```bash
$ export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
$ export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
$ export SECURE_INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="https")].port}')
```
设置环境变量 GATEWAY_URL：
```bash
$ export GATEWAY_URL=$INGRESS_HOST:$INGRESS_PORT
```
确保 IP 地址和端口均成功地赋值给了环境变量：
```bash
$ echo "$GATEWAY_URL"
192.168.99.100:32194
```
用浏览器查看 Bookinfo 应用的产品页面，验证 Bookinfo 已经实现了外部访问。

###### 查看仪表盘（Dashboard）
`Istio` 也提供了 `Dashboard`，可以通过UI 界面更加方便的进行管理，安装命令如下：
```bash
$ kubectl apply -f samples/addons
```
等待安装完成
```bash
$ kubectl rollout status deployment/kiali -n istio-system
...
Waiting for deployment "kiali" rollout to finish: 0 of 1 updated replicas are available..
kubectl rollout status deployment/kiali -n istio-system.
```

访问 bashboard：
```bash
# 安装 Kiali 和其他插件，等待部署完成。
$ kubectl apply -f samples/addons
$ kubectl rollout status deployment/kiali -n istio-system
Waiting for deployment "kiali" rollout to finish: 0 of 1 updated replicas are available...
deployment "kiali" successfully rolled out
# 访问 Kiali 仪表板。
$ istioctl dashboard kiali
```
外部访问则把 `service` 改成 `nodeport` 类型即可
```bash
$ kubectl -n istio-system  patch svc kiali -p '{"spec":{"type":"NodePort"}}'
```
查看修改后的端口
```bash
$ kubectl -n istio-system get svc kiali
NAME    TYPE       CLUSTER-IP     EXTERNAL-IP   PORT(S)                          AGE
kiali   NodePort   10.101.47.41   <none>        20001:31989/TCP,9090:32136/TCP   4m14s
```
访问 `http://$IP:31989` 即可，在主界面可以看到部署的服务以及请求量、资源使用量等情况。在 `Graph` 界面则能够看到，服务间流量分发情况。由于之前部署的 `bookinfo` 服务没怎么访问，所以界面是空白的，先通过 `curl` 命令访问一下。在左侧的导航菜单，选择 Graph ，然后在 Namespace 下拉列表中，选择 istio-system 。
>**要查看追踪数据，必须向服务发送请求。请求的数量取决于 Istio 的采样率。 采样率在安装 `Istio` 时设置，默认采样速率为 1%。在第一个跟踪可见之前，您需要发送至少 100 个请求。 使用以下命令向 `productpage` 服务发送 100 个请求：**
```bash
for i in `seq 1 100`; do curl -s -o /dev/null http://$GATEWAY_URL/productpage; done
```
`Kiali` 仪表板展示了网格的概览以及 `Bookinfo` 示例应用的各个服务之间的关系。 它还提供过滤器来可视化流量的流动。
{% asset_img service_mesh_21.png %}

###### 卸载 Istio
以下命令卸载 Istio 并删除所有相关资源
```bash
$ kubectl delete -f samples/addons
$ istioctl x uninstall --purge -y
```
删除 `namespace`
```bash
kubectl delete namespace k8s-istio
```
移除之前打的 `label`
```bash
$ kubectl label namespace default istio-injection-
```
到这里，Istio 的安装就结束了，后续就可以用起来了。

##### Istio 遥测（Telemetry）

`Istio` 服务网格最受欢迎和最强大的功能之一是其高级可观察性。由于所有服务到服务的通信都是通过 `Envoy` 代理进行路由，并且 `Istio` 的控制平面能够从这些代理收集日志和指标，因此服务网格可以为我们提供有关网络状态和服务行为的数据。这为运营商提供了独特的故障排除、管理和优化服务的方法，而不会给应用程序开发人员带来任何额外的负担。

###### Istio 遥测指标
- 代理级别指标
  代理级别指标是 `Envoy` 代理本身提供的有关所有直通流量的标准指标，以及有关代理管理功能的详细统计信息，包括配置和运行状况信息。Envoy 生成的指标存在于 Envoy 资源（例如监听器和集群）的粒度级别。
  ```json
  # TYPE envoy_cluster_internal_upstream_rq_200 counter
  envoy_cluster_internal_upstream_rq_200{cluster_name="xds-grpc"} 2

  # TYPE envoy_cluster_upstream_rq_200 counter
  envoy_cluster_upstream_rq_200{cluster_name="xds-grpc"} 2

  # TYPE envoy_cluster_upstream_rq_completed counter
  envoy_cluster_upstream_rq_completed{cluster_name="xds-grpc"} 3

  # TYPE envoy_cluster_internal_upstream_rq_503 counter
  envoy_cluster_internal_upstream_rq_503{cluster_name="xds-grpc"} 1

  # TYPE envoy_cluster_upstream_cx_rx_bytes_total counter
  envoy_cluster_upstream_cx_rx_bytes_total{cluster_name="xds-grpc"} 2056154

  # TYPE envoy_server_memory_allocated gauge
  envoy_server_memory_allocated{} 15853480
  ```

- 服务级别指标
  除了代理级别的指标之外，`Istio` 还提供了一组面向服务的指标来监控服务通信。这些指标涵盖了四种基本服务监控需求：延迟、流量、错误和饱和度。`Istio` 附带了一组默认的仪表板，用于根据这些指标监控服务行为。
  ```json
  # TYPE istio_requests_total counter
  istio_requests_total{
      connection_security_policy="mutual_tls",
      destination_app="analytics",
      destination_principal="cluster.local/ns/backyards-demo/sa/default",
      destination_service="analytics.backyards-demo.svc.cluster.local",
      destination_service_name="analytics",
      destination_service_namespace="backyards-demo",
      destination_version="v1",
      destination_workload="analytics-v1",
      destination_workload_namespace="backyards-demo",
      permissive_response_code="none",
      permissive_response_policyid="none",
      reporter="destination",
      request_protocol="http",
      response_code="200",
      response_flags="-",
      source_app="bookings",
      source_principal="cluster.local/ns/backyards-demo/sa/default",
      source_version="v1",
      source_workload="bookings-v1",
      source_workload_namespace="backyards-demo"
  } 1855
  ```
###### Istio 遥测架构 v2
{% asset_img service_mesh_24.png %}
>根据 `Istio` 文档，新的遥测系统将延迟减少了一半 - 90% 的延迟已从 7 毫秒减少到 3.3 毫秒。不仅如此，`Mixer` 的消除还使总 `CPU` 消耗减少了 50%，达到每秒每 1,000 个请求 0.55 个 `vCPU`。
###### WASM架构
`WebAssembly`（通常缩写为 `WASM`）是一种开放标准，它定义了可执行程序的可移植二进制代码格式、相应的文本汇编语言以及促进程序与其主机环境之间交互的接口。`WebAssembly` 的主要目标是在网页上启用高性能应用程序，但该格式也设计用于在其他环境中执行和集成。它提供了一个基于精简堆栈的虚拟机，允许 `Web` 应用程序通过利用快速加载的二进制格式以接近本机的速度运行，该格式也可以转换为文本格式以进行调试。而且，虽然 `WebAssembly` 最初是作为客户端技术出现的，但在服务器端使用它有很多优点。`Istio` 社区一直在领导 `Envoy` 的 `WebAssembly` (`WASM`) 运行时的实现。该实现使用基于 `Google` 高性能`V8` 引擎构建的 `WebAssembly` 运行时。借助 Envoy 的 `WebAssembly` 插件，开发人员可以编写自定义代码，将其编译为 `WebAssembly` 插件，并配置 `Envoy` 来执行它。

`Telemetry V2` 中的代理内服务级别指标由两个自定义插件提供，
- `metadata-exchange`：必须解决的第一个问题是如何在代理中提供有关连接两端的客户端/服务器元数据。对于基于 `HTTP` 的流量，这是通过包含对方元数据属性的请求/响应中的自定义 `HTTP` 标头 (`envoy.wasm.metadata_exchange.upstream、envoy.wasm.metadata_exchange.downstream`) 来完成的。对于通用 `TCP` 流量，元数据交换使用基于 `ALPN` 的隧道和基于前缀的协议。定义了一个新协议`istio-peer-exchange`，该协议由网格中的客户端和服务器 `sidecar` 进行通告和优先级排序。`ALPN` 协商将协议解析为 `istio-peer-exchange`，用于启用 `Istio` 的代理之间的连接，但不解析启用 `Istio` 的代理和任何客户端之间的连接。

- `stats`：`stats` 插件将传入和传出的流量指标记录到 `Envoy` 统计子系统中，并可供 `Prometheus` 抓取。以下是服务级别指标的默认标签。
```json
eporter: conditional((context.reporter.kind | "inbound") == "outbound", "source", "destination")
source_workload: source.workload.name | "unknown"
source_workload_namespace: source.workload.namespace | "unknown"
source_principal: source.principal | "unknown"
source_app: source.labels["app"] | "unknown"
source_version: source.labels["version"] | "unknown"
destination_workload: destination.workload.name | "unknown"
destination_workload_namespace: destination.workload.namespace | "unknown"
destination_principal: destination.principal | "unknown"
destination_app: destination.labels["app"] | "unknown"
destination_version: destination.labels["version"] | "unknown"
destination_service: destination.service.host | "unknown"
destination_service_name: destination.service.name | "unknown"
destination_service_namespace: destination.service.namespace | "unknown"
request_protocol: api.protocol | context.protocol | "unknown"
response_code: response.code | 200
connection_security_policy: conditional((context.reporter.kind | "inbound") == "outbound", "unknown", conditional(connection.mtls | false, "mutual_tls", "none"))
response_flags: context.proxy_error_code | "-"
source_canonical_service
source_canonical_revision
destination_canonical_service
destination_canonical_revision
```
###### Istio 的安全机制
- 透明的安全层
- `CA`：秘钥和证书管理
- `API Server`：认证、授权策略分发
- `Envoy`：服务间安全通信（认证、加密）

##### Envoy 架构

###### Envoy 流量五元组
{% asset_img service_mesh_25.png %}

###### Envoy 调试关键字段（RESPONSE_FLAG）
- `UH：upstream_cluster` 中没有监控的`host`, 503
- `UF: upstream` 连接失败， 503
- `UO: upstream_overflow` (熔断)
- `NR:` 没有路由配置，404
- `URX:` 请求被拒绝因为限流或超过最大连接次数
- ... ...

##### 分布式追踪

- 分析和监控应用的监控方法
- 查找故障点，分析性能问题
- 观测请求范围内的信息
- 起源于`Google的Dapper`
- `OpenTracing: API` 规范、框架、库的组合

- `Span`: 逻辑单元；有操作名、执行时间；嵌套、有序、因果关系
- `Trace`：数据/执行路径；`Span`的组合
{% asset_img service_mesh_26.png %}

###### Jaejer 组件
[分布式追踪框架 Jaejer](https://www.jaegertracing.io/) `Jaeger` 是`Uber Technologies`开源发布的分布式追踪平台。组件包括：
- `Tracing SDKs`: 
  为了生成跟踪数据，必须对应用程序进行检测。受检测的应用程序在接收新请求时创建跨度，并将上下文信息（`trace id, span id, and baggage`）附加到传出请求。仅`ids`和`baggage`通过请求传播；不会传播所有其他分析数据，例如操作名称、时间、标签和日志。它会在后台异步导出到 `Jaeger` 后端。
  {% asset_img service_mesh_27.png %}
- `Agent`: 
  >`jaeger-agent`已弃用。`OpenTelemetry`数据可以直接发送到 `Jaeger` 后端，也可以使用 `OpenTelemetry Collector` 作为代理。
- `Collector`: `jaeger-collector`接收跟踪的数据，通过管道处理数据以进行检验和清晰/补全，并将跟踪的数据存储在数据库中。`jaeger-collector`内置了对多种数据库的支持,以及实现了可扩展插件框架用于自定义存储插件。
- `Query`: `jaeger-query`提供了从存储中检索跟踪的`API`，并托管用于搜索和分析跟踪的 `Web UI`。
- `Ingester`: `jaeger-ingester`是一项从 `Kafka` 读取跟踪并将其写入存储后端的服务。实际上，它是 `Jaeger` 收集器的精简版本，支持 `Kafka` 作为唯一的输入协议。
###### Jaejer 架构
- 直接存储模式
  此部署方式，收集器从跟踪的应用程序接收数据并将其直接写入存储。存储必须能够处理平均流量和峰值流量。收集器使用内存队列来平滑短期流量峰值，但如果存储无法跟上，持续的流量峰值可能会导致数据丢失。收集器能够向 `SDK` 集中提供采样配置，称为远程采样模式。它们还可以启用自动采样配置计算，称为自适应采样。
  {% asset_img service_mesh_28.png %}
- `Kafka`存储模式
  为了防止收集器和存储之间的数据丢失，`Kafka` 可以用作中间的持久队列。需要部署一个附加组件`jaeger-ingester`来从 `Kafka` 读取数据并保存到数据库。可以部署多个`jaeger-ingester`来扩大摄取规模；他们会自动分配负载。
  {% asset_img service_mesh_29.png %}
- 开放遥测模式
  `Jaeger Collectors` 可以直接从 `OpenTelemetry SDK` 接收 `OpenTelemetry` 数据。如果您已经使用 `OpenTelemetry Collectors`，例如用于收集其他类型的遥测数据或用于预处理/丰富跟踪数据，则可以将其放置在 `SDK` 和 `Jaeger Collectors` 之间。`OpenTelemetry Collector` 可以作为应用程序边车、主机代理/守护程序或中央集群运行。`OpenTelemetry Collector`支持`Jaeger` 的远程采样协议，可以直接从配置文件提供静态配置，也可以将请求代理到 `Jaeger` 后端（例如，当使用自适应采样时）。
  {% asset_img service_mesh_30.png %}

##### 项目实践

###### 典型的CI/CD过程（DevOps）
{% asset_img service_mesh_33.png %}
###### GitOps 持续集成/交付过程
- `GitOps`：集群管理和应用分发的持续交付方式
- 使用`git`作为信任源，保存声明式基础架构（`declarative infrastructure`）和应用程序
- 以git作为交付过程（`pipeline`）的中心
- 开发者只需通过`pull request`完成应用的部署和运维任务
- 优势：提高生产率、改进开发体验、一致性和标准化、安全
{% asset_img service_mesh_34.png %}
###### DevOps（push pipeline）vs GitOps（pull pipeline）
{% asset_img service_mesh_35.png %}

优点：`GitOps（pull pipeline）`方式，部署时不用暴露安全相关的脚本操作

###### Flux 介绍
`Flux` 是一套开放、可扩展的 `Kubernetes` 持续渐进式交付解决方案。
- 定义：`The GitOps operator for kubernetes`
- 自动化部署工具（基于GitOps）
- 官方地址：![Flux](https://fluxcd.io/)
- 只需推送到 Git，Flux 就会完成剩下的工作
- 自动同步自动部署
- 声明式
- 基于代码（`Pull Request`），而不是容器
{% asset_img service_mesh_36.png %}

###### Flagger 自动化灰度发布(金丝雀部署)
`Flagger` 实现了一个控制循环，逐渐将流量转移到金丝雀，同时测量 `HTTP` 请求成功率、请求平均持续时间和 `Pod` 运行状况等关键性能指标。根据设置的阈值，金丝雀将被升级或中止，其分析将被推送到 `Slack` 通道。
- 自动化的灰度发布工具
- 支持多种`Service Mesh`,包括(`istio`、`linkerd`、`app aws mesh`)
- 指标监控灰度发布状态
- 通知功能（接入`slack、microsoft term`）
{% asset_img service_mesh_37.png %}

`Flagger`工作流程（状态流转）
- `Initializing`
- `Initialized`
- `Progressing`
- `Successed(Failed)`
{% asset_img service_mesh_38.png %}
官方地址：![Flagger](https://docs.flagger.app/)

##### 弹性设计
系统/服务的弹性能力直接决定了它的可用性，可用性的度量是由服务级别协议（SLA - Service Level Agreement）来计算的，可用性计算公式 $Availability = \dfrac{MTBF}{MTBF + MTTR}$。
- 应对故障的一种方法，让系统具有容错和适应能力。
- 防止故障（`Fault`）转化为失败（`Failure`）。
- 特点：包括1.容错性：重试、幂等；伸缩性：自动水平扩展（`autoscaling`）;过载保护：超时、熔断、降级、限流；弹性测试：故障注入

`Istio`提供的弹性能力包括：超时、重试、熔断、故障注入。
