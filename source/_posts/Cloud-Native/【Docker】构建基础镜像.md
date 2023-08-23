---
title: 【Docker】构建基础镜像
date: 2023-08-14 10:23:01
tag: 
   - docker
   - Cloud Native
category:
   - Docker
---

##### 构建Nginx基础镜像

1. 使用apt或源码编译安装(1.`configure`; 2.`make`; 3.`make install`)
2. 启用哪些模块
3. `nginx` 初始化
4. 启动容器

在宿主机当前目录下目录下创建一个`Dockerfile-nginx`的文件，用`DockerFile`命令编写构建`nginx`镜像操作流程。如下：
<!-- more -->
```
FROM centos:8
MAINTAINER www.umbrella.com
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-* && \
    sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-* && \
    yum update -y && \
    yum install -y gcc gcc-c++ make \
    openssl-devel pcre-devel gd-devel \
    iproute net-tools telnet wget curl && \
    yum clean all && \
    rm -rf /var/cache/yum/*

RUN wget https://nginx.org/download/nginx-1.22.1.tar.gz && \
    tar zxf nginx-1.22.1.tar.gz && \
    cd nginx-1.22.1 && \
    ./configure --prefix=/usr/local/nginx \
    --with-http_ssl_module \
    --with-http_stub_status_module && \
    make -j 4 && make install && \
    rm -rf /usr/local/nginx/html/* && \
    echo "ok" >> /usr/local/nginx/html/status.html && \
    cd / && rm -rf nginx-1.22.1* && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

ENV PATH $PATH:/usr/local/nginx/sbin
#COPY nginx.conf /usr/local/nginx/conf/nginx.conf
WORKDIR /usr/local/nginx
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```
在命令行中用容器构建命令执行Dockerfile-nginx`文件脚本：
```bash
$ docker build -t nginx:v1 -f Dockerfile-nginx .
```
执行完成之后，通过容器镜像查看命令docker images命令查看生成的nginx镜像文件。
```
REPOSITORY                    TAG         IMAGE ID       CREATED         SIZE
nginx                         v2          1363f9428d14   12 hours ago    693MB
```
确认生成镜像文件之后，执行容器运行命令来启动来启动刚创建好的`nginx`镜像文件。
```bash
$ docker container run -d --name nginx02 -p 89:80 nginx:v2
```
配置容器的名称`--name nginx02`,配置容器对外映射的端口`-p 89:80`。通过docker ps命令查看这个容器是否启动成功：
```bash
$ docker ps

CONTAINER ID   IMAGE      COMMAND                  CREATED        STATUS        PORTS                               NAMES
7934c463c7d4   nginx:v2   "nginx -g 'daemon of…"   12 hours ago   Up 12 hours   0.0.0.0:89->80/tcp, :::89->80/tcp   nginx02
```
最后在宿主机上打开浏览器访问`http://{分配的虚拟ip}:89` 查看`nginx`镜像是否创建成功。

##### Harbor 镜像仓库

Harbor 是由VMWare公司开源的容器镜像仓库。事实上，Harbor是在Docker Register上进行了相应的企业级扩展，从而获得了更加广泛的应用，这些新的企业级特性包括：管理用户界面，基于角色的访问控制，确保镜像经过扫描且不存在漏洞，并将镜像标记为可信，帮助您跨 Kubernetes 和 Docker 等云原生计算平台一致、安全地管理工具。

它的特性包括：
- 安全：安全和漏洞分析
- 安全：内容签名和验证
- 管理：多租户
- 管理：可扩展的 API 和 Web UI
- 管理：跨多个注册表（包括 `Harbor`）进行复制
- 管理：身份集成和基于角色的访问控制

0. 依赖：docker-engine、docker-compose、openssl需要下载安装
1. 转至 `Harbor` 发布页面[下载](https://github.com/goharbor/harbor/releases)。
2. 下载您要安装的版本的在线或离线安装程序。
   - 在线安装程序：在线安装程序从 `Docker hub` 下载 `Harbor` 镜像。因此，安装程序的尺寸非常小。
   - 离线安装程序：如果部署 `Harbor` 的主机没有连接到 `Internet`，请使用离线安装程序。离线安装程序包含构建的镜像，因此它比在线安装程序大
3. 用于tar解压安装包
```bash
$ tar xzvf harbor-offline-installer-version.tgz
```
4. 配置
```bash
$ cd harbor
$ vi harbor.cfg
```
```
hostname = {ip}
ui_url_protocol = http
harbor_admin_password = ******
```
启动`harbor`运行脚本
```bash
$ ./install.sh
```

##### Prometheus 监控和警报系统

`Prometheus` 是一个监控和警报系统。它于 2012 年由 `SoundCloud` 开源，是继 `Kubernetes` 之后第二个加入并毕业的云原生计算基金会项目。`Prometheus` 将所有指标数据存储为时间序列，即指标信息与其记录的时间戳一起存储，称为标签的可选键值对也可以与指标一起存储。

度量是衡量的标准。我们想要测量的内容取决于应用程序的不同。对于 Web 服务器，它可以是请求时间，对于数据库，它可以是 CPU 使用率或活动连接数等。

指标在理解应用程序为何以某种方式运行方面发挥着重要作用。如果您运行一个 Web 应用程序，有人走到您面前并说该应用程序速度很慢。您将需要一些信息来了解您的应用程序发生了什么。例如，当请求数量较多时，应用程序可能会变慢。如果您有请求计数指标，您可以找出原因并增加服务器数量来处理重负载。每当您为应用程序定义指标时，您都必须加入侦探并问：**如果我的应用程序中出现任何问题，哪些信息对我调试很重要？**

Prometheus特点：
- 多维数据模型：有度量名称和键值对标识的时间序列数据
- `PromSQL`: 一种灵活的查询语言，可以利用多维数据完成复杂的查询
- 不依赖于分布式存储，单个服务节点可直接工作
- 基于`http`的`pull`方式采集时间序列数据
- 推送时间序列数据通过`PushGateway`组件支持
- 通过服务发现或静态配置发现
- 多种图形模式和仪表盘支持(`grafana`)

Prometheus组件：
`Prometheus` 系统由多个组件组成，其中许多组件是可选的：
- Prometheus 服务器，用于抓取和存储时间序列数据
- 用于检测应用程序代码的客户端库
- 支持短期工作的推送网关
- `HAProxy、StatsD、Graphite` 等服务的特殊用途 `exporters`。
- 处理警报的警报管理器
- 各种支持工具

大多数 `Prometheus` 组件都是用`Go`编写的，这使得它们很容易作为静态二进制文件构建和部署。

Prometheus架构

{% asset_img prometheus.png %}

`Prometheus` 直接或通过短期作业的中间推送网关从仪表化作业中获取指标。它在本地存储所有抓取的样本，并对这些数据运行规则，以聚合和记录现有数据的新时间序列或生成警报。`Grafana`或其他 API 使用者可用于可视化收集的数据。

`Prometheus` 非常适合记录任何纯数字时间序列。它既适合以机器为中心的监控，也适合高度动态的面向服务的架构的监控。在微服务的世界中，它对多维数据收集和查询的支持是一个特殊的优势。`Prometheus` 的设计注重可靠性，是您在中断期间可以使用的系统，以便您快速诊断问题。每个 `Prometheus` 服务器都是独立的，不依赖于网络存储或其他远程服务。当基础设施的其他部分损坏时，您可以依赖它，并且无需设置大量基础设施即可使用它。

Prometheus 容器安装
```bash
$ docker run -d --name prometheus-container -p 9090:9090 --mount type=bind,source=/tmp/prometheus.yml,target=/etc/prometheus/prometheus.yml ubuntu/prometheus:2.46.0-22.04_stable
```
 |     参数       |       说明     |
 |:--------------|:---------------|
 |`-e TZ=UTC`    |时区|
 |`-v /tmp/prometheus.yml:/etc/prometheus/prometheus.yml`|本地配置文件`prometheus.yml`|
 |`-v /path/to/alerts.yml:/etc/prometheus/alerts.yml`|本地警报配置文件`alerts.yml`|

进入容器调试：
```bash
$ docker exec -it prometheus-container /bin/bash
```

##### cAdvisor（Container Advisor）

为容器用户提供对其运行容器的资源使用情况和性能特征的了解。它是一个正在运行的守护进程，用于收集、聚合、处理和导出有关正在运行的容器的信息。具体来说，它为每个容器保留资源隔离参数、历史资源使用情况、完整历史资源使用情况的直方图和网络统计信息。

Docker 容器中运行 cAdvisor
```bash
$ docker run  \
     --volume=/:/rootfs:ro \
     --volume=/var/run:/var/run:ro  \
     --volume=/sys:/sys:ro   \
     --volume=/var/lib/docker/:/var/lib/docker:ro   \
     --volume=/dev/disk/:/dev/disk:ro   \
     --publish=8080:8080   \
     --detach=true   \
     --name=cadvisor  \
     google/cadvisor:latest
```
访问您的 cAdvisor `http://localhost:8080/metrics`：
```json
container_cpu_load_average_10s
{
   container_label_org_label_schema_build_date="",container_label_org_label_schema_license="",container_label_org_label_schema_name="",container_label_org_label_schema_schema_version="",container_label_org_label_schema_vendor="",container_label_org_opencontainers_image_base_digest="",container_label_org_opencontainers_image_created="",container_label_org_opencontainers_image_licenses="",container_label_org_opencontainers_image_ref_name="",container_label_org_opencontainers_image_title="",container_label_org_opencontainers_image_version="",id="/",image="",name=""
} 0
```

访问您的 Prometheus 实例`http://localhost:9090`：

我们可以查看采集的指标：
`container_cpu_cfs_periods_total` -> `container_cpu_cfs_periods_total{id="/aegis", instance="198.19.37.126:8080", job="docker"}`
`ontainer_cpu_load_average_10s` -> `container_cpu_load_average_10s{id="/", instance="198.19.37.126:8080", job="docker"}`

##### Grafana 监控仪表盘

无需构建、安装、维护和扩展可观察性堆栈的开销即可实现可观察性。Grafana Cloud 是开放且可组合的可观测平台。

Docker 容器中运行 Grafana：
```bash
$ docker container run -d --name grafana -p 3000:3000 grafana/grafana
```

访问您的 Prometheus 实例`http://localhost:3000`：
1. 选择数据源 -> Prometheus
2. 导入仪表盘库（193：容器实例监控仪表盘、195：宿主机实例监控仪表盘）

{% asset_img grafana.png %}
