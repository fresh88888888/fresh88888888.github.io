---
title: S3 架构原理
date: 2023-12-22 21:20:41
tags:
  - S3
category:
  - other
---

当你上传一个文件到`S3`服务器会发生什么？ 让我们来看一下`S3`对象存储系统是如何工作的？

{% asset_img s3-design1.png %}

<!-- more -->

**桶**：对象的逻辑容器，桶的名称是全局唯一的。要将数据上传到`S3`, 首先必须创建一个桶。
**对象**： 对象是我们存储在桶中的单个数据，它包含对象数据（也称为有效载荷）和元数据，对象数据可以是我们想要存储的任何字节序列，元数据是一组描述对象的键-值对。

一个`S3`对象包括：
- **元数据**：它是可变的，包括`ID`、桶名称、对象名称等属性。
- **对象数据**：它是不可变的，包含实际数据。

在`S3`中，对象驻留在桶中，桶只有元数据；而对象有元数据和实际数据。图二中阐述了文件上传的工作原理，在本例中，我们首先创建一个名称为`bucket-to-share`的桶，然后将一个名为`script.txt`上传到存储桶中。

1. 客户端发送一个`HTTP PUT`请求去创建一个名为`bucket-to-share`的桶，请求被转发到`API`服务。
2. `API`服务调用`IAM`系统(身份和访问管理系统)，以确保用户获得授权并具有`WRITE`权限。
3. `API`服务调用元数据存储，在元数据的数据库中创建一条包含桶信息的条目。创建条目后，将向客户端返回成功的消息。
4. 创建桶后，客户端发送`HTTP PUT`请求，创建一个名为`script.txt`的对象。
5. `API`服务验证用户的身份，并确保用户在存储桶上拥有写入权限。
6. 一旦验证成功，`API`服务将`HTTP PUT`有效载荷中的对象数据发送到存储节点，存储节点将有效载荷保存为对象，并返回对象的`UUID`。
7. API服务调用元数据存储，在元数据的数据库中创建一个新条目，它包含重要的元数据信息，如`object_id(UUID)`、 `bucket_id`（对象属于那个桶）、`object_name`等。

{% asset_img s3-design2.png %}