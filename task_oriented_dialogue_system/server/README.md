# 人机交互服务

该服务作为B榜人工交互式评测环节使用



## 代码实现

- 参考`server.py`中的`kefuBertHandler`。该代码作为人机测试的子重要模块，需要维护对话状态。


- 需要注意的是，完整的对话记录由外部模块记录。每次调用当前子模块的时候，会将完整的对话记录通过`context`字段传入，传入格式为`[U_0, S_0, U_1, S_1, ...., U_n]`其中`U`代表用户输入，`S`代表系统输出，通常`U_0`为空字符串，`context`中每个元素都是字符串(没有经过分词)。


- 当前子模块只要将系统的文本输出通过`answer`字段输出，外部模块就会自动记录对话历史。

- 其他信息通过`private_memory`维护，需要注意的是，`private_memory`在输出前需要成功经过`json.dumps`的处理。

  ​



## 启动服务

```
cd task_oriented_dialogue_system/task_oriented_dialogue_system/server
bash run.sh
```



## 服务说明

- 接口样例

```
http://10.244.22.235:12595/kefu?context=["您好,我是在那个手机上的时候,它手机上面那个就是那个是什么都没了"]&private_memory={}
```

- 返回结构样例

```
{
    "status_code": 0,
    "status_msg": "ok",
    "answer": "哦?是来电手机号码绑定的账号吗",   # 模型回答
    "private_memory": "{\"is_begin\": true}",  
    "private_memory_drop": ""
}
```

- 参数说明

  - context : 传入格式为`[U_0, S_0, U_1, S_1, ...., U_n]`其中`U`代表用户输入，`S`代表系统输出，通常`U_0`为空字符串，`context`中每个元素都是字符串(没有经过分词)。

  - private_memory: json格式，具体作用见上文，传入上一次对话返回的`private_memory`内容

    ​

    ​