# AI系统实践

解决一个基于ai的问题，需要经过以下的步骤：
数据获取 -> 数据预处理 -> 建模与调参 -> 系统部署 ->->->持续维护

## 网络爬虫
1.确定爬取链接
2.读取链接指向的内容
3.从中抽取关键元素

网络爬虫可以基于API(json)或者基于网页(html)爬取。
可以在python中使用selenium包访问网页，也可以使用requests包访问静态网页。

注意点：
* 设置参数模拟真实浏览器
* 请求频率不要过高
* **请求被拒绝后更换ip**

