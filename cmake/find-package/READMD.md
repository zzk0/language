# reference

https://zhuanlan.zhihu.com/p/97369704

find\_package 有两种模式的区别：module 和 config。

find\_package 本质上是跑一段预先写好的代码，在代码里找到库和头文件，并设置到对应的变量里，就这样了。

所以如果自己写 find\_package 那么就是设置一些变量。
