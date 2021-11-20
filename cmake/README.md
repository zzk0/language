# CMake 指令表

## get\_filename\_component 

从 FileName 中根据 mode 获取信息，比如路径、名字、扩展名等。

```
get_filename_component(<var> <FileName> <mode> [CACHE])
get_filename_component(<VAR> FileName
                         PATH|ABSOLUTE|NAME|EXT|NAME_WE|REALPATH
                         [CACHE])

message(${CMAKE_CURRENT_LIST_FILE})  # 当前 CMakeLists.txt 的路径
get_filename_component(TXT "${CMAKE_CURRENT_LIST_FILE}" EXT)
message(${TXT})  # 输出 .txt
```

