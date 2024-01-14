#ifndef FLOW_H
#define FLOW_H

#include <string>
#include <vector>
#include "constants.h"

class Flow {
public:
    // 基本参数
    int src;
    int dst;
    int dataSize;
    std::string protocol;

    // 待计算参数
    int rate;
    std::vector<int> path;    

    // Flow构造函数
    Flow() = default;

    // Flow构造函数，带参数
    Flow(int src, int dst, int dataSize, std::string protocol);

    // Flow的初始化函数，给Flow的每个成员变量都赋值
    void init(int src, int dst, int dataSize, std::string protocol);

    // 设置rate
    void setRate(int rate);

    // 设置path
    void setPath(std::vector<int> path);
};

#endif // FLOW_H
