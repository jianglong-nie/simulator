#ifndef FLOW_H
#define FLOW_H

#include <string>
#include <vector>
#include <utility>
#include "constants.h"

class Flow {
public:
    // 基本参数
    std::pair<int, int> src; // first表示server id，second表示gpu rank
    std::pair<int, int> dst;
    int srcId; // 在topo中的src节点编号
    int dstId; // 在topo中的dst节点编号
    float dataSize = 0;
    std::string protocol;

    // 待计算参数
    float rate = 0;
    std::vector<int> path;

    // Flow构造函数
    Flow() = default;

    // Flow构造函数，带参数
    Flow(std::pair<int, int> src, std::pair<int, int> dst, float dataSize, std::string protocol);

    // Flow的初始化函数，给Flow的每个成员变量都赋值
    void init(std::pair<int, int> src, std::pair<int, int> dst, float dataSize, std::string protocol);

    // 设置rate
    void setRate(float rate);

    // 设置path
    void setPath(std::vector<int> path);
};

#endif // FLOW_H
