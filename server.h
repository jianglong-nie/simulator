#ifndef SERVER_H
#define SERVER_H

#include <vector>
#include "gpu.h"

class Server {
public:
    int id;
    int gpuNum = 8;
    std::vector<GPU> gpus; // Vector of GPU objects
    std::vector<int> ring = {1, 2, 3, 4, 5, 6, 7, 0};
    std::vector<std::vector<float>> NVLink;

    // Server构造函数
    Server(int id, int gpuNum, float gpuDataSize, std::vector<std::vector<float>> NVLink);

    // Server的初始化函数，给Server的每个成员变量都赋值
    void init(int id, int gpuNum, float gpuDataSize, std::vector<std::vector<float>> NVLink);

    //
    void step(float unitTime);

    //
    void control(float unitTime);
};

#endif // SERVER_H