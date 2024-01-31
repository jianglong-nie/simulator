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

    // 有关滑动窗口的变量
    float len; // len = dataSize;
    float Cnvl;
    float Cnet;
    float Wnvl;
    float Wnet;
    float left = -1;
    float right = -1;
    float timeChunkZero = FLOAT_MAX; // Time of the zero chunk
    float timeChunkNow = 0; // Current time
    float timeChunkLast = timeChunkZero; // Last time a chunk was sent
    float alpha; // Alpha value for adjusting window size
    float delta; // Delta value for adjusting window size

    // Server构造函数
    Server(int id, int gpuNum, float gpuDataSize, std::vector<std::vector<float>> NVLink);

    // Server的初始化函数，给Server的每个成员变量都赋值
    void init(int id, int gpuNum, float gpuDataSize, std::vector<std::vector<float>> NVLink);

    //对滑动窗口的初始化
    void initWindow(float len, float Cnvl, float Cnet, float Wnvl, float Wnet, float alpha, float delta);
    
    bool allRanksFinish(std::string protocol);

    void computingNonCC(float unitTime);
    
    void computing(float unitTime);

    //
    void step(float unitTime);

    //
    void control(float unitTime);
};

#endif // SERVER_H