#pragma once
#include "Sample.h"
#include <vector>
class NeuralNetwork
{
    float learning_rate = 0.05;

    const static int inputSize = 2;
    const static int HiddenLayerSize = 3;
    const static int OutputSize = 2;



public:


    enum ActivationFunction { SIGMOID, RELU };

private:
    
    ActivationFunction ActivFun;
public:

    //NEURONS VALUES
    float input[inputSize];
    float A1[HiddenLayerSize];
    float A2[OutputSize];

    //Sums
    float Z1[HiddenLayerSize];
    float Z2[OutputSize];

    //WEIGHTS VALUES
    float W1[inputSize * HiddenLayerSize];
    float W2[HiddenLayerSize * OutputSize];

    //BIASES VALUES
    float b1[HiddenLayerSize];
    float b2[OutputSize];

    NeuralNetwork(float learningRate, NeuralNetwork::ActivationFunction _function);

    void InitiateValues();

    void Forward_prop(Sample x);

    inline float ReLU(float x);

    float Activation(float x);

    float Derivitive(float x);

    inline float Sigmoid(float x);

    void Back_prop(std::vector <Sample> data);

    void PrintCost(std::vector<Sample> data);

    inline float Sigmoid_d(float x);

    inline float ReLU_d(float x);
	
	void SaveValues();
    
    void LoadValues();
    private:
    void ClearArray(float array[],int arrSize);
};

