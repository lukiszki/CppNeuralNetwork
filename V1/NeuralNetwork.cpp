#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>
#include <fstream>

using namespace std;

NeuralNetwork::NeuralNetwork(float learningRate, NeuralNetwork::ActivationFunction _function)
{
	NeuralNetwork::InitiateValues();
	learning_rate = learningRate;
    ActivFun = _function;
}
void NeuralNetwork::InitiateValues()
{
    for (int i = 0; i < inputSize * HiddenLayerSize; i++)
    {
        W1[i] = (float)(rand() % 1000 - 500) / 1000;
    }
    for (int i = 0; i < HiddenLayerSize*OutputSize; i++)
    {
        W2[i] = (float)(rand() % 1000 - 500) / 1000;
    }
    for (int i = 0; i < HiddenLayerSize; i++)
    {
        b1[i] = 1;
    }
    for(int i = 0; i < OutputSize; i++)
    { 
        b2[i] = 1;
    }
}
void NeuralNetwork::Forward_prop(Sample x)
{
    input[0] = x.input[0];
    input[1] = x.input[1];
    ClearArray(Z1,HiddenLayerSize);
    ClearArray(A1,HiddenLayerSize);
    ClearArray(Z2,OutputSize);
    ClearArray(A2,OutputSize);



    for (int i = 0; i < HiddenLayerSize; i++)
    {
        for (int j = 0; j < inputSize; j++)
        {
            Z1[i] += input[j] * W1[2 * i + j];

        }
        Z1[i] += b1[i];
        A1[i] = Activation(Z1[i]);

        
    }
    for(int i = 0; i < OutputSize; i++)
    {
        for(int j = 0; j < HiddenLayerSize; j++)
        {
            Z2[i] += A1[j] * W2[2 * i + j];
        }
        Z2[i] += b2[i];
        A2[i] = Activation(Z2[i]);
    }


}

void NeuralNetwork::ClearArray(float array[],int arrSize)
{
    for(int i= 0; i< arrSize; i++)
    {
        array[i] = 0;
    }
}

inline float NeuralNetwork::ReLU(float x)
{
    return x > 0 ? x : 0;
}
inline float NeuralNetwork::Sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}
inline float NeuralNetwork::Sigmoid_d(float x)
{
    return Sigmoid(x) * (1 - Sigmoid(x));
}
inline float NeuralNetwork::ReLU_d(float x)
{
    return x > 0;
}

float NeuralNetwork::Activation(float x)
{
    switch (ActivFun)
    {
    case NeuralNetwork::SIGMOID:
        return Sigmoid(x);
        break;
    case NeuralNetwork::RELU:
        return ReLU(x);
        break;
    default:
        return Sigmoid(x);
        break;
    }
}
float NeuralNetwork::Derivitive(float x)
{
    switch (ActivFun)
    {
    case NeuralNetwork::SIGMOID:
        return Sigmoid_d(x);
        break;
    case NeuralNetwork::RELU:
        return ReLU_d(x);
        break;
    default:
        return Sigmoid_d(x);
        break;
    }
}
void NeuralNetwork::PrintCost(vector<Sample> data)
{
    float totalCost = 0;
    float totalError = 0;
    float iter = 1;

    for (Sample point : data)
    {
        Forward_prop(point);
        float target[] = {point.output[0],point.output[1]};

        for(int i =0; i < OutputSize;i++)
        {
            totalCost += pow(A2[i] - target[i], 2);
            totalError += 1 - abs(A2[i] - target[i]);
            iter++;
        }

    }
    cout << "Cost: " << totalCost << "Precent Accuracy: " << totalError / iter << "%" << endl;
}
void NeuralNetwork::Back_prop(vector<Sample> data)
{
    for (Sample point : data)
    {
        for(int i = 0; i < OutputSize; i++)
        {
            Forward_prop(point);
            float target = point.output[i];

            float dcost_pred = 2 * (A2[i] - target);
            float dpred_dz2 = Derivitive(Z2[i]);

            float dz_dw2[HiddenLayerSize];
            for (int k = 0; k < HiddenLayerSize; k++)
            {
                dz_dw2[k] = A1[k*(i+1)];
            }

            float dcost_dz2 = dcost_pred * dpred_dz2;

            float dcost_dw2[HiddenLayerSize];
            for (int k = 0; k < HiddenLayerSize; k++)
            {
                dcost_dw2[k] = dcost_dz2 * dz_dw2[k];
                W2[k*(i+1)] -= learning_rate * dcost_dw2[k];
            }

            float dcost_db2 = dcost_dz2;
            b2[i] -= learning_rate * dcost_db2;


            for (int k = 0; k < HiddenLayerSize; k++)
            {
                float dcost_pred1 = W2[k*(i+1)] * dcost_dz2;
                float dpred1_dz1 = Derivitive(Z1[k]);

                float dz_dw1[inputSize];
                for (int l = 0; l < inputSize; l++)
                {
                    dz_dw1[l] = input[l];
                }
                float dz_db1 = 1;

                float dcost_dz1 = dcost_pred1 * dpred1_dz1;

                float dcost_dw1[inputSize];
                for (int l = 0; l < inputSize; l++)
                {
                    dcost_dw1[l] = dcost_dz1 * dz_dw1[l];
                    W1[l * (k + 1)] -= learning_rate * dcost_dw1[l];
                }
                float dcost_db1 = dz_db1 * dcost_dz1;
                b1[k] -= learning_rate * dcost_db1;
            }
        }


    }
}
void NeuralNetwork::SaveValues()
{
    fstream file;
    file.open("values.txt", ios::out);
    for(float val:W1)
    {
        file << val << endl;
    }
    for (float val : W2)
    {
        file << val << endl;
    }
    for (float val : b1)
    {
        file << val << endl;
    }
    for (float val : b1)
    {
        file << val << endl;
    }
    

    file.close();
}
void NeuralNetwork::LoadValues()
{
    fstream file;
    file.open("values.txt", ios::in);

    if (!file.good())
    {
        cout << "File with values does not exist!" << endl;
        return;
    }
    string line;
    int index = 1;
    while (getline(file, line))
    {
        if (index>0&&index<=inputSize * HiddenLayerSize)
        {
            W1[index - 1] = stof(line);
        }
        if (index > inputSize * HiddenLayerSize && index <= (inputSize * HiddenLayerSize)+ (HiddenLayerSize*OutputSize))
        {
            W2[(index - 1)- (inputSize * HiddenLayerSize)] = stof(line);
        }
        if (index > ((inputSize * HiddenLayerSize) + (HiddenLayerSize*OutputSize)) && index <= ((inputSize * HiddenLayerSize) + (HiddenLayerSize*OutputSize)*2))
        {
            b1[(index - 1)- ((inputSize * HiddenLayerSize) + HiddenLayerSize)] = stof(line);
        }
        if(index >= ((inputSize * HiddenLayerSize) + (HiddenLayerSize*OutputSize) * 2)+1)
        {
            b2[index-1] = stof(line);
        }

        index++;
    }

}
