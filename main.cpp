//
// Created by KWAZAR_ on 13.11.2024.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// Нормализация данных
void normalize(std::vector<std::vector<double>>& features) {
    size_t num_features = features[0].size();
    for (size_t j = 0; j < num_features; ++j) {
        double min_val = features[0][j];
        double max_val = features[0][j];
        for (const auto& sample : features) {
            min_val = std::min(min_val, sample[j]);
            max_val = std::max(max_val, sample[j]);
        }
        for (auto& sample : features) {
            sample[j] = (sample[j] - min_val) / (max_val - min_val + 1e-9); // Нормализация
        }
    }
}

// Нелинейная модель
double predict(const std::vector<double>& coefficients, const std::vector<double>& features) {
    double prediction = coefficients[0]; // Свободный член
    prediction += coefficients[1] * std::exp(std::min(coefficients[2] * features[0], 50.0)); // Ограничение экспоненты
    prediction += coefficients[3] * std::sin(coefficients[4] * features[1]);
    prediction += coefficients[5] * std::pow(features[2], 2);
    prediction += coefficients[6] * std::log(features[3] + 1.0); // Проверено, что features[3] >= -1
    prediction += coefficients[7] * features[4] * features[5];
    return prediction;
}

// Обучение модели с добавлением проверки значений
void train_model(std::vector<double>& coefficients, const std::vector<std::vector<double>>& features,
                 const std::vector<double>& targets, double learning_rate, int iterations) {
    size_t num_samples = features.size();
    size_t num_features = coefficients.size();

    for (int it = 0; it < iterations; ++it) {
        std::vector<double> gradients(num_features, 0.0);
        std::vector<double> predictions(num_samples);

        for (size_t i = 0; i < num_samples; ++i) {
            predictions[i] = predict(coefficients, features[i]);
        }

        for (size_t i = 0; i < num_samples; ++i) {
            double error = predictions[i] - targets[i];
            gradients[0] += error;
            gradients[1] += error * std::exp(std::min(coefficients[2] * features[i][0], 50.0));
            gradients[2] += error * coefficients[1] * features[i][0] * std::exp(std::min(coefficients[2] * features[i][0], 50.0));
            gradients[3] += error * std::sin(coefficients[4] * features[i][1]);
            gradients[4] += error * coefficients[3] * features[i][1] * std::cos(coefficients[4] * features[i][1]);
            gradients[5] += error * 2 * features[i][2];
            gradients[6] += error * 1.0 / (features[i][3] + 1.0);
            gradients[7] += error * features[i][4] * features[i][5];
        }

        for (size_t j = 0; j < num_features; ++j) {
            coefficients[j] -= (learning_rate / num_samples) * gradients[j];
        }

        if (it % 100 == 0) {
            double loss = 0.0;
            for (size_t i = 0; i < num_samples; ++i) {
                loss += std::pow(predict(coefficients, features[i]) - targets[i], 2);
            }
            std::cout << "Iteration " << it << ", Loss: " << loss / num_samples << std::endl;
        }
    }
}

int main() {
    // Пример данных
    std::vector<std::vector<double>> features = {
            {3.5, 1.2, 50, 30, 120, 6.5},
            {4.0, 1.1, 45, 35, 110, 6.8},
            {3.8, 1.3, 55, 25, 100, 6.7},
            {4.2, 1.4, 60, 40, 125, 6.6}
    };
    std::vector<double> targets = {40.0, 45.0, 43.0, 50.0};

    // Нормализация данных
    normalize(features);

    // Инициализация коэффициентов
    std::vector<double> coefficients(8, 0.1);

    // Обучение модели
    double learning_rate = 0.001;
    int iterations = 1000;
    train_model(coefficients, features, targets, learning_rate, iterations);

    /// Прогноз
    std::vector<double> new_features = {4.1, 1.2, 52, 38, 115, 6.7};
    std::vector<std::vector<double>> single_feature = {new_features};
    normalize(single_feature); // Нормализуем новые данные
    new_features = single_feature[0]; // Извлекаем нормализованные данные
    double predicted_yield = predict(coefficients, new_features);
    std::cout << "Predicted yield: " << predicted_yield << " Ц/га" << std::endl;


    return 0;
}
